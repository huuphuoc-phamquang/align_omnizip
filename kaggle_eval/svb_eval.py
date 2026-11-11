"""ShortVid-Bench worker: full-token Qwen2.5-Omni vs OmniZip, one process per GPU.

Run by the Kaggle notebook (one subprocess per GPU), or directly:
    python svb_eval.py --config run_config.json --shard 0 --num-shards 2

Every finished sample is appended to <results_dir>/<run_tag>__<method>__shard<k>.jsonl, so a
restarted run skips what is already there. Helpers below the worker (load_records, summarize)
have no torch/transformers import and are reused by the notebook's summary cell.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
import traceback

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]


def extract_answer(resp_text):
    """Same rule as eval/eval_shortvideobench.py, so scores stay comparable with the repo script."""
    resp_text = (resp_text or "").strip()
    for opt in OPTION_LETTERS:
        if resp_text.upper().startswith(opt):
            return opt
    return resp_text[0].upper() if resp_text else None


def qid(r):
    """Question id: WorldSense has several questions per video; ShortVid-Bench one, keyed by the video."""
    return r.get("qid") or r["video"]


def result_path(results_dir, run_tag, method, shard):
    return os.path.join(results_dir, f"{run_tag}__{method}__shard{shard}.jsonl")


def done_videos(result_roots, run_tag, method):
    """Question ids (qid) already scored for (run_tag, method) in this session or in restored inputs."""
    done = set()
    for root in result_roots:
        for p in glob.glob(os.path.join(root, "**", f"{run_tag}__{method}__shard*.jsonl"), recursive=True):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # a line cut off by a killed session; the sample is simply redone
                        if not r.get("error"):  # failed samples are retried on the next start
                            done.add(r.get("qid", r["video"]))
    return done


def select_samples(samples, mini_test_num, shard, num_shards, videos=None):
    if videos:  # explicit list (e.g. re-running failed videos); idx stays the sample's index in the 1k set
        want = set(videos)
        picked = [(i, s) for i, s in enumerate(samples) if s["video"] in want]
    else:
        picked = list(enumerate(samples[:mini_test_num] if mini_test_num else samples))
    return [p for j, p in enumerate(picked) if j % num_shards == shard]


# ----------------------------------------------------------------------------- T4 workarounds

def segment_sdpa(q, k, v, cu_seqlens):
    """Attention restricted to each [cu[i-1], cu[i]) segment; q, k, v: (S, H, D) -> (S, H, D).

    Same result as SDPA with the dense block-diagonal (1, S, S) mask the vision encoder builds,
    without that mask: on T4 (no FlashAttention) the masked call falls back to a kernel that
    materialises H x S x S scores and runs out of memory at ~32k patches (128 frames). Segments of
    equal length are batched into one call.
    """
    import torch
    import torch.nn.functional as F
    out = torch.empty_like(q)
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else list(cu_seqlens)
    by_len = {}
    for a, b in zip(cu[:-1], cu[1:]):
        if b > a:
            by_len.setdefault(b - a, []).append(a)
    for length, starts in by_len.items():
        idx = torch.tensor(starts, device=q.device)[:, None] + torch.arange(length, device=q.device)  # (n, L)
        o = F.scaled_dot_product_attention(q[idx].transpose(1, 2), k[idx].transpose(1, 2), v[idx].transpose(1, 2))
        out[idx] = o.transpose(1, 2)
    return out


def patch_vision_sdpa(module):
    """Swap Qwen2_5OmniVisionSdpaAttention.forward in `module` for the segment-wise version (same math)."""
    rope = module.apply_rotary_pos_emb_vision

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = rope(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = rope(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        attn_output = segment_sdpa(q, k, v, cu_seqlens).reshape(seq_length, -1)
        return self.proj(attn_output)

    module.Qwen2_5OmniVisionSdpaAttention.forward = forward


def patch_audio_sdpa(module):
    """Same segment-wise attention for the audio encoder, whose sdpa path also builds a dense (1, S, S) mask
    and ran out of memory on long audio (~4k audio tokens). The OmniZip copy also returns its audio scores,
    computed exactly as before (whole sequence, unmasked); the transformers copy returns only the output."""
    import inspect
    cls = module.Qwen2_5OmniAudioSdpaAttention
    returns_scores = "return_logits" in inspect.signature(cls.forward).parameters

    def forward(self, hidden_states, cu_seqlens=None, return_logits=False, **kwargs):
        seq_length = hidden_states.shape[0]
        q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        out = self.out_proj(segment_sdpa(q, k, v, cu_seqlens).reshape(seq_length, self.embed_dim))
        if not returns_scores:
            return out
        if not return_logits:
            return out, None, None
        q, k = q.transpose(0, 1), k.transpose(0, 1)  # [H, T, D], as the original sdpa path
        return out, module._audio_token_importance(q, k), k

    cls.forward = forward


def make_last_token_head(head):
    """lm_head that only projects the last position. Prefill otherwise builds logits for the whole sequence
    (11k tokens x 152k vocab x fp16 = 3.2 GB, the OOM of long videos in `full`); greedy generate() reads only
    logits[:, -1], so answers are unchanged."""
    import torch

    class LastTokenHead(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, hidden_states):
            return self.head(hidden_states[:, -1:, :])

    return LastTokenHead(head)


class StageTimer:
    """CUDA-synchronised wall time and peak memory per stage of one sample (vision, audio, omnizip, LLM...).

    Hooks synchronise before and after each stage, so stages do not overlap; the peak memory of the gaps
    between stages is folded into `peak_gb` before each reset."""

    def __init__(self, torch):
        self.torch = torch
        self.reset()

    def reset(self):
        self.t, self.peak, self.n, self.peak_gb, self.llm_tokens, self.ttft_at = {}, {}, {}, 0.0, None, None

    def begin(self):
        tc = self.torch.cuda
        tc.synchronize()
        self.peak_gb = max(self.peak_gb, tc.max_memory_allocated() / 2**30)
        tc.reset_peak_memory_stats()
        return time.perf_counter()

    def end(self, name, t0):
        tc = self.torch.cuda
        tc.synchronize()
        p = tc.max_memory_allocated() / 2**30
        self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0
        self.peak[name] = max(self.peak.get(name, 0.0), p)
        self.n[name] = self.n.get(name, 0) + 1
        self.peak_gb = max(self.peak_gb, p)

    def hook(self, module, name, llm=False):
        state = {}

        def pre(m, args, kwargs):
            if llm and self.llm_tokens is None:  # first decoder call of the sample = prefill
                x = kwargs.get("inputs_embeds")
                self.llm_tokens = int(x.shape[1]) if x is not None else None
            state["t0"] = self.begin()

        def post(m, args, kwargs, out):
            stage = ("llm_prefill" if "llm_prefill" not in self.t else "llm_decode") if llm else name
            self.end(stage, state["t0"])
            if name == "lm_head" and self.ttft_at is None:
                self.ttft_at = time.perf_counter()

        module.register_forward_pre_hook(pre, with_kwargs=True)
        module.register_forward_hook(post, with_kwargs=True)

    def record(self):
        r = {f"t_{k}": v for k, v in self.t.items()}
        r.update({f"peak_{k}_gb": v for k, v in self.peak.items()})
        r["n_decode_steps"] = self.n.get("llm_decode", 0)
        r["n_llm_tokens"] = self.llm_tokens
        return r


def read_video_pyav(ele):
    """PyAV reader with qwen-omni-utils' sampling; stands in for its torchvision fallback,
    whose torchvision.io.read_video no longer exists in the torchvision Kaggle ships."""
    import av
    import numpy as np
    import torch
    from qwen_omni_utils.v2_5 import vision_process as vp
    path = ele["video"]
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or 30)
        total = sum(1 for p in c.demux(s) if p.size > 0)
    nframes = vp.smart_nframes(ele, total_frames=total, video_fps=fps)
    idx = torch.linspace(0, total - 1, nframes).round().long().tolist()
    want, got = set(idx), {}
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = f.to_ndarray(format="rgb24")
            if i >= idx[-1]:
                break
    assert got, f"PyAV decoded no frames from {path}"
    have = sorted(got)
    frames = [got[j] if j in got else got[min(have, key=lambda h: abs(h - j))] for j in idx]
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # TCHW, like the other readers
    return video, nframes / max(total, 1e-6) * fps


def rss_gb():
    """Resident memory of this process (Linux); nan elsewhere."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return float("nan")


def trim_memory():
    # Decoding 128 full-HD frames per sample leaves freed GB in glibc arenas; hand them back to the OS
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def mmr_select(scores, feats, k, lam):
    """Greedy MMR in every frame at once: scores [T, P], feats [T, P, D] -> kept indices [T, k].

    i* = argmax_{i not in S} lam * s_i - (1 - lam) * max_{j in S} cos(x_i, x_j), with s min-max normalised per frame
    (scores and cosines on one scale). lam = 1 is the plain per-frame top-k of the score.
    """
    import torch
    s = scores.float()
    lo, hi = s.amin(1, keepdim=True), s.amax(1, keepdim=True)
    s = (s - lo) / (hi - lo).clamp_min(1e-6)
    x = torch.nn.functional.normalize(feats.float(), dim=-1)
    sim = x @ x.transpose(1, 2)                      # [T, P, P]
    T = s.shape[0]
    rows = torch.arange(T, device=s.device)
    maxsim = torch.full_like(s, -1.0)               # empty S: same penalty for every token, so step 1 = argmax s
    picked = torch.zeros_like(s, dtype=torch.bool)
    out = torch.empty(T, k, dtype=torch.long, device=s.device)
    for step in range(k):
        gain = (lam * s - (1 - lam) * maxsim).masked_fill(picked, float("-inf"))
        i = gain.argmax(1)
        out[:, step] = i
        picked[rows, i] = True
        maxsim = torch.maximum(maxsim, sim[rows, :, i])
    return out


def av_align_select(video_feature, audio_feature, num_tokens_per_frame, merging_ratio, aligner, mmr_lambda):
    """omnizip_core.omnizip_av_align with its per-frame top-k replaced by MMR over the same AV_Aligner scores."""
    import torch
    P = num_tokens_per_frame
    T = video_feature.shape[0] // P
    mask = torch.zeros(video_feature.shape[0], dtype=torch.bool, device=video_feature.device)
    k = int(P * (1.0 - merging_ratio))
    if T == 0 or k >= P:
        mask[: T * P] = True
        return mask
    frames = video_feature[: T * P].reshape(T, P, -1)
    aud = audio_feature.to(frames.dtype).unsqueeze(0).expand(T, -1, -1)
    with torch.no_grad():
        scores = aligner.token_scores(aud, frames)  # [T, P]
        keep = mmr_select(scores, frames, k, mmr_lambda)
    mask[(keep + torch.arange(T, device=mask.device)[:, None] * P).reshape(-1)] = True
    return mask


def save_viz(results_dir, method, video, frames, grid_thw, video_keep, n_show=4):
    """4 frames + the kept-video-token mask (t, h/2, w/2), for the notebook's overlay plot."""
    import numpy as np
    t, h, w = (int(x) for x in grid_thw.tolist())
    keep = (np.ones(t * (h // 2) * (w // 2), bool) if video_keep is None else video_keep).reshape(t, h // 2, w // 2)
    tp = np.linspace(0, t - 1, n_show).round().astype(int)  # temporal patch index; it covers frames 2tp, 2tp+1
    imgs = frames[[min(2 * i, frames.shape[0] - 1) for i in tp]].float().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    os.makedirs(os.path.join(results_dir, "viz"), exist_ok=True)
    np.savez_compressed(os.path.join(results_dir, "viz", f"{method}__{video}.npz"),
                        frames=imgs, tp=tp, keep=keep[tp], grid=np.array([t, h, w]))


EXIT_HIGH_RSS = 75  # worker stopped itself between samples; the notebook restarts it and resume skips done samples


# ----------------------------------------------------------------------------- worker

def run_worker(cfg, shard, num_shards, methods):
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", cfg["video_reader"])
    sys.path.insert(0, cfg["code_root"])

    import numpy as np
    import torch
    from transformers import Qwen2_5OmniProcessor
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as hf_modeling
    import omnizip.modeling_qwen2_5_omni as oz_modeling
    from qwen_omni_utils import process_mm_info
    from qwen_omni_utils.v2_5 import vision_process
    from qwen_omni_utils.v2_5.audio_process import _check_if_video_has_audio

    # Both methods get the same vision-encoder fix, so the comparison stays like for like
    for m in (hf_modeling, oz_modeling):
        patch_vision_sdpa(m)
        if cfg.get("fix_audio_sdpa"):
            patch_audio_sdpa(m)
    vision_process.VIDEO_READER_BACKENDS["torchvision"] = read_video_pyav

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    with open(cfg["data_json"], encoding="utf-8") as f:
        samples = json.load(f)
    todo_all = select_samples(samples, cfg["mini_test_num"], shard, num_shards, cfg.get("videos"))
    viz_videos = set(cfg.get("viz_videos") or [])
    os.makedirs(cfg["results_dir"], exist_ok=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(cfg["model_id"])
    dtype = getattr(torch, cfg["dtype"])

    for method in methods:
        mcfg = cfg["methods"][method]
        done = done_videos(cfg["result_roots"], cfg["run_tag"], method)
        todo = [(i, s) for i, s in todo_all if qid(s) not in done]
        print(f"[shard{shard}] {method}: {len(todo_all) - len(todo)} done, {len(todo)} to go", flush=True)
        if not todo:
            continue

        if mcfg["wrapper"] == "omnizip":
            from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
        else:
            from transformers import Qwen2_5OmniForConditionalGeneration
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=dtype, device_map="cuda:0", attn_implementation=cfg["attn_implementation"],
        )
        model.disable_talker()  # text answers only; frees the talker/token2wav weights
        model.eval()
        model.thinker.omnizip_config = (
            {k: mcfg[k] for k in ("rho_audio", "rho_video", "g", "contextual_ratio")}
            if mcfg["wrapper"] == "omnizip" else None
        )

        if cfg.get("fix_lm_head"):
            model.thinker.lm_head = make_last_token_head(model.thinker.lm_head)
        timer = StageTimer(torch) if cfg.get("profile") else None
        if timer:
            th = model.thinker
            for mod, name in ((th.visual, "vision"), (th.audio_tower, "audio"), (th.lm_head, "lm_head")):
                timer.hook(mod, name)
            timer.hook(th.model, "llm", llm=True)

        tcfg = model.config.thinker_config
        audio_id = getattr(tcfg, "audio_token_index", None) or getattr(tcfg, "audio_token_id")
        video_id = getattr(tcfg, "video_token_index", None) or getattr(tcfg, "video_token_id")

        # Record what OmniZip actually kept; the thinker looks the function up on every call
        last = {}
        if mcfg["wrapper"] == "omnizip":
            import omnizip.omnizip_core as oc
            orig = getattr(oc, "_orig_omnizip", oc.omnizip)
            oc._orig_omnizip = orig

            def recorded(*args, **kw):
                t_oz = timer.begin() if timer else None
                embeds, mask = orig(*args, **kw)
                if timer:
                    timer.end("omnizip", t_oz)
                ids = kw["input_ids"].reshape(-1)
                last.update(
                    kept_audio=int(mask[ids == kw["audio_token_id"]].sum()),
                    kept_video=int(mask[ids == kw["video_token_id"]].sum()),
                    n_audio_scores=int(kw["attn_logits"].numel()),
                )
                if last.get("viz"):
                    last["video_keep"] = mask[ids == kw["video_token_id"]].cpu().numpy()
                return embeds, mask
            oc.omnizip = recorded

            orig_align = getattr(oc, "_orig_av_align", oc.omnizip_av_align)
            oc._orig_av_align = orig_align

            def istm_only(video_feature, audio_feature, num_tokens_per_frame, merging_ratio, aligner):
                # Paper's ISTC, as in upstream KD-TAO/OmniZip: same groups and ratios, AV_Aligner bypassed
                last["n_istm_groups"] = last.get("n_istm_groups", 0) + 1
                return oc.omnizip_istm(video_feature, num_tokens_per_frame, merging_ratio)

            def av_align_mmr(video_feature, audio_feature, num_tokens_per_frame, merging_ratio, aligner):
                last["n_mmr_groups"] = last.get("n_mmr_groups", 0) + 1
                return av_align_select(video_feature, audio_feature, num_tokens_per_frame, merging_ratio, aligner,
                                       mmr_lambda=mcfg.get("mmr_lambda", 0.5))
            oc.omnizip_av_align = {"istm": istm_only, "av_aligner_mmr": av_align_mmr}.get(
                mcfg.get("video_compressor"), orig_align)

        out_path = result_path(cfg["results_dir"], cfg["run_tag"], method, shard)
        with open(out_path, "a", encoding="utf-8") as out:
            for n_done, (idx, s) in enumerate(todo, 1):
                last.clear()
                last["viz"] = s["video"] in viz_videos
                if timer:
                    timer.reset()
                rec = {"run_tag": cfg["run_tag"], "method": method, "idx": idx, "qid": qid(s), "video": s["video"],
                       "problem_type": s["problem_type"], "answer": s["answer"], "error": None}
                video_path = os.path.join(cfg["video_dir"], s["video"])
                prompt = (f"{s['question']}\nOptions:\n" + "\n".join(s["candidates"]) +
                          "\nAnswer with the option's letter from the given choices directly.")
                conversation = [
                    {"role": "system", "content": [{"type": "text", "text": cfg["system_prompt"]}]},
                    {"role": "user", "content": [
                        {"type": "video", "video": video_path, "fps": cfg["fps"],
                         "max_frames": cfg["max_frames"], "max_pixels": cfg["video_max_pixels"]},
                        {"type": "text", "text": prompt},
                    ]},
                ]
                try:
                    t0 = time.time()
                    use_audio = _check_if_video_has_audio(video_path)
                    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
                    t_read = time.time()
                    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
                                       padding=True, use_audio_in_video=use_audio)
                    t_proc = time.time()
                    inputs = inputs.to(model.device).to(model.dtype)
                    torch.cuda.synchronize()
                    rec.update(t_read_decode=t_read - t0, t_processor=t_proc - t_read, t_to_gpu=time.time() - t_proc)
                    ids = inputs["input_ids"]
                    n_audio = int((ids == audio_id).sum()); n_video = int((ids == video_id).sum())
                    # recorded before generate() so a failed sample still says what it was
                    rec.update(has_audio=use_audio, n_frames=int(videos[0].shape[0]), n_input_tokens=int(ids.shape[1]),
                               n_audio_tokens=n_audio, n_video_tokens=n_video)
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    t1 = time.time()
                    g0 = time.perf_counter()  # same clock as StageTimer.ttft_at
                    with torch.no_grad():
                        out_ids = model.generate(
                            **inputs, use_audio_in_video=use_audio, return_audio=False,
                            thinker_do_sample=False, thinker_max_new_tokens=cfg["max_new_tokens"],
                            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
                        )
                    torch.cuda.synchronize()
                    t2 = time.time()
                    resp = processor.batch_decode(out_ids[:, ids.shape[1]:], skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0].strip()
                    pred = extract_answer(resp)
                    kept_a = last.get("kept_audio", n_audio); kept_v = last.get("kept_video", n_video)
                    rec.update(
                        pred=pred, correct=pred == s["answer"], response=resp, kept_audio=kept_a, kept_video=kept_v,
                        retained_av_ratio=(kept_a + kept_v) / max(n_audio + n_video, 1),
                        n_audio_scores=last.get("n_audio_scores"), n_istm_groups=last.get("n_istm_groups"),
                        n_mmr_groups=last.get("n_mmr_groups"),
                        n_new_tokens=int(out_ids.shape[1] - ids.shape[1]),
                        prep_seconds=t1 - t0, gen_seconds=t2 - t1,
                        peak_mem_gb=max(torch.cuda.max_memory_allocated() / 2**30, timer.peak_gb if timer else 0),
                    )
                    if timer:
                        rec.update(timer.record(), ttft_seconds=timer.ttft_at - g0 if timer.ttft_at else None)
                    if last["viz"]:
                        save_viz(cfg["results_dir"], method, s["video"], videos[0], inputs["video_grid_thw"][0],
                                 last.get("video_keep"))
                except Exception as e:  # one bad video must not stop the shard; it is scored as wrong
                    rec.update(pred=None, correct=False, response=None,
                               error=f"{type(e).__name__}: {e}".splitlines()[0][:500],
                               traceback=traceback.format_exc()[-4000:],
                               peak_mem_gb=max(torch.cuda.max_memory_allocated() / 2**30, timer.peak_gb if timer else 0))
                    if timer:
                        rec.update(timer.record())  # stages reached before the failure
                    traceback.print_exc()
                    torch.cuda.empty_cache()
                inputs = out_ids = audios = images = videos = None
                trim_memory()
                rec["rss_gb"] = rss_gb()
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush(); os.fsync(out.fileno())
                if n_done % 10 == 0 or n_done == len(todo):
                    print(f"[shard{shard}] {method}: {n_done}/{len(todo)} rss={rec['rss_gb']:.1f}GB", flush=True)
                if rec["rss_gb"] > cfg.get("max_rss_gb", float("inf")):
                    print(f"[shard{shard}] rss {rec['rss_gb']:.1f}GB > {cfg['max_rss_gb']}GB, exiting for a restart", flush=True)
                    sys.exit(EXIT_HIGH_RSS)

        # Free this model before the next method loads one: `th`/`mod` (profiling hooks), the last `ids`, and the cached
        # AV_Aligner (it shares layer-0 modules) would otherwise keep ~7 GB of it on the GPU
        model = th = mod = ids = None
        if mcfg["wrapper"] == "omnizip":
            oc._AV_ALIGNERS.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[shard{shard}] {method}: done, GPU still allocated {torch.cuda.memory_allocated() / 2**30:.2f} GB",
              flush=True)


# ----------------------------------------------------------------------------- summary helpers

def load_records(result_roots, run_tag):
    """All records of run_tag, one per (method, qid): first non-error record wins."""
    best = {}
    for root in result_roots:
        for p in sorted(glob.glob(os.path.join(root, "**", f"{run_tag}__*__shard*.jsonl"), recursive=True)):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = (r["method"], qid(r))
                    if key not in best or (best[key]["error"] and not r["error"]):
                        best[key] = r
    return list(best.values())


def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_compare(recs, base, other, n_boot=2000, seed=0):
    """Accuracy difference other - base on the questions both methods scored, with McNemar p and bootstrap CI."""
    a = {qid(r): bool(r["correct"]) for r in recs if r["method"] == base}
    o = {qid(r): bool(r["correct"]) for r in recs if r["method"] == other}
    vids = sorted(a.keys() & o.keys())
    if not vids:
        return None
    x = [a[v] for v in vids]; y = [o[v] for v in vids]
    b = sum(1 for p, q in zip(x, y) if p and not q)   # base right, other wrong
    c = sum(1 for p, q in zip(x, y) if q and not p)   # other right, base wrong
    diff = (sum(y) - sum(x)) / len(vids)
    rng = random.Random(seed)
    n = len(vids)
    boots = sorted(sum(y[j] - x[j] for j in (rng.randrange(n) for _ in range(n))) / n for _ in range(n_boot))
    return {"base": base, "other": other, "n_paired": n, "acc_base": sum(x) / n, "acc_other": sum(y) / n,
            "diff": diff, "ci95_low": boots[int(0.025 * n_boot)], "ci95_high": boots[int(0.975 * n_boot) - 1],
            "base_only_correct": b, "other_only_correct": c, "mcnemar_p": mcnemar_exact_p(b, c),
            "answer_agreement": sum(1 for v in vids if a[v] == o[v]) / n}


def summarize(recs):
    """Per-method accuracy and efficiency rows (plain dicts, so the notebook can build DataFrames)."""
    rows, by_type = [], []
    for m in sorted({r["method"] for r in recs}):
        rs = [r for r in recs if r["method"] == m]
        ok = [r for r in rs if not r["error"]]
        mean = lambda k: sum(r[k] for r in ok) / len(ok) if ok else float("nan")
        gen = sorted(r["gen_seconds"] for r in ok)
        rows.append({"method": m, "n": len(rs), "n_errors": len(rs) - len(ok),
                     "accuracy": sum(r["correct"] for r in rs) / len(rs),
                     "accuracy_no_errors": sum(r["correct"] for r in ok) / len(ok) if ok else float("nan"),
                     "retained_av_ratio": mean("retained_av_ratio"), "n_input_tokens": mean("n_input_tokens"),
                     "gen_seconds_mean": mean("gen_seconds"), "gen_seconds_median": gen[len(gen) // 2] if gen else float("nan"),
                     "peak_mem_gb_mean": mean("peak_mem_gb"), "no_audio_videos": sum(1 for r in ok if not r["has_audio"])})
        for t in sorted({r["problem_type"] for r in rs}):
            tr = [r for r in rs if r["problem_type"] == t]
            by_type.append({"method": m, "problem_type": t, "n": len(tr), "accuracy": sum(r["correct"] for r in tr) / len(tr)})
    return rows, by_type


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--methods", nargs="*", default=None)
    a = ap.parse_args()
    with open(a.config, encoding="utf-8") as f:
        cfg = json.load(f)
    run_worker(cfg, a.shard, a.num_shards, a.methods or list(cfg["methods"]))
