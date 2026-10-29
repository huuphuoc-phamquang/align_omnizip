#!/usr/bin/env python
"""
Phase P2 - measure OmniVinci's token structure and validate the model API.

Answers the four questions everything downstream depends on:
  1. Which modeling_vila.py is actually imported at runtime (= the file to patch)?
  2. How many vision tokens does one frame produce? How many audio tokens per second?
  3. How many tokens of each kind live in one 30s segment? How long is the LLM sequence?
  4. How many seconds and how many GB does the un-pruned baseline cost for one video?

Without these numbers the FLOPs-matched compression ratios cannot be computed, and the
paper's (rho_a=0.3, rho_v=0.6) - derived from Qwen2.5-Omni's 50 audio : 288 video split -
would be applied to the wrong token distribution.

Usage:
    python probe_omnivinci.py --smoke-only
    CUDA_VISIBLE_DEVICES=2 python probe_omnivinci.py --video /path/to/video.mp4 --frames 32
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=os.environ.get(
        "OV_MODEL_DIR", "/mnt/hdd2/trung/projects/new_users/Nghia/omnivinci"))
    p.add_argument("--video", default=None, help="Sample video. Omit to only load the model.")
    p.add_argument("--frames", type=int, default=32,
                   help="num_video_frames. Keep at 32 on a 24GB card for the first run.")
    p.add_argument("--audio-length", default="max_3600")
    p.add_argument("--prompt", default="Describe what happens in this video.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--smoke-only", action="store_true", help="Only load the model, then exit.")
    p.add_argument("--out", default=None, help="Output JSON path.")
    p.add_argument("--vision-chunk", type=int, default=32,
                   help="Encode video frames in chunks of this size (0 = off). Default 32: "
                        "measured strictly better - 128 frames chunked peaks at 20.57 GB "
                        "versus 23.14 GB for 64 frames unchunked, with twice the video "
                        "tokens. SigLIP "
                        "processes each frame independently, so chunking is numerically "
                        "identical - it only trades speed for peak memory. Needed to fit "
                        "--frames 128 on a 24GB card, where the OOM happens inside the "
                        "vision tower, long before the LLM sequence exists.")
    p.add_argument("--device-map", default="auto",
                   help="Keep 'auto' and select the GPU with CUDA_VISIBLE_DEVICES. "
                        "'auto' installs accelerate hooks that move inputs to the right "
                        "device; a plain 'cuda:0' moves only the weights, leaving input_ids "
                        "on CPU -> device mismatch error.")
    return p.parse_args()


def describe_video(path):
    """Duration, fps and whether the file carries audio.

    Uses decord then cv2 - both are existing OmniVinci dependencies (see media.py).
    No new packages are added to a working environment.
    """
    info = {"duration_sec": None, "fps": None, "has_audio": None}
    try:
        import decord
        vr = decord.VideoReader(path, num_threads=1)
        info["fps"] = float(vr.get_avg_fps())
        info["duration_sec"] = len(vr) / info["fps"] if info["fps"] else None
    except Exception:
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            info["fps"] = fps or None
            info["duration_sec"] = (n / fps) if fps else None
        except Exception as e:
            info["error"] = str(e)
    try:
        from decord import AudioReader, cpu
        AudioReader(path, ctx=cpu(0))
        info["has_audio"] = True
    except Exception:
        info["has_audio"] = False
    return info


def tensor_tokens(out):
    """Count tokens in an encoder output (tensor, list of tensors, or tuple)."""
    if torch.is_tensor(out):
        return int(out.shape[0]) if out.dim() == 2 else int(out.shape[-2])
    if isinstance(out, (list, tuple)):
        return sum(tensor_tokens(o) for o in out if o is not None)
    return 0


def main():
    args = get_args()
    from transformers import AutoConfig, AutoModel, AutoProcessor

    report = {"model_path": args.model_path}

    print("=" * 70)
    print("[1] Loading model")
    print("=" * 70)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=args.device_map)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()

    # --- Question 1: which file is actually running ---
    mod = sys.modules.get(type(model).__module__)
    modeling_file = getattr(mod, "__file__", "?")
    report["modeling_file"] = modeling_file
    print("  class          : %s" % type(model).__name__)
    print("  modeling file  : %s" % modeling_file)
    print("                   ^-- THIS is the file to patch")

    knobs = ["num_video_frames", "interleaved_vis_aud_in_video",
             "interleaved_video_segment_duration", "load_audio_in_video",
             "audio_chunk_length", "audio_sampling_rate", "sound_tower",
             "vision_tower", "mm_projector", "model_dtype"]
    report["config"] = {k: getattr(config, k, None) for k in knobs}
    print("  config:")
    for k in knobs:
        print("    %-36s = %s" % (k, getattr(config, k, None)))

    if not getattr(config, "interleaved_vis_aud_in_video", False):
        print("")
        print("  WARNING: interleaved_vis_aud_in_video = False.")
        print("  OmniZip relies on this time-window structure; without it the method has")
        print("  nothing to anchor to.")

    encs = getattr(model, "encoders", {})
    print("  encoders: %s" % (list(encs.keys()) if encs else "(no .encoders attribute)"))

    # --- Optional: chunk the vision tower to survive high frame counts ---
    # The OOM at --frames 128 happens inside SigLIP, while encoding all frames as one
    # batch. SigLIP has no cross-frame attention, so splitting the batch and
    # concatenating gives bit-identical features at a fraction of the peak memory.
    if args.vision_chunk > 0:
        try:
            vt = model.get_vision_tower()
            orig_forward = vt.forward
            chunk = args.vision_chunk

            def chunked_forward(images, *a, **kw):
                if not torch.is_tensor(images) or images.shape[0] <= chunk:
                    return orig_forward(images, *a, **kw)
                outs = [orig_forward(images[i:i + chunk], *a, **kw)
                        for i in range(0, images.shape[0], chunk)]
                return torch.cat(outs, dim=0)

            vt.forward = chunked_forward
            report["vision_chunk"] = chunk
            print("  vision tower: chunked at %d frames per batch" % chunk)
        except Exception as e:
            print("  WARNING: could not install vision chunking: %s" % e)

    n_gpu = torch.cuda.device_count()
    print("  GPUs visible to this process: %d" % n_gpu)
    if n_gpu > 1:
        print("  WARNING: more than one GPU -> latency and peak-memory numbers will be")
        print("           polluted by cross-device traffic. Re-run with")
        print("           CUDA_VISIBLE_DEVICES=<single_card>.")

    if args.smoke_only or not args.video:
        print("")
        print("[smoke-only] Model loaded successfully. Stopping here.")
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
        return

    # --- Input configuration ---
    for obj in (model.config, processor.config):
        obj.load_audio_in_video = True
        obj.num_video_frames = args.frames
        obj.audio_chunk_length = args.audio_length

    vinfo = describe_video(args.video)
    report["video"] = dict(vinfo)
    report["video"]["path"] = args.video
    report["video"]["num_video_frames"] = args.frames
    print("")
    print("=" * 70)
    print("[2] Sample video")
    print("=" * 70)
    for k, v in vinfo.items():
        print("  %-14s = %s" % (k, v))

    # --- Questions 2 & 3: hooks that count tokens ---
    counts = {"video_tokens": 0, "sound_tokens": 0, "llm_seq_len": None, "encoder_calls": {}}

    def mk_hook(name):
        def hook(_m, _inp, out):
            n = tensor_tokens(out)
            counts[name + "_tokens"] += n
            counts["encoder_calls"].setdefault(name, []).append(n)
        return hook

    handles = []
    for name in ("video", "sound"):
        if encs and name in encs:
            handles.append(encs[name].register_forward_hook(mk_hook(name)))

    llm = getattr(model, "llm", None)
    if llm is not None:
        def pre_hook(_m, _a, kwargs):
            emb = kwargs.get("inputs_embeds")
            if emb is not None:
                counts["llm_seq_len"] = int(emb.shape[-2])
            return None
        try:
            handles.append(llm.register_forward_pre_hook(pre_hook, with_kwargs=True))
        except TypeError:
            print("  (old torch: with_kwargs unsupported, skipping llm_seq_len)")

    # --- Single run ---
    print("")
    print("=" * 70)
    print("[3] Running inference")
    print("=" * 70)
    conversation = [{"role": "user", "content": [
        {"type": "video", "video": args.video},
        {"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor([text])

    # Move input_ids onto the model's device. With device_map='auto' the accelerate
    # hooks would handle this, but doing it explicitly removes the dependency on them.
    input_ids = inputs.input_ids
    try:
        input_ids = input_ids.to(model.device)
    except Exception:
        pass

    gc = model.default_generation_config
    gc.update(max_new_tokens=args.max_new_tokens, max_length=99999999, do_sample=False)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(
            input_ids=input_ids,
            media=getattr(inputs, "media", None),
            media_config=getattr(inputs, "media_config", None),
            generation_config=gc)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    for h in handles:
        h.remove()

    answer = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    print("  output: %s" % answer[:200])
    print("  elapsed: %.2f s | peak mem: %.2f GB" % (elapsed, peak_gb))

    # --- Derive the numbers needed for FLOPs matching ---
    dur = vinfo.get("duration_sec")
    seg_dur = getattr(config, "interleaved_video_segment_duration", None)
    vt, at = counts["video_tokens"], counts["sound_tokens"]

    # audio_chunk_length TRUNCATES the audio stream, it does not compress it:
    # an int value N means only the first N seconds are encoded, the rest is dropped.
    # Dividing audio tokens by the full video duration would therefore understate the
    # encoder rate. Work out how many seconds of audio were actually consumed.
    acl = args.audio_length
    if isinstance(acl, str) and "max" in acl:
        cap = float(acl.split("_")[1]) if "_" in acl else float("inf")
    else:
        cap = float(acl)
    audio_sec_used = min(dur, cap) if dur else None
    audio_truncated = bool(dur and cap < dur)

    derived = {
        "video_tokens_total": vt,
        "sound_tokens_total": at,
        "llm_seq_len": counts["llm_seq_len"],
        "tokens_per_frame": (vt / args.frames) if args.frames else None,
        "audio_seconds_used": audio_sec_used,
        "audio_truncated": audio_truncated,
        "audio_tokens_per_sec": (at / audio_sec_used) if audio_sec_used else None,
        "num_segments": (int(dur // seg_dur) + 1) if (dur and seg_dur) else None,
        "audio_share_of_av_tokens": (at / (at + vt)) if (at + vt) else None,
        "latency_sec": elapsed,
        "peak_mem_gb": peak_gb,
    }
    if derived["num_segments"]:
        derived["video_tokens_per_segment"] = vt / derived["num_segments"]
        derived["audio_tokens_per_segment"] = at / derived["num_segments"]

    report["counts"] = counts
    report["derived"] = derived

    print("")
    print("=" * 70)
    print("[4] Results - inputs for the compression ratios")
    print("=" * 70)
    for k, v in derived.items():
        if isinstance(v, float):
            print("  %-28s = %.4g" % (k, v))
        else:
            print("  %-28s = %s" % (k, v))

    if audio_truncated:
        print("")
        print("  === AUDIO TRUNCATED ===")
        print("  audio_chunk_length=%s keeps only the first %.0fs of a %.0fs video."
              % (acl, audio_sec_used, dur))
        print("  The model is deaf for the remaining %.0fs. Unusable for audio-visual"
              % (dur - audio_sec_used))
        print("  evaluation - use --audio-length max_3600 instead.")

    print("")
    print("  Reference, Qwen2.5-Omni (the paper): 288 video : 50 audio per 2s window,")
    print("  audio = 14.8%% of the sequence.")
    sh = derived["audio_share_of_av_tokens"]
    if sh is not None:
        print("  OmniVinci here: audio = %.1f%% of audio-visual tokens." % (sh * 100))
        if sh < 0.05:
            print("  -> Audio is tiny: dropping every audio token would barely reduce FLOPs.")
            print("     The compression budget must go almost entirely to video")
            print("     (high rho_v, low rho_a).")

    out = args.out or os.path.join(
        os.environ.get("RESULTS_DIR", "./results"), "token_stats",
        "omnivinci_token_profile_%s.json" % Path(args.video).stem)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("")
    print("  Written: %s" % out)


if __name__ == "__main__":
    main()
