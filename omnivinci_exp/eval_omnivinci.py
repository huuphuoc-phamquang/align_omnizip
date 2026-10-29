#!/usr/bin/env python
"""
Evaluate OmniVinci on WorldSense / AVUT / ShortVid-Bench.

One harness for all three benchmarks and all compression methods, so every number in
the report comes from the same code path. The three datasets ship different JSON
schemas; they are normalised into a single sample format here.

Compression methods:
    none     un-pruned baseline (works today)
    random   random token pruning, the control group   (needs the model patch, phase P3/P4)
    omnizip  audio-guided pruning                      (needs the model patch, phase P3/P4)

Standard configuration, fixed in phase P2 (see PORT_PLAN.md):
    --frames 128 --audio-length max_3600 --vision-chunk 32
128 frames without chunking OOMs inside SigLIP on a 24GB card; audio_chunk_length=120
silently truncates the audio stream at 120s, which makes audio-visual questions
unanswerable for longer videos.

Usage:
    # smoke test, 50 samples
    CUDA_VISIBLE_DEVICES=2 python eval_omnivinci.py --benchmark worldsense \
        --method none --subset 50

    # full baseline
    CUDA_VISIBLE_DEVICES=2 python eval_omnivinci.py --benchmark worldsense --method none
"""
import argparse
import json
import os

# Silence the ffmpeg/OpenCV decoder chatter. Several benchmark clips are AV1-encoded;
# OpenCV tries hardware decoding, fails, and falls back to software while printing
# "Your platform doesn't support hardware accelerated AV1 decoding" for every frame.
# Decoding still works - the messages are noise that buries the progress bar and real
# errors. Must be set before cv2 is first imported (the model's remote code imports it).
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")   # -8 = AV_LOG_QUIET
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

# Sequence length varies by two orders of magnitude across a benchmark (a 23s clip
# versus a 641s one), which fragments the caching allocator over a long run.
# Expandable segments let the allocator grow a block instead of reserving a new one.
# Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

VALID_LETTERS = ["A", "B", "C", "D", "E", "F"]


# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True,
                   choices=["worldsense", "avut", "shortvid"])
    p.add_argument("--method", default="none",
                   choices=["none", "random", "omnizip", "omnizip_audio", "omnizip_video"])
    p.add_argument("--model-path", default=os.environ.get(
        "OV_MODEL_DIR", "/mnt/hdd2/trung/projects/new_users/Nghia/omnivinci"))
    p.add_argument("--data-root", default=os.environ.get(
        "DATA_DIR", "/mnt/hdd2/trung/projects/new_users/Nghia/data"))
    p.add_argument("--results-root", default=os.environ.get(
        "RESULTS_DIR", "/mnt/hdd2/trung/projects/new_users/Nghia/OmniZip/results"))

    # Standard configuration - do not change without re-running the P2 probe.
    p.add_argument("--frames", type=int, default=128)
    p.add_argument("--audio-length", default="max_3600")
    p.add_argument("--vision-chunk", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=16)

    # Compression ratios (rho = fraction REMOVED, not kept - see CLAUDE.md)
    p.add_argument("--rho-audio", type=float, default=0.3)
    p.add_argument("--rho-video", type=float, default=0.6)
    p.add_argument("--target-retained", type=float, default=None,
                   help="Fraction of audio-visual tokens to KEEP, e.g. 0.45. Preferred "
                        "over the rho pair: the video/audio mix changes with duration, "
                        "so a fixed rho pair does not hold a constant budget.")
    p.add_argument("--retained-tag", default=None,
                   help="Label for the run directory, e.g. r45. Derived if omitted.")

    # Sampling. Every method must run on the SAME samples: the subset is drawn
    # deterministically from a sorted list, so the comparison stays paired.
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", action="store_true",
                   help="Skip samples already present in predictions.jsonl.")
    p.add_argument("--decode-check", type=int, default=20,
                   help="Open this many random videos with cv2 before starting, to catch "
                        "unsupported codecs. 0 disables.")
    p.add_argument("--limit-videos", type=int, default=None,
                   help="Debug only: keep samples from the first N distinct videos.")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Dataset loaders -> unified sample schema
#   {sample_id, video_path, question, options: ["A. ...", ...], answer: "A", meta: {...}}
# --------------------------------------------------------------------------------------

def _norm_letter(ans, options):
    """Return the answer as a single letter."""
    if isinstance(ans, str):
        s = ans.strip()
        if len(s) == 1 and s.upper() in VALID_LETTERS:
            return s.upper()
        # "A." / "A)" / "A. some text"
        if len(s) > 1 and s[0].upper() in VALID_LETTERS and not s[1].isalnum():
            return s[0].upper()
        # fall back to matching the option text
        for i, opt in enumerate(options):
            body = opt.split(".", 1)[-1].strip()
            if body and body.lower() == s.lower():
                return VALID_LETTERS[i]
    return None


def load_worldsense(root):
    """dict keyed by video, each holding one or more task* entries."""
    path = Path(root) / "worldsense" / "worldsense_qa.json"
    video_dir = Path(root) / "worldsense" / "videos"
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    for vid, v in data.items():
        video_id = v.get("video_id", vid)
        for key in sorted(v):
            if not key.startswith("task"):
                continue
            t = v[key]
            options = list(t["candidates"])
            samples.append({
                "sample_id": "%s::%s" % (video_id, key),
                "video_path": str(video_dir / ("%s.mp4" % video_id)),
                "question": t["question"],
                "options": options,
                "answer": _norm_letter(t["answer"], options),
                "meta": {"domain": v.get("domain", ""),
                         "task_type": t.get("task_type", key)},
            })
    return samples


def load_avut(root):
    """Flat list; options live in separate option_A..option_D fields.

    The video_path field carries a 'data/' prefix that does not exist in the released
    archive, where clips sit both at the dataset root and inside category folders
    (meeting/, vlog/, news/, ...). Resolving by path string therefore fails. Index every
    mp4 once and look files up by name instead, which is robust to the layout.
    """
    base = Path(root) / "avut"
    data = json.loads((base / "AV_Human_data.json").read_text(encoding="utf-8"))

    index = {}
    for p in base.rglob("*.mp4"):
        index.setdefault(p.name, p)
        index.setdefault(p.stem, p)

    def resolve(vp, video_id):
        keys = []
        if vp:
            keys += [os.path.basename(vp), Path(vp).stem]
        if video_id:
            keys += ["%s.mp4" % video_id, video_id]
        for k in keys:
            if k in index:
                return str(index[k])
        # clips are sometimes named <video_id>_<start>s_<end>s.mp4
        if video_id:
            pref = "%s_" % video_id
            for k, p in index.items():
                if k.startswith(pref):
                    return str(p)
        return vp or ""

    samples = []
    for i, r in enumerate(data):
        options, letters = [], ["A", "B", "C", "D", "E", "F"]
        for L in letters:
            val = r.get("option_%s" % L)
            if val is None or val == "":
                continue
            options.append("%s. %s" % (L, val))
        vp = resolve(r.get("video_path") or "", r.get("video_id", ""))
        samples.append({
            "sample_id": str(r.get("QA_id", i)),
            "video_path": vp,
            "question": r["question"],
            "options": options,
            "answer": _norm_letter(r["answer"], options),
            "meta": {"task_type": r.get("task_type", ""),
                     "video_type": r.get("video_type", ""),
                     "video_id": r.get("video_id", "")},
        })
    return samples


def load_shortvid(root):
    """Flat list, one question per video, same shape as a flattened WorldSense task."""
    base = Path(root) / "shortvid"
    data = json.loads((base / "ShortVid-Bench-1k.json").read_text(encoding="utf-8"))
    samples = []
    for i, r in enumerate(data):
        options = list(r["candidates"])
        name = r["video"]
        # Prefer the transcoded copies: ~80% of ShortVid-Bench is AV1, which the
        # opencv build in this env cannot decode (see transcode_shortvid.sh).
        cand = base / "videos_h264" / name
        if not cand.exists():
            cand = base / "videos" / name
        if not cand.exists():
            cand = base / name
        samples.append({
            "sample_id": "%s::%d" % (name, i),
            "video_path": str(cand),
            "question": r["question"],
            "options": options,
            "answer": _norm_letter(r["answer"], options),
            "meta": {"problem_type": r.get("problem_type", ""),
                     "data_type": r.get("data_type", "")},
        })
    return samples


LOADERS = {"worldsense": load_worldsense, "avut": load_avut, "shortvid": load_shortvid}


def preflight(samples, benchmark, n_check=20):
    """Fail fast on missing videos or unparseable answers.

    Discovering at hour three that half the video paths are wrong is the most expensive
    failure mode in this pipeline, so check everything up front.
    """
    missing_video, bad_answer = [], []
    for s in samples:
        if not os.path.exists(s["video_path"]):
            missing_video.append(s["video_path"])
        if s["answer"] not in VALID_LETTERS:
            bad_answer.append(s["sample_id"])

    n_videos = len({s["video_path"] for s in samples})
    print("  samples        : %d" % len(samples))
    print("  distinct videos: %d (%.2f questions per video)"
          % (n_videos, len(samples) / max(n_videos, 1)))
    print("  missing videos : %d" % len(missing_video))
    print("  bad answers    : %d" % len(bad_answer))

    if missing_video:
        for p in missing_video[:5]:
            print("    missing: %s" % p)
        print("  Fix the path mapping for '%s' before running." % benchmark)
        sys.exit(1)

    # Existing is not the same as decodable. ShortVid-Bench is mostly AV1, which this
    # opencv build opens happily and then reports zero frames for - the model fails
    # with "has no frames" on every such sample. Checking a sample up front turns a
    # half-wasted run into a 5-second warning.
    if n_check > 0:
        import cv2
        paths = sorted({s["video_path"] for s in samples})
        probe = random.Random(0).sample(paths, min(n_check, len(paths)))
        undecodable = []
        for p in probe:
            cap = cv2.VideoCapture(p)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frames <= 0:
                undecodable.append(p)
        rate = len(undecodable) / max(len(probe), 1)
        print("  decode check  : %d/%d sampled videos undecodable (%.0f%%)"
              % (len(undecodable), len(probe), rate * 100))
        if undecodable:
            for p in undecodable[:3]:
                print("    undecodable: %s" % p)
            print("  Almost always an unsupported codec (AV1 in this opencv build).")
            print("  Check with: ffprobe -v error -select_streams v:0 \\")
            print("                -show_entries stream=codec_name -of csv=p=0 <file>")
            if benchmark == "shortvid":
                print("  Fix: bash transcode_shortvid.sh")
            if rate > 0.2:
                print("  Too many to ignore - stopping.")
                sys.exit(1)
    if bad_answer:
        for i in bad_answer[:5]:
            print("    unparseable answer in sample: %s" % i)
        sys.exit(1)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

def build_model(args):
    from transformers import AutoConfig, AutoModel, AutoProcessor

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()

    for obj in (model.config, processor.config):
        obj.load_audio_in_video = True
        obj.num_video_frames = args.frames
        obj.audio_chunk_length = args.audio_length

    # Chunk the vision tower. SigLIP has no cross-frame attention, so splitting the
    # batch is numerically identical; it only trades speed for peak memory. Measured:
    # 128 frames chunked peaks at 20.57 GB, versus OOM without chunking.
    if args.vision_chunk > 0:
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

    if args.method != "none":
        # The pruning methods need the modeling_vila.py patch from phase P3/P4.
        try:
            from omnizip_ov import install_method
        except ImportError as e:
            # Print the real reason. Reporting only "the patch is missing" sent the
            # last debugging round in the wrong direction entirely.
            print("ERROR: could not import install_method from omnizip_ov.")
            print("       %s: %s" % (type(e).__name__, e))
            print("       script dir: %s" % os.path.dirname(os.path.abspath(__file__)))
            print("       cwd       : %s" % os.getcwd())
            sys.exit(2)
        # Refuse to run unpruned while claiming a compression method: without the
        # hook the model silently ignores every setting and the run looks valid.
        vila = Path(args.model_path) / "modeling_vila.py"
        if "OMNIZIP HOOK" not in vila.read_text(encoding="utf-8"):
            print("ERROR: %s is not patched, so --method %s would have no effect."
                  % (vila, args.method))
            print("       Run: python patch_model.py")
            sys.exit(2)
        install_method(model, method=args.method,
                       rho_audio=args.rho_audio, rho_video=args.rho_video,
                       target_retained=args.target_retained)

    return model, processor, config


def build_prompt(sample):
    opts = "\n".join(sample["options"])
    return ("%s\nOptions:\n%s\n"
            "Answer with the option's letter from the given choices directly."
            % (sample["question"], opts))


def parse_prediction(text):
    """Extract the chosen letter from the model's reply.

    Scanning for any character in A-F is wrong: "The answer is D" would return E,
    taken from the E inside "THE". Only standalone letters count.
    """
    t = (text or "").strip()
    if not t:
        return None
    head = t.upper().lstrip("(*# ")

    # 1. Reply starts with the letter, e.g. "B", "C.", "D) green"
    for L in VALID_LETTERS:
        if head.startswith(L) and (len(head) == 1 or not head[1].isalnum()):
            return L

    # 2. Explicit phrasing, e.g. "the answer is D", "option (C)"
    m = re.search(r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-F])\b", head)
    if m:
        return m.group(1)

    # 3. First standalone letter anywhere in the reply
    m = re.search(r"\b([A-F])\b", head)
    if m:
        return m.group(1)

    return None


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------

def score(records):
    total = len(records)
    correct = sum(1 for r in records if r["is_correct"])
    acc = correct / total if total else 0.0
    # Wald interval. Questions from the same video are correlated, so the true
    # interval is slightly wider than this; the report should also state the number
    # of distinct videos.
    half = 1.96 * ((acc * (1 - acc) / total) ** 0.5) if total else 0.0

    groups = {}
    for r in records:
        for k, v in (r.get("meta") or {}).items():
            if not v:
                continue
            g = groups.setdefault(k, {}).setdefault(str(v), {"correct": 0, "total": 0})
            g["total"] += 1
            g["correct"] += int(r["is_correct"])
    for k in groups:
        for v in groups[k]:
            g = groups[k][v]
            g["accuracy"] = g["correct"] / g["total"] if g["total"] else 0.0

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "ci95_half_width": half,
        "distinct_videos": len({r["video_path"] for r in records}),
        "unparsed_predictions": sum(1 for r in records if r["predicted"] is None),
        "errors": sum(1 for r in records if r.get("error")),
        "breakdown": groups,
    }


def write_env(path, args):
    lines = ["timestamp: %s" % datetime.now().isoformat(),
             "argv: %s" % " ".join(sys.argv),
             "python: %s" % sys.version.replace("\n", " "),
             "torch: %s (cuda %s)" % (torch.__version__, torch.version.cuda),
             "CUDA_VISIBLE_DEVICES: %s" % os.environ.get("CUDA_VISIBLE_DEVICES", "")]
    for cmd in (["git", "rev-parse", "HEAD"],
                ["nvidia-smi", "--query-gpu=index,name,driver_version",
                 "--format=csv,noheader"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
            lines.append("%s: %s" % (cmd[0], out.strip().replace("\n", " | ")))
        except Exception as e:
            lines.append("%s: unavailable (%s)" % (cmd[0], e))
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=120).stdout
        lines.append("\n--- pip freeze ---\n" + out)
    except Exception as e:
        lines.append("pip freeze: unavailable (%s)" % e)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    args = get_args()

    print("=" * 70)
    print("[1] Dataset: %s" % args.benchmark)
    print("=" * 70)
    samples = LOADERS[args.benchmark](args.data_root)
    samples.sort(key=lambda s: s["sample_id"])

    if args.limit_videos:
        keep, seen = [], []
        for s in samples:
            if s["video_path"] not in seen:
                if len(seen) >= args.limit_videos:
                    continue
                seen.append(s["video_path"])
            keep.append(s)
        samples = keep

    if args.subset and args.subset < len(samples):
        # Deterministic, so every method sees the same questions and the comparison
        # between methods stays paired.
        samples = random.Random(args.seed).sample(samples, args.subset)
        samples.sort(key=lambda s: s["sample_id"])
        print("  subset: %d samples (seed %d)" % (len(samples), args.seed))

    preflight(samples, args.benchmark, n_check=args.decode_check)

    tag = args.retained_tag
    if tag is None and args.method != "none":
        if args.target_retained:
            tag = "r%.0f" % (args.target_retained * 100)
        else:
            tag = "ra%.0f_rv%.0f" % (args.rho_audio * 100, args.rho_video * 100)
    run_name = args.run_name or "_".join(
        [args.method] + ([tag] if tag else []) +
        (["n%d" % args.subset] if args.subset else []))
    out_dir = Path(args.results_root) / "omnivinci-9b" / args.benchmark / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    print("  output: %s" % out_dir)

    # Resume appends to predictions.jsonl; a fresh run must truncate it. Appending to a
    # previous run's file would leave predictions.jsonl holding two runs while
    # metrics.json describes only the new one.
    done = set()
    records = []
    if pred_path.exists():
        if args.resume:
            for line in pred_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r["sample_id"] in done:
                    continue  # tolerate duplicates from an interrupted run
                done.add(r["sample_id"])
                records.append(r)
            print("  resume: %d samples already done" % len(done))
        else:
            print("  overwriting previous predictions.jsonl (%d bytes). Use --resume to continue it."
                  % pred_path.stat().st_size)
    write_mode = "a" if args.resume else "w"

    (out_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    write_env(out_dir / "env.txt", args)

    print()
    print("=" * 70)
    print("[2] Loading model (method=%s)" % args.method)
    print("=" * 70)
    model, processor, _ = build_model(args)
    gen_cfg = model.default_generation_config
    gen_cfg.update(max_new_tokens=args.max_new_tokens, max_length=99999999,
                   do_sample=False)
    print("  GPUs visible: %d" % torch.cuda.device_count())

    print()
    print("=" * 70)
    print("[3] Running")
    print("=" * 70)

    todo = [s for s in samples if s["sample_id"] not in done]
    n_oom = 0
    t_start = time.perf_counter()
    fh = pred_path.open(write_mode, encoding="utf-8")
    for i, s in enumerate(tqdm(todo, desc=args.benchmark)):
        rec = {"sample_id": s["sample_id"], "video_path": s["video_path"],
               "answer": s["answer"], "meta": s["meta"]}
        if args.method != "none":
            # The budget carries over between segments of one video; it must not
            # leak into the next sample.
            import omnizip_ov
            omnizip_ov.reset_video()
        try:
            conversation = [{"role": "user", "content": [
                {"type": "video", "video": s["video_path"]},
                {"type": "text", "text": build_prompt(s)}]}]
            text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor([text])
            input_ids = inputs.input_ids
            try:
                input_ids = input_ids.to(model.device)
            except Exception:
                pass
            with torch.inference_mode():
                out_ids = model.generate(
                    input_ids=input_ids,
                    media=getattr(inputs, "media", None),
                    media_config=getattr(inputs, "media_config", None),
                    generation_config=gen_cfg)
            raw = processor.tokenizer.batch_decode(
                out_ids, skip_special_tokens=True)[0]
            rec["raw_output"] = raw[:300]
            rec["predicted"] = parse_prediction(raw)
        except Exception as e:
            rec["raw_output"] = ""
            rec["predicted"] = None
            rec["error"] = str(e)[:300]
            # A CUDA OOM leaves the caching allocator holding blocks that the next
            # sample cannot reuse, so one failure cascades into all of them. Releasing
            # the cache lets the run continue and lose only the offending sample.
            if "out of memory" in str(e).lower():
                n_oom += 1
                torch.cuda.empty_cache()
                print("\n  CUDA OOM on sample %s (%d so far). Cache cleared, continuing."
                      % (s["sample_id"], n_oom))
                if n_oom >= 10:
                    print("  Too many OOMs - another process is probably sharing this GPU.")
                    print("  Check: nvidia-smi --query-compute-apps=pid,used_memory --format=csv")

        rec["is_correct"] = bool(rec["predicted"] == s["answer"])
        records.append(rec)
        # Flush every record: a 45-minute run that dies at minute 40 should still leave
        # everything it computed on disk, ready for --resume.
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()

        if (i + 1) % 100 == 0:
            (out_dir / "metrics.json").write_text(
                json.dumps(score(records), indent=2, ensure_ascii=False),
                encoding="utf-8")
    fh.close()

    elapsed = time.perf_counter() - t_start
    metrics = score(records)
    metrics["elapsed_sec"] = elapsed
    metrics["sec_per_sample"] = elapsed / max(len(todo), 1)
    metrics["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    if args.method != "none":
        import omnizip_ov
        metrics["compression"] = omnizip_ov.stats()
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 70)
    print("[4] Result")
    print("=" * 70)
    print("  accuracy   : %.4f  (%d / %d)  +/- %.4f"
          % (metrics["accuracy"], metrics["correct"], metrics["total"],
             metrics["ci95_half_width"]))
    print("  videos     : %d" % metrics["distinct_videos"])
    print("  unparsed   : %d" % metrics["unparsed_predictions"])
    print("  errors     : %d" % metrics["errors"])
    print("  elapsed    : %.1f min (%.2f s/sample)"
          % (elapsed / 60, metrics["sec_per_sample"]))
    print("  peak mem   : %.2f GB" % metrics["peak_mem_gb"])
    print("  written    : %s" % out_dir)


if __name__ == "__main__":
    main()
