"""Export every image output of the notebooks here into ../docs/figures, and draw the WorldSense speed/memory plots
(base AV_Aligner vs AV_Aligner + MMR) from ws_results_*.zip.

    python make_doc_figures.py
"""
import base64
import glob
import hashlib
import json
import os
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# ---- 1. image outputs of the notebooks (identical images from copies of a notebook are written once)
NAMES = {  # (notebook stem, cell index) -> name of the plot; unknown cells keep their index
    ("svb_4_fix_rerun", 28): "thoi-gian-theo-giai-doan", ("svb_4_fix_rerun", 29): "thoi-gian-video-loi",
    ("svb_4_fix_rerun", 31): "thoi-gian-theo-so-token", ("svb_4_fix_rerun", 33): "vram-dinh",
    ("svb_4_fix_rerun", 35): "token-duoc-giu", ("svb_4_fix_rerun", 37): "ghep-cap-500-cau",
}
seen, exported = set(), []
# newest first: of two copies of a notebook (e.g. an aborted run downloaded as 'svb-4-…'), the latest run wins
for nb_path in sorted(glob.glob(os.path.join(HERE, "*.ipynb")), key=os.path.getmtime, reverse=True):
    stem = os.path.splitext(os.path.basename(nb_path))[0].replace("-", "_")
    nb = json.load(open(nb_path, encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        k = 0
        for o in c.get("outputs", []):
            png = o.get("data", {}).get("image/png")
            if not png:
                continue
            raw = base64.b64decode("".join(png) if isinstance(png, list) else png)
            h = hashlib.sha1(raw).hexdigest()
            k += 1
            if h in seen:
                continue
            seen.add(h)
            name = f"{stem}__{NAMES.get((stem, i), f'cell{i}')}{'' if k == 1 else f'-{k}'}.png"
            if name in exported:  # another copy of the same notebook (e.g. the Kaggle download with '-' in its name)
                continue
            open(os.path.join(OUT, name), "wb").write(raw)
            exported.append(name)
print("ảnh từ notebook:", exported)

# ---- 2. WorldSense: base vs MMR, speed and memory
TAG = "ws_omni3B_f128_px128_mini500"
LABEL = {"omnizip_r45": "AV_Aligner (base)", "av_mmr_r45": "AV_Aligner + MMR"}
COLOR = {"omnizip_r45": "tab:orange", "av_mmr_r45": "tab:green"}
rows = []
for m in LABEL:
    with zipfile.ZipFile(os.path.join(HERE, f"ws_results_{m}.zip")) as z:
        for n in z.namelist():
            if n.endswith(".jsonl") and f"{TAG}__{m}__" in n:
                rows += [json.loads(l) for l in z.read(n).decode("utf-8").splitlines() if l.strip()]
df = pd.DataFrame(rows)
df = df[df["error"].isna()].copy()
methods = list(LABEL)
CPU = ["t_read_decode", "t_processor", "t_to_gpu"]
GPU = ["t_vision", "t_audio", "t_omnizip", "t_llm_prefill", "t_lm_head", "t_llm_decode"]
for c in CPU + GPU:
    df[c] = df[c].fillna(0.0)
df["t_other"] = (df["gen_seconds"] - df[GPU].sum(axis=1)).clip(lower=0)
STAGES = CPU + GPU + ["t_other"]
STAGE_VI = {"t_read_decode": "đọc + giải mã video (CPU)", "t_processor": "processor (CPU)", "t_to_gpu": "chép lên GPU",
            "t_vision": "vision encoder", "t_audio": "audio encoder", "t_omnizip": "bộ nén OmniZip",
            "t_llm_prefill": "LLM prefill", "t_lm_head": "lm_head", "t_llm_decode": "LLM decode", "t_other": "khác trong generate()"}
# KV cache of Qwen2.5-Omni-3B thinker: 36 layers x 2 (K, V) x 2 KV heads x 128 dims x 2 bytes (fp16)
KV_BYTES_PER_TOKEN = 36 * 2 * 2 * 128 * 2
df["kv_full_gb"] = df["n_input_tokens"] * KV_BYTES_PER_TOKEN / 2**30
df["kv_kept_gb"] = df["n_llm_tokens"] * KV_BYTES_PER_TOKEN / 2**30
df["audio_s"] = df["n_audio_tokens"] / 25  # Qwen2.5-Omni: 25 audio tokens per second
n_q = df["qid"].nunique()


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    exported.append(name)


# 2a. mean time per question, stacked by stage
fig, ax = plt.subplots(figsize=(9, 5))
mean = df.groupby("method")[STAGES].mean().reindex(methods)
colors = plt.cm.tab10(np.linspace(0, 1, 10))
bottom = np.zeros(len(mean))
for c, col in zip(STAGES, colors):
    ax.bar([LABEL[m] for m in mean.index], mean[c], bottom=bottom, color=col, label=STAGE_VI[c])
    bottom += mean[c].values
for x, tot in enumerate(bottom):
    ax.text(x, tot, f"{tot:.1f} s", ha="center", va="bottom")
ax.set_ylabel("giây / câu hỏi (trung bình)")
ax.set_title(f"WorldSense: thời gian theo giai đoạn ({n_q} câu, 500 video, 2×T4)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
save(fig, "ws__thoi-gian-theo-giai-doan.png")

# 2b. time vs tokens: generate(), LLM prefill, OmniZip cost
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
for m in methods:
    d = df[df["method"] == m]
    kw = dict(s=8, alpha=.45, color=COLOR[m], label=LABEL[m])
    axes[0].scatter(d["n_input_tokens"], d["gen_seconds"], **kw)
    axes[1].scatter(d["n_llm_tokens"], d["t_llm_prefill"], **kw)
    axes[2].scatter(d["n_video_tokens"] + d["n_audio_tokens"], d["t_omnizip"], **kw)
for ax, (x, y) in zip(axes, [("token đầu vào (trước nén)", "generate() (s)"), ("token vào LLM (sau nén)", "LLM prefill (s)"),
                              ("token audio + video", "bộ nén OmniZip (s)")]):
    ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(alpha=.3); ax.legend(markerscale=2)
fig.suptitle("WorldSense: thời gian theo số token")
fig.tight_layout()
save(fig, "ws__thoi-gian-theo-so-token.png")

# 2c. paired per question: MMR / base
p = df.pivot_table(index="qid", columns="method", values=["gen_seconds", "t_omnizip"])
ratio = (p["gen_seconds"]["av_mmr_r45"] / p["gen_seconds"]["omnizip_r45"]).dropna()
extra = (p["t_omnizip"]["av_mmr_r45"] - p["t_omnizip"]["omnizip_r45"]).dropna()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
axes[0].hist(ratio.clip(0.7, 1.4), bins=50, color="tab:green")
axes[0].axvline(1, color="k", lw=1); axes[0].axvline(ratio.median(), color="tab:red", ls="--", label=f"trung vị {ratio.median():.3f}")
axes[0].set_xlabel("generate() của MMR / base, cùng câu hỏi"); axes[0].set_ylabel("số câu"); axes[0].legend()
axes[1].hist(extra, bins=50, color="tab:purple")
axes[1].axvline(extra.median(), color="tab:red", ls="--", label=f"trung vị +{extra.median():.2f} s")
axes[1].set_xlabel("thời gian bộ nén tăng thêm do MMR (s)"); axes[1].set_ylabel("số câu"); axes[1].legend()
fig.suptitle("WorldSense: chi phí thêm của MMR, ghép theo từng câu hỏi")
fig.tight_layout()
save(fig, "ws__mmr-chi-phi-them.png")

# 2d. memory: peak per stage, peak vs tokens, LLM token / KV-cache saving
PEAKS = ["peak_vision_gb", "peak_audio_gb", "peak_omnizip_gb", "peak_llm_prefill_gb", "peak_lm_head_gb", "peak_llm_decode_gb"]
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
med = df.groupby("method")[PEAKS].median().reindex(methods)
x = np.arange(len(PEAKS))
for j, m in enumerate(methods):
    axes[0].bar(x + (j - .5) * .38, med.loc[m], .38, color=COLOR[m], label=LABEL[m])
axes[0].axhline(14.56, color="red", ls="--", lw=1, label="VRAM T4 (14.56 GB)")
axes[0].set_xticks(x); axes[0].set_xticklabels([c[5:-3] for c in PEAKS], rotation=20)
axes[0].set_ylim(8, 15); axes[0].set_ylabel("GB (trung vị)"); axes[0].set_title("VRAM đỉnh theo giai đoạn"); axes[0].legend(fontsize=8)
for m in methods:
    d = df[df["method"] == m]
    axes[1].scatter(d["n_input_tokens"], d["peak_mem_gb"], s=8, alpha=.45, color=COLOR[m], label=LABEL[m])
axes[1].axhline(14.56, color="red", ls="--", lw=1)
axes[1].set_xlabel("token đầu vào (trước nén)"); axes[1].set_ylabel("VRAM đỉnh cả mẫu (GB)"); axes[1].grid(alpha=.3)
axes[1].legend(markerscale=2); axes[1].set_title("VRAM đỉnh theo số token")
b = df[df["method"] == "omnizip_r45"].sort_values("n_input_tokens")
axes[2].plot(b["n_input_tokens"], b["kv_full_gb"], color="gray", label="KV cache nếu không nén")
axes[2].scatter(b["n_input_tokens"], b["kv_kept_gb"], s=8, color="tab:orange", label="KV cache sau nén (cả 2 method)")
axes[2].set_xlabel("token đầu vào (trước nén)"); axes[2].set_ylabel("GB (ước tính)"); axes[2].grid(alpha=.3); axes[2].legend()
axes[2].set_title(f"KV cache của LLM: giữ {np.median(b['n_llm_tokens'] / b['n_input_tokens']):.0%} token (trung vị)")
fig.suptitle("WorldSense: bộ nhớ GPU")
fig.tight_layout()
save(fig, "ws__bo-nho.png")

# 2e. accuracy by audio length (proxy for video duration; audio is cut at ~300 s)
bins = [0, 60, 120, 240, 301]
lab = ["<1 phút", "1–2 phút", "2–4 phút", "≥4 phút"]
df["dur_bin"] = pd.cut(df["audio_s"], bins, labels=lab, include_lowest=True)
acc = df.groupby(["dur_bin", "method"], observed=True)["correct"].agg(["mean", "size"]).unstack("method")
fig, ax = plt.subplots(figsize=(9, 4.2))
xx = np.arange(len(acc))
for j, m in enumerate(methods):
    ax.bar(xx + (j - .5) * .38, acc["mean"][m] * 100, .38, color=COLOR[m], label=LABEL[m])
ax.set_xticks(xx); ax.set_xticklabels([f"{i}\n(n={int(acc['size'][methods[0]][i])})" for i in acc.index])
ax.set_ylabel("accuracy (%)"); ax.axhline(25, color="gray", ls=":", lw=1, label="đoán ngẫu nhiên (4 lựa chọn)")
ax.set_title("WorldSense: accuracy theo độ dài audio"); ax.legend(loc="lower right", framealpha=1)
save(fig, "ws__accuracy-theo-do-dai.png")

summary = df.groupby("method").agg(
    n=("qid", "size"), accuracy=("correct", "mean"), gen_s=("gen_seconds", "median"), ttft_s=("ttft_seconds", "median"),
    omnizip_s=("t_omnizip", "median"), vision_s=("t_vision", "median"), prefill_s=("t_llm_prefill", "median"),
    read_decode_s=("t_read_decode", "median"), peak_gb=("peak_mem_gb", "median"), max_peak_gb=("peak_mem_gb", "max"),
    llm_token_ratio=("n_llm_tokens", lambda s: float(np.median(s / df.loc[s.index, "n_input_tokens"]))),
    kv_saved_gb=("kv_full_gb", lambda s: float(np.median(s - df.loc[s.index, "kv_kept_gb"]))))
summary.round(3).to_csv(os.path.join(OUT, "ws__tong-hop.csv"))
print(summary.round(3).to_string())
print("ảnh:", exported)
