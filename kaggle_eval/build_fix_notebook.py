"""Build svb_4_fix_rerun.ipynb: re-run the videos that failed in svb_1/2/3 (+ a control set) with the lm_head and
audio-SDPA fixes, time every stage, and plot the results.

    python build_fix_notebook.py

Reads the three result zips (svb_results_<method>.zip) next to this file and embeds what it needs from them,
so the notebook needs no Kaggle input besides internet.
"""
import json
import os
import zipfile

from build_notebook import CODE, CODE_FILES, COMMIT, HERE, VARIANTS, WORKER

PREV_TAG = "svb_omni3B_f128_px128_mini500"
METHODS = {m: c for m, c, _ in VARIANTS.values()}  # full, istm_r45, omnizip_r45 with the exact configs used before
N_CONTROL = 24
KEEP = ["method", "video", "idx", "problem_type", "answer", "pred", "correct", "error", "has_audio", "gen_seconds",
        "peak_mem_gb", "n_input_tokens", "n_video_tokens", "n_audio_tokens", "retained_av_ratio"]


def load_prev():
    """Records of the previous run, one per (method, video), first non-error wins (as svb_eval.load_records)."""
    best = {}
    for m in METHODS:
        with zipfile.ZipFile(os.path.join(HERE, f"svb_results_{m}.zip")) as z:
            for name in sorted(n for n in z.namelist() if n.endswith(".jsonl") and f"{PREV_TAG}__{m}__" in n):
                for line in z.read(name).decode("utf-8").splitlines():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = (r["method"], r["video"])
                    if key not in best or (best[key]["error"] and not r["error"]):
                        best[key] = r
    return [{k: (r.get(k)[:160] if k == "error" and r.get(k) else r.get(k)) for k in KEEP} for r in best.values()]


prev = load_prev()
by = {}
for r in prev:
    by.setdefault(r["video"], {})[r["method"]] = r
FAILED = sorted(v for v, d in by.items() if any(x["error"] for x in d.values()))
ok = [v for v, d in by.items() if len(d) == len(METHODS) and not any(x["error"] for x in d.values())]
ok.sort(key=lambda v: by[v]["full"]["n_input_tokens"])
CONTROL = [ok[round(i * (len(ok) - 1) / (N_CONTROL - 1))] for i in range(N_CONTROL)]  # spread over input length
VIZ = [CONTROL[N_CONTROL // 2], next(v for v in FAILED if not by[v]["istm_r45"]["error"])]
assert len(prev) == 1500 and len(FAILED) == 18, (len(prev), len(FAILED))

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(rf"""
# ShortVid-Bench · chạy lại video lỗi với bản sửa `lm_head` + audio SDPA, đo tách thời gian

Lần chạy trước (`svb_1_full`, `svb_2_istm`, `svb_3_av_aligner`, 500 câu) có **18 video lỗi CUDA OOM**:

| nhóm | số video | nơi lỗi | sửa trong notebook này |
|---|---|---|---|
| chỉ `full` | 12 | `lm_head` tính logits cho cả chuỗi khi prefill (≈ 11k token × 152k từ vựng × fp16 ≈ 3.2 GB) | `fix_lm_head`: chỉ chiếu token cuối; greedy `generate()` chỉ đọc `logits[:, -1]` nên đáp án không đổi |
| cả 3 method | 6 | SDPA của audio encoder dựng mask đặc `[1, S, S]` | `fix_audio_sdpa`: attention theo từng đoạn `cu_seqlens`, cùng kết quả toán học (giống bản sửa vision); điểm audio của OmniZip tính y như cũ |

Notebook chạy cả 3 method (`full`, `istm_r45` = ISTM theo paper, `omnizip_r45` = AV_Aligner của fork) trên
18 video lỗi + {N_CONTROL} video **đối chứng** đã chạy được (trải đều theo độ dài chuỗi). Nhóm đối chứng kiểm tra bản sửa
không đổi đáp án, và cho số đo thời gian ở video bình thường.

**Đo thêm:** thời gian từng giai đoạn (đọc video, processor, chép lên GPU, vision encoder, audio encoder, OmniZip,
LLM prefill, `lm_head`, decode), TTFT, VRAM đỉnh từng giai đoạn, số token thật vào LLM; ảnh token video được giữ.
Cuối notebook: accuracy 500 câu **sau khi thay** kết quả của các video chạy lại, và so ghép cặp.

Code commit `{COMMIT}` nhúng sẵn; kết quả lần trước nhúng sẵn (mục 1b). Settings: **GPU T4 x2**, **Internet On**,
nên có secret `HF_TOKEN`.
""")

code(rf"""
# ===== Cấu hình =====
import glob, json, os

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
METHODS = {json.dumps(METHODS, ensure_ascii=False)}
BASELINE = "full"
FIX_LM_HEAD = True                # chỉ tính logits của token cuối khi prefill
FIX_AUDIO_SDPA = True             # attention của audio encoder theo từng đoạn cu_seqlens
PROFILE = True                    # đo thời gian + VRAM từng giai đoạn (có synchronize, chậm hơn vài %)

# 18 video lỗi ở lần trước (lỗi ở ít nhất một method)
FAILED_VIDEOS = {json.dumps(FAILED)}
# video đối chứng: chạy được ở cả 3 method lần trước, trải đều theo số token đầu vào; [] để bỏ
CONTROL_VIDEOS = {json.dumps(CONTROL)}
VIZ_VIDEOS = {json.dumps(VIZ)}  # lưu 4 frame + mặt nạ token được giữ để vẽ

# Phải giống lần trước để so được
MAX_FRAMES = 128
FPS = 2.0
VIDEO_MAX_PIXELS = 128 * 28 * 28
MAX_NEW_TOKENS = 100
DTYPE = "float16"
ATTN_IMPLEMENTATION = "sdpa"
SEED = 0
PREV_TAG = "{PREV_TAG}"
RUN_TAG = "svbfix_omni3B_f128_px128"
CODE_COMMIT = "{COMMIT}"

WORK = "/kaggle/working"
TEMP = "/kaggle/temp"
CODE_ROOT = "/kaggle/temp/code"
RESULTS_DIR = f"{{WORK}}/results"
LOG_DIR = f"{{WORK}}/logs"
DATA_DIR = f"{{TEMP}}/shortvid"
PREV_JSON = f"{{WORK}}/prev_records.json"
os.environ["HF_HOME"] = f"{{TEMP}}/hf"
for d in (TEMP, RESULTS_DIR, LOG_DIR, DATA_DIR, *(f"{{CODE_ROOT}}/{{os.path.dirname(p)}}" for p in {CODE_FILES!r})):
    os.makedirs(d, exist_ok=True)
RUN_VIDEOS = FAILED_VIDEOS + [v for v in CONTROL_VIDEOS if v not in FAILED_VIDEOS]
print(len(FAILED_VIDEOS), "video lỗi +", len(CONTROL_VIDEOS), "đối chứng ×", len(METHODS), "method")
""")

md(f"## 1. Code OmniZip (commit `{COMMIT}`, nhúng sẵn) + cài thư viện")
for p in CODE_FILES:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": f"%%writefile /kaggle/temp/code/{p}\n" + CODE[p]})
code(r"""
import shutil, subprocess, sys
open(f"{CODE_ROOT}/qwen-omni-utils/README.md", "w").write("qwen-omni-utils, vendored from the OmniZip repo\n")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.52.3", "accelerate", "decord", "einops",
                f"{CODE_ROOT}/qwen-omni-utils"], check=True)
import transformers; print("transformers", transformers.__version__, "| code commit", CODE_COMMIT)
""")

md(f"## 1b. Kết quả lần trước ({len(prev)} bản ghi, 500 câu × 3 method, rút gọn)")
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": "%%writefile /kaggle/working/prev_records.json\n" + json.dumps(prev, ensure_ascii=False)})

md("## 2. GPU")
code(r"""
import torch
n_gpu = torch.cuda.device_count()
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i)
    print(f"cuda:{i} {p.name} {p.total_memory / 2**30:.1f} GB")
assert n_gpu >= 1, "Không có GPU: Settings → Accelerator → GPU T4 x2"
NUM_SHARDS = min(n_gpu, 2)
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
""")

md("## 3. Dữ liệu (chỉ các video cần chạy) + model")
code(r"""
import time
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("dùng HF_TOKEN từ Kaggle Secrets")
except Exception:
    print("không có secret HF_TOKEN — tải ẩn danh, dễ bị 429")
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download, snapshot_download

def with_retry(fn, tries=6):
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as ex:
            if attempt == tries:
                raise
            print(f"lần {attempt} lỗi: {type(ex).__name__}: {str(ex)[:150]} → chờ {60 * attempt}s"); time.sleep(60 * attempt)

REPO = "TencentARC/ShortVid-Bench"
data_json = with_retry(lambda: hf_hub_download(REPO, "ShortVid-Bench-1k.json", repo_type="dataset", local_dir=DATA_DIR))
with_retry(lambda: snapshot_download(REPO, repo_type="dataset", local_dir=DATA_DIR, max_workers=4,
                                     allow_patterns=[f"videos/{v}" for v in RUN_VIDEOS]))
missing = [v for v in RUN_VIDEOS if not os.path.exists(f"{DATA_DIR}/videos/{v}")]
assert not missing, f"thiếu {len(missing)} video — chạy lại cell này"
with_retry(lambda: snapshot_download(MODEL_ID, max_workers=4))
print(len(RUN_VIDEOS), "video đã có")
""")

md("## 4. Worker")
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": "%%writefile /kaggle/working/svb_eval.py\n" + WORKER})
code(r"""
def write_config(path):
    cfg = {"model_id": MODEL_ID, "methods": METHODS, "run_tag": RUN_TAG, "results_dir": RESULTS_DIR,
           "result_roots": [RESULTS_DIR], "code_root": CODE_ROOT, "data_json": data_json,
           "video_dir": f"{DATA_DIR}/videos", "mini_test_num": None, "videos": RUN_VIDEOS, "viz_videos": VIZ_VIDEOS,
           "fix_lm_head": FIX_LM_HEAD, "fix_audio_sdpa": FIX_AUDIO_SDPA, "profile": PROFILE,
           "max_frames": MAX_FRAMES, "fps": FPS, "video_max_pixels": VIDEO_MAX_PIXELS, "max_new_tokens": MAX_NEW_TOKENS,
           "dtype": DTYPE, "attn_implementation": ATTN_IMPLEMENTATION, "seed": SEED, "video_reader": "decord",
           "max_rss_gb": round(RAM_GB * 0.75 / NUM_SHARDS, 1),
           "system_prompt": ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
                             "perceiving auditory and visual inputs, as well as generating text and speech. Please "
                             "analyze the video carefully and select the most appropriate answer from the given options.")}
    json.dump(cfg, open(path, "w"), indent=1)
    return path

def launch(config_path, shard, log_path, gpus):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTHONUNBUFFERED="1", MALLOC_ARENA_MAX="2", PYTHONFAULTHANDLER="1")
    return subprocess.Popen([sys.executable, f"{WORK}/svb_eval.py", "--config", config_path,
                             "--shard", str(shard), "--num-shards", str(NUM_SHARDS)],
                            stdout=open(log_path, "a"), stderr=subprocess.STDOUT, env=env)

def tail(path, n=40):
    return "".join(open(path, encoding="utf-8", errors="replace").readlines()[-n:])

RAM_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
sys.path.insert(0, WORK)
import importlib, svb_eval; importlib.reload(svb_eval)
""")

md("## 5. Chạy (mỗi GPU một shard; tự khởi động lại khi worker thoát vì RAM)")
code(r"""
SNAPSHOT_ZIP = f"{WORK}/svb_results_fix_rerun.zip"

def snapshot_and_zip():
    shutil.copytree(LOG_DIR, f"{RESULTS_DIR}/logs_fix_rerun", dirs_exist_ok=True)
    shutil.make_archive(SNAPSHOT_ZIP[:-4] + ".tmp", "zip", root_dir=WORK, base_dir="results")
    os.replace(SNAPSHOT_ZIP[:-4] + ".tmp.zip", SNAPSHOT_ZIP)

subprocess.run(["pkill", "-f", "svb_eval.py"], check=False); time.sleep(3)
cfg_path = write_config(f"{WORK}/run_config.json")
procs = {k: launch(cfg_path, k, f"{LOG_DIR}/shard{k}.log", str(k)) for k in range(NUM_SHARDS)}
restarts = {k: 0 for k in procs}
t0 = time.time()
while procs:
    time.sleep(30)
    recs = svb_eval.load_records([RESULTS_DIR], RUN_TAG)
    print(f"[{(time.time() - t0) / 60:.1f} min]", {m: sum(r["method"] == m for r in recs) for m in METHODS},
          f"/ {len(RUN_VIDEOS)}", flush=True)
    for k, p in list(procs.items()):
        rc = p.poll()
        if rc is None:
            continue
        del procs[k]
        print(f"shard{k}: exit {rc}")
        if rc in (svb_eval.EXIT_HIGH_RSS, -9) and restarts[k] < 10:
            restarts[k] += 1; procs[k] = launch(cfg_path, k, f"{LOG_DIR}/shard{k}.log", str(k))
        elif rc != 0:
            print(tail(f"{LOG_DIR}/shard{k}.log"))
snapshot_and_zip()
print(f"xong sau {(time.time() - t0) / 60:.1f} phút")
""")

md("""
## 6. Video lỗi lần trước: giờ thế nào?

Mỗi ô: trạng thái lần trước → lần này (`OOM` = lỗi, chữ cái = đáp án, ✓/✗ = đúng/sai).
""")
code(r"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.width", 200)
prev = pd.DataFrame(json.load(open(PREV_JSON, encoding="utf-8")))
new = pd.DataFrame(svb_eval.load_records([RESULTS_DIR], RUN_TAG))
new["set"] = np.where(new["video"].isin(FAILED_VIDEOS), "lỗi lần trước", "đối chứng")
pv = prev.set_index(["method", "video"]); nv = new.set_index(["method", "video"])

def cell(r):
    if r is None:
        return "–"
    return "OOM" if pd.notna(r["error"]) else f"{r['pred']}{'✓' if r['correct'] else '✗'}"

rows = []
for v in FAILED_VIDEOS:
    row = {"video": v, "n_input_tokens": int(np.nan_to_num(pv.loc[("full", v), "n_input_tokens"])),
           "n_audio_tokens": int(np.nan_to_num(pv.loc[("full", v), "n_audio_tokens"]))}
    for m in METHODS:
        a = pv.loc[(m, v)] if (m, v) in pv.index else None
        b = nv.loc[(m, v)] if (m, v) in nv.index else None
        row[m] = f"{cell(a)} → {cell(b)}"
    rows.append(row)
display(pd.DataFrame(rows).set_index("video"))

for m in METHODS:
    was = [v for v in FAILED_VIDEOS if (m, v) in pv.index and pd.notna(pv.loc[(m, v), "error"])]
    now = nv.loc[[(m, v) for v in was if (m, v) in nv.index]] if was else nv.iloc[:0]
    print(f"{m:12s}: lỗi lần trước {len(was):2d} → chạy được {int(now['error'].isna().sum()):2d}, đúng {int(now['correct'].sum()):2d}")
err = new[new["error"].notna()]
if len(err):
    print("\nVẪN LỖI:")
    for _, r in err.iterrows():
        print(f"--- {r['method']} {r['video']}: {r['error']}\n{(r.get('traceback') or '').strip().splitlines()[-1:]}")
""")

md("""
## 7. Kiểm tra bản sửa không đổi đáp án (nhóm đối chứng)

Video đối chứng đã chạy được lần trước. Sửa `lm_head` không đổi phép tính đáp án; sửa audio SDPA cùng toán học nhưng
sai số fp16 khác đi một chút, nên có thể lệch vài câu.
""")
code(r"""
ctl = new[new["video"].isin(CONTROL_VIDEOS) & new["error"].isna()].merge(
    prev[["method", "video", "pred", "correct", "gen_seconds", "peak_mem_gb"]], on=["method", "video"], suffixes=("", "_prev"))
agree = ctl.groupby("method").apply(lambda d: pd.Series({
    "n": len(d), "cùng đáp án": (d["pred"] == d["pred_prev"]).mean(),
    "acc trước": d["correct_prev"].mean(), "acc sau": d["correct"].mean(),
    "gen_s trước (median)": d["gen_seconds_prev"].median(), "gen_s sau (median)": d["gen_seconds"].median(),
    "VRAM đỉnh trước (median)": d["peak_mem_gb_prev"].median(), "VRAM đỉnh sau (median)": d["peak_mem_gb"].median()}))
display(agree.round(3))
diff = ctl[ctl["pred"] != ctl["pred_prev"]]
if len(diff):
    display(diff[["method", "video", "pred_prev", "pred", "answer"]])
""")

md("""
## 8. Thời gian từng giai đoạn

`t_other` = phần còn lại của `generate()` ngoài các giai đoạn GPU đã đo (ghép embedding, tính vị trí M-RoPE, overhead
của generate). Ba giai đoạn đầu (đọc video, processor, chép lên GPU) chạy trên CPU, trước `generate()`.
""")
code(r"""
CPU = ["t_read_decode", "t_processor", "t_to_gpu"]
GPU = ["t_vision", "t_audio", "t_omnizip", "t_llm_prefill", "t_lm_head", "t_llm_decode"]
ok = new[new["error"].isna()].copy()
for c in CPU + GPU:
    ok[c] = ok[c].fillna(0.0) if c in ok else 0.0
ok["t_other"] = (ok["gen_seconds"] - ok[GPU].sum(axis=1)).clip(lower=0)
STAGES = CPU + GPU + ["t_other"]
colors = plt.cm.tab10(np.linspace(0, 1, 10))
mlist = list(METHODS)

fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=False)
for ax, (name, d) in zip(axes, ok.groupby("set")):
    mean = d.groupby("method")[STAGES].mean().reindex(mlist)
    bottom = np.zeros(len(mean))
    for c, col in zip(STAGES, colors):
        ax.bar(mean.index, mean[c], bottom=bottom, label=c[2:], color=col); bottom += mean[c].values
    for x, tot in enumerate(bottom):
        ax.text(x, tot, f"{tot:.1f}s", ha="center", va="bottom")
    ax.set_title(f"{name}: thời gian trung bình mỗi mẫu (n={d['video'].nunique()})"); ax.set_ylabel("giây")
axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.show()

display(ok.groupby(["set", "method"])[STAGES + ["ttft_seconds", "gen_seconds"]].median().round(3))
""")

code(r"""
# Từng video lỗi lần trước, từng method
d = ok[ok["set"] == "lỗi lần trước"]
fig, axes = plt.subplots(1, len(mlist), figsize=(6 * len(mlist), 0.35 * len(FAILED_VIDEOS) + 1.5), sharey=True, sharex=True)
for ax, m in zip(np.atleast_1d(axes), mlist):
    dm = d[d["method"] == m].set_index("video").reindex(FAILED_VIDEOS)
    left = np.zeros(len(dm))
    for c, col in zip(STAGES, colors):
        vals = dm[c].fillna(0).values
        ax.barh(range(len(dm)), vals, left=left, color=col, label=c[2:]); left += vals
    ax.set_yticks(range(len(dm))); ax.set_yticklabels([v[:11] for v in dm.index], fontsize=8); ax.invert_yaxis()
    ax.set_title(m); ax.set_xlabel("giây")
np.atleast_1d(axes)[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.suptitle("Video lỗi lần trước: thời gian theo giai đoạn (ô trống = vẫn lỗi)"); plt.tight_layout(); plt.show()
""")

md("## 9. Thời gian theo số token, và tốc độ so với `full` (ghép theo từng video)")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
for m, col in zip(mlist, ["tab:gray", "tab:blue", "tab:orange"]):
    dm = ok[ok["method"] == m]
    axes[0].scatter(dm["n_llm_tokens"], dm["t_llm_prefill"], s=18, label=m, color=col)
    axes[1].scatter(dm["n_video_tokens"], dm["t_vision"], s=18, label=m, color=col)
    axes[2].scatter(dm["n_audio_tokens"], dm["t_audio"], s=18, label=m, color=col)
for ax, (x, y) in zip(axes, [("số token vào LLM", "LLM prefill (s)"), ("số video token", "vision encoder (s)"),
                              ("số audio token", "audio encoder (s)")]):
    ax.set_xlabel(x); ax.set_ylabel(y); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()

base = ok[ok["method"] == BASELINE].set_index("video")
rows = []
for m in mlist:
    if m == BASELINE:
        continue
    dm = ok[ok["method"] == m].set_index("video")
    v = dm.index.intersection(base.index)
    r = {"method": m, "n_video": len(v), "token vào LLM / full": (dm.loc[v, "n_llm_tokens"] / base.loc[v, "n_llm_tokens"]).median()}
    for c in ["t_llm_prefill", "t_llm_decode", "ttft_seconds", "gen_seconds"]:
        r[f"{c} / full"] = (dm.loc[v, c] / base.loc[v, c]).median()
    r["chi phí OmniZip (s, median)"] = dm.loc[v, "t_omnizip"].median()
    r["VRAM đỉnh − full (GB, median)"] = (dm.loc[v, "peak_mem_gb"] - base.loc[v, "peak_mem_gb"]).median()
    rows.append(r)
display(pd.DataFrame(rows).set_index("method").round(3))
""")

md("## 10. VRAM đỉnh")
code(r"""
PEAKS = [c for c in ["peak_vision_gb", "peak_audio_gb", "peak_omnizip_gb", "peak_llm_prefill_gb", "peak_lm_head_gb",
                     "peak_llm_decode_gb"] if c in ok]
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
med = ok.groupby("method")[PEAKS].median().reindex(mlist)
med.columns = [c[5:-3] for c in PEAKS]
med.T.plot.bar(ax=axes[0]); axes[0].set_ylabel("GB (median)"); axes[0].set_title("VRAM đỉnh theo giai đoạn")
axes[0].axhline(GPU_GB, color="red", ls="--", lw=1, label="VRAM T4")
for m, col in zip(mlist, ["tab:gray", "tab:blue", "tab:orange"]):
    dm = new[new["method"] == m]
    e = dm["error"].notna()
    axes[1].scatter(dm.loc[~e, "n_input_tokens"], dm.loc[~e, "peak_mem_gb"], s=18, color=col, label=m)
    axes[1].scatter(dm.loc[e, "n_input_tokens"], dm.loc[e, "peak_mem_gb"], s=50, color=col, marker="x",
                    label=f"{m} (vẫn lỗi)" if e.any() else None)
axes[1].axhline(GPU_GB, color="red", ls="--", lw=1); axes[1].set_xlabel("số token đầu vào (trước nén)")
axes[1].set_ylabel("VRAM đỉnh cả mẫu (GB)"); axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md("## 11. Token video được giữ (vùng tối = bị bỏ)")
code(r"""
for v in VIZ_VIDEOS:
    files = {m: f"{RESULTS_DIR}/viz/{m}__{v}.npz" for m in mlist}
    files = {m: f for m, f in files.items() if os.path.exists(f)}
    if not files:
        print("không có ảnh cho", v); continue
    fig, axes = plt.subplots(len(files), 4, figsize=(16, 3.2 * len(files)), squeeze=False)
    for i, (m, f) in enumerate(files.items()):
        z = np.load(f)
        for j in range(4):
            img = z["frames"][j].astype(float) / 255
            mask = np.kron(z["keep"][j], np.ones((28, 28)))[: img.shape[0], : img.shape[1]]
            axes[i, j].imshow(img * (0.25 + 0.75 * mask[..., None]))
            axes[i, j].set_title(f"{m} · t={int(z['tp'][j])} · giữ {z['keep'][j].mean():.0%}", fontsize=9)
            axes[i, j].axis("off")
    plt.suptitle(v); plt.tight_layout(); plt.show()
""")

md("""
## 12. Accuracy 500 câu sau khi thay kết quả các video chạy lại

Bản ghi của các video chạy lại (lỗi + đối chứng) thay cho bản ghi cũ; 500 − số video chạy lại giữ kết quả cũ. Accuracy
lúc này không còn thiên vị do OOM của `full`. Thời gian trong bảng trộn lần cũ và lần mới, chỉ đọc cột accuracy.
""")
code(r"""
import itertools
cols = [c for c in prev.columns]
key_new = set(zip(new["method"], new["video"]))
merged = pd.concat([prev[[k not in key_new for k in zip(prev["method"], prev["video"])]],
                    new[[c for c in cols if c in new]]], ignore_index=True)
mrecs = merged.replace({np.nan: None}).to_dict("records")
for r in mrecs:
    r["correct"] = bool(r["correct"])
rows, by_type = svb_eval.summarize([r for r in mrecs if r["error"] or r["gen_seconds"] is not None])
display(pd.DataFrame(rows).set_index("method")[["n", "n_errors", "accuracy", "accuracy_no_errors", "retained_av_ratio"]].round(4))
paired = pd.DataFrame([svb_eval.paired_compare(mrecs, a, b) for a, b in itertools.combinations(mlist, 2)])
display(paired.round(4))

fig, ax = plt.subplots(figsize=(7, 3.2))
for i, r in paired.iterrows():
    ax.errorbar(r["diff"] * 100, i, xerr=[[(r["diff"] - r["ci95_low"]) * 100], [(r["ci95_high"] - r["diff"]) * 100]],
                fmt="o", capsize=4)
ax.axvline(0, color="gray", lw=1); ax.set_yticks(range(len(paired)))
ax.set_yticklabels([f"{a} → {b}" for a, b in zip(paired["base"], paired["other"])])
ax.set_xlabel("chênh accuracy (điểm %), KTC 95% bootstrap"); ax.set_title("So ghép cặp, 500 câu")
plt.tight_layout(); plt.show()
""")

md("## 13. Lưu")
code(r"""
new.to_csv(f"{RESULTS_DIR}/{RUN_TAG}__records.csv", index=False)
ok[["method", "video", "set"] + STAGES + ["ttft_seconds", "gen_seconds", "n_llm_tokens"] + PEAKS].to_csv(
    f"{RESULTS_DIR}/{RUN_TAG}__stages.csv", index=False)
merged.to_csv(f"{RESULTS_DIR}/{RUN_TAG}__merged500_records.csv", index=False)
paired.to_csv(f"{RESULTS_DIR}/{RUN_TAG}__merged500_paired.csv", index=False)
snapshot_and_zip()
print("tải về:", SNAPSHOT_ZIP)
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    src = c["source"]
    c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
out_path = os.path.join(HERE, "svb_4_fix_rerun.ipynb")
json.dump(nb, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out_path, len(cells), "cells,", os.path.getsize(out_path) // 1024, "KB |",
      len(FAILED), "failed +", len(CONTROL), "control videos | viz:", VIZ)
