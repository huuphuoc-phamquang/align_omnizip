"""Build one self-contained notebook per method: ShortVid-Bench (full / istm / av_aligner) and WorldSense
(av_aligner base / av_aligner + MMR).

    python build_notebook.py

Every notebook embeds svb_eval.py and the OmniZip code at the repo HEAD (via %%writefile), so Kaggle needs no
code dataset. Run the notebooks of one benchmark with the same N_VIDEOS; add the other notebooks' result zips as
Input to compare them.
"""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")  # OmniZip code lives at the repo root
WORKER = open(os.path.join(HERE, "svb_eval.py"), encoding="utf-8").read()
CODE_FILES = [  # what the worker imports: the OmniZip Qwen2.5-Omni model + the vendored qwen-omni-utils
    "omnizip/omnizip_core.py",
    "omnizip/av_malign.py",
    "omnizip/modeling_qwen2_5_omni.py",
    "qwen-omni-utils/pyproject.toml",
    "qwen-omni-utils/src/qwen_omni_utils/__init__.py",
    "qwen-omni-utils/src/qwen_omni_utils/v2_5/__init__.py",
    "qwen-omni-utils/src/qwen_omni_utils/v2_5/audio_process.py",
    "qwen-omni-utils/src/qwen_omni_utils/v2_5/vision_process.py",
]
git = lambda *a: subprocess.run(["git", "-C", REPO, *a], check=True, capture_output=True, text=True, encoding="utf-8").stdout
COMMIT = git("rev-parse", "--short", "HEAD").strip()
dirty = git("status", "--porcelain", "--", *CODE_FILES).strip()
assert not dirty, f"repo có thay đổi chưa commit trong code OmniZip, notebook sẽ không khớp commit {COMMIT}:\n{dirty}"
CODE = {p: git("show", f"HEAD:{p}") for p in CODE_FILES}

R45 = {"rho_audio": 0.3, "rho_video": 0.6, "g": 3, "contextual_ratio": 0.05}  # mức giữ ~45% của paper
VARIANTS = {
    # file name -> (method name, method config, one-line description)
    "svb_1_full": ("full", {"wrapper": "none"},
                   "`transformers.Qwen2_5OmniForConditionalGeneration` gốc, không nén token (baseline)"),
    "svb_2_istm": ("istm_r45", {"wrapper": "omnizip", "video_compressor": "istm", **R45},
                   "OmniZip **theo paper / repo gốc KD-TAO**: video nén bằng ISTC/ISTM, AV_Aligner bị bỏ qua"),
    "svb_3_av_aligner": ("omnizip_r45", {"wrapper": "omnizip", "video_compressor": "av_aligner", **R45},
                         "OmniZip **của fork**: video nén bằng AV_Aligner (tên method giữ như lần chạy trước để resume được)"),
}
WS_VARIANTS = {
    "ws_0_full": ("full", {"wrapper": "none"},
                  "**Baseline**: `transformers.Qwen2_5OmniForConditionalGeneration` gốc, không nén token trên WorldSense"),
    "ws_1_av_aligner": ("omnizip_r45", {"wrapper": "omnizip", "video_compressor": "av_aligner", **R45},
                        "**Base**: OmniZip của fork, video nén bằng AV_Aligner (top-k điểm audio–video mỗi frame)"),
    "ws_2_av_aligner_mmr": ("av_mmr_r45", {"wrapper": "omnizip", "video_compressor": "av_aligner_mmr", "mmr_lambda": 0.5, **R45},
                            "**Mới**: cùng điểm AV_Aligner nhưng chọn bằng MMR, "
                            "i* = argmax λ·s_i − (1−λ)·max_{j∈S} cos(x_i, x_j), λ = 0.5 (phạt token trùng lặp trong frame)"),
    "ws_3_istm": ("istm_r45", {"wrapper": "omnizip", "video_compressor": "istm", **R45},
                  "**Paper gốc KD-TAO**: OmniZip với video nén bằng ISTC/ISTM trên WorldSense"),
}

SVB_DATA = r"""
REPO = "TencentARC/ShortVid-Bench"
data_json = with_retry(lambda: hf_hub_download(REPO, "ShortVid-Bench-1k.json", repo_type="dataset", local_dir=DATA_DIR))
samples = json.load(open(data_json, encoding="utf-8"))
need = samples[:N_VIDEOS] if N_VIDEOS else samples
n_total = len(need)
need_videos = sorted({s["video"] for s in need} | {s["video"] for s in samples[:SMOKE_N]})
with_retry(lambda: snapshot_download(REPO, repo_type="dataset", local_dir=DATA_DIR, max_workers=4,
                                     allow_patterns=[f"videos/{v}" for v in need_videos]))
missing = [v for v in need_videos if not os.path.exists(f"{DATA_DIR}/videos/{v}")]
assert not missing, f"thiếu {len(missing)} video, vd {missing[:3]} — chạy lại cell này để tải tiếp"
print(f"{n_total}/{len(samples)} câu hỏi sẽ chạy; {len(need_videos)} video đã có")
"""
BENCHES = {
    "svb": dict(
        title="ShortVid-Bench", prefix="svb", data_dir="shortvid", fixes=False, data=SVB_DATA, mini="N_VIDEOS",
        siblings="Đây là 1 trong 3 notebook (`svb_1_full`, `svb_2_istm`, `svb_3_av_aligner`). Cả ba dùng **cùng code**",
        same="cùng `N_VIDEOS` câu đầu tiên của bộ 1000 câu",
        n_line="N_VIDEOS = 100                    # số video (= số câu hỏi) sẽ chạy, lấy từ đầu bộ 1000 câu; None = cả 1000",
        extra="",
        paper="Tham chiếu paper (Tab. 1, Qwen2.5-Omni-3B, ShortVid-Bench): full 69.4, OmniZip mức 45% là 68.5 (FlashAttention, A6000)."),
    "ws": dict(
        title="WorldSense", prefix="ws", data_dir="worldsense", fixes=True,
        data=open(os.path.join(HERE, "ws_data_cell.py"), encoding="utf-8").read(), mini="None",
        siblings="Đây là 1 trong 2 notebook (`ws_1_av_aligner` = base, `ws_2_av_aligner_mmr` = mới). Cả hai dùng **cùng code**",
        same="cùng `N_VIDEOS` video chọn ngẫu nhiên theo `SEED` (mọi câu hỏi của các video đó)",
        n_line="N_VIDEOS = 100                    # số VIDEO (mỗi video 1–5 câu hỏi, trung bình 1.9); None = cả 1662 video",
        extra="MAX_DURATION_S = None             # vd 60: chỉ lấy video ≤ 1 phút (ContextGuard/ReMo làm vậy); None = mọi độ dài\n",
        paper="WorldSense: 1662 video (15 s – 11 phút, trung vị 89 s), 3172 câu hỏi, 26 loại task. Video dài → nhiều audio token "
              "(25 token/giây); số video token bị chặn bởi MAX_FRAMES × VIDEO_MAX_PIXELS."),
}


def build(nb_name, method, mcfg, desc, bench="svb"):
    B = BENCHES[bench]
    cells = []
    md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
    code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

    md(rf"""
# {B["title"]} · `{method}` · Qwen2.5-Omni-3B

{desc}.

{B["siblings"]} (commit `{COMMIT}` của
nhánh `fix/omnizip-review-bugs`, đã nhúng sẵn ở mục 1), cùng cấu hình video và {B["same"]},
nên kết quả ghép cặp được với nhau.

**Trace:** mỗi mẫu lỗi lưu traceback đầy đủ; log của worker (kèm stack khi crash cứng, nhờ `PYTHONFAULTHANDLER`) nằm trong
zip kết quả; khi `PROFILE = True` mỗi mẫu có thời gian + VRAM đỉnh từng giai đoạn (vision, audio, OmniZip, LLM prefill, decode).

**Chuẩn bị:** Settings → **Accelerator = GPU T4 x2**, **Internet = On**. Thêm Hugging Face token ở Add-ons → Secrets, tên
`HF_TOKEN` (notebook tải video + model bằng token này; tải ẩn danh từng bị lỗi 429). Không cần Add Input code.

**So sánh các method:** chạy xong notebook khác, tải zip kết quả ở tab Output của nó, tạo Dataset rồi Add Input vào notebook này.
Mục 3 tự nhận các file cùng `RUN_TAG`, mục 8 in bảng ghép cặp (McNemar, khoảng tin cậy bootstrap) cho mọi cặp method có mặt.
Add Input zip của chính notebook này từ phiên trước thì notebook chỉ chạy phần còn thiếu.

**Chỉnh sửa cho T4, như nhau cho mọi method** (trong `svb_eval.py`): attention của vision encoder tính theo từng đoạn
`cu_seqlens` (bản `sdpa` gốc dựng mask đặc và OOM), đọc video AV1 bằng PyAV. T4 không có FlashAttention-2 và bf16, nên dùng
`sdpa` + fp16 (paper dùng FlashAttention + A6000).
""")

    code(rf"""
# ===== Cấu hình — mọi tham số nằm ở đây =====
import glob, json, os

{B["n_line"]}
                                  # các notebook phải cùng N_VIDEOS thì mới ghép cặp được (N_VIDEOS nằm trong RUN_TAG)
{B["extra"]}FIX_LM_HEAD = {B["fixes"]}               # chỉ tính logits token cuối khi prefill (tránh OOM 3 GB ở chuỗi dài)
FIX_AUDIO_SDPA = {B["fixes"]}            # attention audio encoder theo từng đoạn cu_seqlens (tránh OOM ở audio dài)
PROFILE = {B["fixes"]}                   # đo thời gian + VRAM từng giai đoạn
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
# Đổi tham số thì đổi luôn TÊN method: tên nằm trong tên file kết quả (resume theo tên)
METHODS = {json.dumps({method: mcfg}, ensure_ascii=False)}
BASELINE = "full"

MAX_FRAMES = 128                  # paper Sec. 4.1: 128 frame cho các benchmark ngoài VideoMME
FPS = 2.0
VIDEO_MAX_PIXELS = 128 * 28 * 28  # README repo: giá trị dùng trong paper
MAX_NEW_TOKENS = 100              # như eval/eval_shortvideobench.py
SMOKE_N = 4                       # số mẫu cho smoke test + đo tốc độ trước khi chạy thật
DTYPE = "float16"                 # T4 không có bf16
ATTN_IMPLEMENTATION = "sdpa"      # T4 không chạy FlashAttention-2
SEED = 0
RUN_MODE = "auto"                 # "auto" | "parallel_2gpu" | "sequential"
SNAPSHOT_EVERY_MIN = 15
CODE_COMMIT = "{COMMIT}"          # commit của code nhúng ở mục 1

WORK = "/kaggle/working"
TEMP = "/kaggle/temp"             # không được lưu vào Output: model, video, code
CODE_ROOT = "/kaggle/temp/code"   # mục 1 ghi code vào đây
RESULTS_DIR = f"{{WORK}}/results"
LOG_DIR = f"{{WORK}}/logs"
DATA_DIR = f"{{TEMP}}/{B["data_dir"]}"
os.environ["HF_HOME"] = f"{{TEMP}}/hf"
model_short = MODEL_ID.split("/")[-1].replace("Qwen2.5-Omni-", "omni")
RUN_TAG = (f"{B["prefix"]}_{{model_short}}_f{{MAX_FRAMES}}_px{{VIDEO_MAX_PIXELS // 784}}_"
           f"{{'mini' + str(N_VIDEOS) if N_VIDEOS else 'all'}}")
for d in (TEMP, RESULTS_DIR, LOG_DIR, DATA_DIR, *(f"{{CODE_ROOT}}/{{os.path.dirname(p)}}" for p in {CODE_FILES!r})):
    os.makedirs(d, exist_ok=True)
DATA_READY = SMOKE_OK = False  # đặt True ở cuối cell tương ứng; cell sau kiểm tra để không chạy trên dữ liệu thiếu
print("RUN_TAG =", RUN_TAG, "| method:", list(METHODS))
""")

    md(f"## 1. Code OmniZip (commit `{COMMIT}`, nhúng sẵn) + cài thư viện")
    for p in CODE_FILES:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": f"%%writefile /kaggle/temp/code/{p}\n" + CODE[p]})
    code(r"""
import shutil, subprocess, sys
# pyproject.toml của qwen-omni-utils khai báo readme; README thật không ảnh hưởng gì nên chỉ ghi một dòng
open(f"{CODE_ROOT}/qwen-omni-utils/README.md", "w").write("qwen-omni-utils, vendored from the OmniZip repo\n")
assert "_audio_token_importance" in open(f"{CODE_ROOT}/omnizip/modeling_qwen2_5_omni.py", encoding="utf-8").read()
# transformers 4.52.3: bản mà omnizip/modeling_qwen2_5_omni.py được chép từ đó (setup.sh)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.52.3", "accelerate", "decord", "einops",
                f"{CODE_ROOT}/qwen-omni-utils"], check=True)
import transformers; print("transformers", transformers.__version__, "| code commit", CODE_COMMIT)
""")

    md("## 2. GPU")
    code(r"""
import torch
n_gpu = torch.cuda.device_count()
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i)
    print(f"cuda:{i} {p.name} {p.total_memory / 2**30:.1f} GB, sm_{p.major}{p.minor}")
assert n_gpu >= 1, "Không có GPU: Settings → Accelerator → GPU T4 x2"
if RUN_MODE == "auto":
    RUN_MODE = "parallel_2gpu" if n_gpu >= 2 else "sequential"
if RUN_MODE == "parallel_2gpu":
    assert n_gpu >= 2, "RUN_MODE=parallel_2gpu cần 2 GPU (Settings → Accelerator → GPU T4 x2)"
NUM_SHARDS = 2 if RUN_MODE == "parallel_2gpu" else 1
print("RUN_MODE =", RUN_MODE, "| shards =", NUM_SHARDS)
""")

    md("## 3. Khôi phục kết quả đã có (phiên trước của notebook này, hoặc zip của notebook khác)")
    code(r"""
restored = {}
for i, p in enumerate(sorted(glob.glob("/kaggle/input/**/results/**/*.jsonl", recursive=True))):
    name = os.path.basename(p)
    if name.startswith(RUN_TAG + "__"):
        dst = os.path.join(RESULTS_DIR, f"restored_{i:03d}", name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy(p, dst)
            m = name.split("__")[1]; restored[m] = restored.get(m, 0) + 1
print("file khôi phục theo method:", restored or "không có")
""")

    md(f"## 4. Dữ liệu {B['title']}")
    code(r"""
import time, random, zipfile, collections
# Tải ẩn danh bị Hugging Face giới hạn (lỗi 429). Thêm token: Add-ons → Secrets → tên HF_TOKEN.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("dùng HF_TOKEN từ Kaggle Secrets")
except Exception:
    print("không có secret HF_TOKEN — tải ẩn danh, dễ bị 429")
os.environ["HF_HUB_DISABLE_XET"] = "1"  # lần chạy trước bị 429 ở endpoint xet-read-token
from huggingface_hub import hf_hub_download, snapshot_download

def with_retry(fn, tries=6):
    # snapshot_download bỏ qua file đã có, nên thử lại chỉ tải phần còn thiếu
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as ex:
            if attempt == tries:
                raise
            wait = 60 * attempt
            print(f"lần {attempt} lỗi: {type(ex).__name__}: {str(ex)[:150]} → chờ {wait}s rồi thử lại")
            time.sleep(wait)
""" + B["data"] + r"""
# Tải model một lần ở đây, để hai worker không cùng tải song song
with_retry(lambda: snapshot_download(MODEL_ID, max_workers=4))
print("model:", MODEL_ID, "| disk free (GB):", {d: round(shutil.disk_usage(d).free / 2**30, 1) for d in (WORK, TEMP)})
DATA_READY = True
""")

    md("## 5. Worker (một tiến trình cho mỗi GPU)")
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": "%%writefile /kaggle/working/svb_eval.py\n" + WORKER})

    code(r"""
def write_config(path, run_tag, results_dir, mini):
    cfg = {"model_id": MODEL_ID, "methods": METHODS, "run_tag": run_tag, "results_dir": results_dir,
           "result_roots": [results_dir], "code_root": CODE_ROOT, "data_json": data_json,
           "video_dir": f"{DATA_DIR}/videos", "mini_test_num": mini, "max_frames": MAX_FRAMES, "fps": FPS,
           "video_max_pixels": VIDEO_MAX_PIXELS, "max_new_tokens": MAX_NEW_TOKENS, "dtype": DTYPE,
           "attn_implementation": ATTN_IMPLEMENTATION, "seed": SEED, "video_reader": "decord",
           "fix_lm_head": FIX_LM_HEAD, "fix_audio_sdpa": FIX_AUDIO_SDPA, "profile": PROFILE,
           # worker tự thoát giữa hai mẫu khi RSS vượt mức này (RAM máy chia cho số worker, chừa chỗ cho kernel)
           "max_rss_gb": round(RAM_GB * 0.75 / NUM_SHARDS, 1),
           "system_prompt": ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
                             "perceiving auditory and visual inputs, as well as generating text and speech. Please "
                             "analyze the video carefully and select the most appropriate answer from the given options.")}
    json.dump(cfg, open(path, "w"), indent=1)
    return path

def launch(config_path, shard, num_shards, log_path, gpus):
    # MALLOC_ARENA_MAX: ít arena hơn → ít RAM bị giữ lại sau mỗi lần giải mã video lớn
    # PYTHONFAULTHANDLER: crash cứng (segfault, abort) vẫn in stack Python vào log
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTHONUNBUFFERED="1", MALLOC_ARENA_MAX="2", PYTHONFAULTHANDLER="1")
    return subprocess.Popen([sys.executable, f"{WORK}/svb_eval.py", "--config", config_path,
                             "--shard", str(shard), "--num-shards", str(num_shards)],
                            stdout=open(log_path, "a"), stderr=subprocess.STDOUT, env=env)

def kill_stale_workers():
    # Một cell bị ngắt (lỗi, Interrupt) không dừng tiến trình con: dọn trước khi chạy lại
    subprocess.run(["pkill", "-f", "svb_eval.py"], check=False)
    time.sleep(3)

def mem_avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20

RAM_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
print(f"RAM máy: {RAM_GB:.1f} GB")

def tail(path, n=40):
    return "".join(open(path, encoding="utf-8", errors="replace").readlines()[-n:])

sys.path.insert(0, WORK)
import importlib, svb_eval; importlib.reload(svb_eval)
""")

    md("""
## 6. Smoke test + đo tốc độ thật

Chạy `SMOKE_N` mẫu đầu trên GPU 0 (kết quả để riêng trong `TEMP`, không lẫn vào kết quả chính). Kiểm tra: không lỗi, model
trả lời được chữ cái; với OmniZip: thật sự nén (retained < 1), số điểm audio bằng số audio token (bản sửa SDPA hoạt động),
và đúng bộ nén video được dùng (ISTM ở notebook `istm`, MMR ở notebook `mmr`).
""")
    code(r"""
import pandas as pd
assert DATA_READY, "cell 4 (dữ liệu) chưa chạy xong — chạy lại cell đó trước"
kill_stale_workers()
smoke_dir = f"{TEMP}/smoke"; shutil.rmtree(smoke_dir, ignore_errors=True)
smoke_tag = "smoke_" + RUN_TAG
cfg_smoke = write_config(f"{TEMP}/smoke_config.json", smoke_tag, smoke_dir, SMOKE_N)
t0 = time.time()
p = launch(cfg_smoke, 0, 1, f"{LOG_DIR}/smoke.log", "0"); p.wait()
print(f"smoke: {time.time() - t0:.0f}s, exit {p.returncode}")
if p.returncode != 0:
    print(tail(f"{LOG_DIR}/smoke.log")); raise RuntimeError("smoke test lỗi, xem log ở trên")

sm = pd.DataFrame(svb_eval.load_records([smoke_dir], smoke_tag))
cols = ["method", "qid", "video", "answer", "pred", "has_audio", "n_audio_tokens", "n_audio_scores", "n_video_tokens",
        "n_istm_groups", "n_mmr_groups", "retained_av_ratio", "n_input_tokens", "n_llm_tokens", "gen_seconds", "ttft_seconds",
        "prep_seconds", "peak_mem_gb", "rss_gb", "error", "response"]
display(sm[[c for c in cols if c in sm]])
errs = sm[sm["error"].notna()]
if len(errs):
    for _, r in errs.iterrows():
        print(f"--- {r['method']} {r.get('qid', r['video'])}: {r['error']}\n{r.get('traceback', '')}")
    raise AssertionError(f"{len(errs)} mẫu lỗi trong smoke test (traceback ở trên)")
for m, mc in METHODS.items():
    if mc["wrapper"] != "omnizip":
        continue
    oz = sm[(sm["method"] == m) & sm["has_audio"]]
    assert len(oz), "smoke không có video nào có audio, không kiểm tra được OmniZip — tăng SMOKE_N"
    assert (oz["retained_av_ratio"] < 1).all(), "OmniZip không nén token nào"
    assert (oz["n_audio_scores"] == oz["n_audio_tokens"]).all(), "số điểm audio != số audio token (lỗi SDPA?)"
    used_istm = oz["n_istm_groups"].fillna(0) > 0 if "n_istm_groups" in oz else pd.Series(False, index=oz.index)
    if mc.get("video_compressor") == "istm":
        assert used_istm.all(), "notebook istm nhưng ISTM không chạy"
    elif mc.get("video_compressor") == "av_aligner_mmr":
        assert "n_mmr_groups" in oz and (oz["n_mmr_groups"].fillna(0) > 0).all(), "notebook MMR nhưng MMR không chạy"
    else:
        print("số nhóm video đi qua ISTM (fork: thường 0):", used_istm.astype(int).tolist())

per = sm.groupby("method")[["gen_seconds", "prep_seconds"]].mean().sum(axis=1)
eta_h = per.sum() * n_total / NUM_SHARDS / 3600
print(f"ước tính: {per.round(1).to_dict()} giây/mẫu → ~{eta_h:.1f} giờ cho {n_total} mẫu "
      f"trên {NUM_SHARDS} GPU (chưa tính tải model). Giới hạn phiên Kaggle: 12 giờ.")
SMOKE_OK = True
""")

    md("## 7. Chạy thật (tự resume, snapshot zip định kỳ kèm log)")
    code(rf"""
SNAPSHOT_ZIP = f"{{WORK}}/{B["prefix"]}_results_{method}.zip"

def snapshot_and_zip():
    # log (có traceback) đi cùng kết quả, để đọc được lỗi sau khi tải zip về
    shutil.copytree(LOG_DIR, f"{{RESULTS_DIR}}/logs_{method}", dirs_exist_ok=True)
    base = SNAPSHOT_ZIP[:-4]
    shutil.make_archive(base + ".tmp", "zip", root_dir=WORK, base_dir="results")
    os.replace(base + ".tmp.zip", SNAPSHOT_ZIP)  # atomic: không bao giờ để lại zip dở
""" + r"""
def progress():
    recs = svb_eval.load_records([RESULTS_DIR], RUN_TAG)
    return {m: sum(r["method"] == m for r in recs) for m in METHODS}

assert SMOKE_OK, "smoke test (cell 6) chưa qua — sửa lỗi ở đó trước khi chạy thật"
kill_stale_workers()
cfg_main = write_config(f"{WORK}/run_config.json", RUN_TAG, RESULTS_DIR, """ + B["mini"] + r""")
# worker tự thoát (75) khi RSS cao, hoặc bị kernel giết (-9) khi hết RAM CPU: khởi động lại, resume bỏ qua mẫu đã chấm
RESTART_CODES = (svb_eval.EXIT_HIGH_RSS, -9)
MAX_RESTARTS = 30

def shard_count(k):
    return sum(r["idx"] % NUM_SHARDS == k and r["method"] in METHODS
               for r in svb_eval.load_records([RESULTS_DIR], RUN_TAG))

def start(k):
    return launch(cfg_main, k, NUM_SHARDS, f"{LOG_DIR}/shard{k}.log", str(k) if NUM_SHARDS > 1 else "0")

procs = {k: start(k) for k in range(NUM_SHARDS)}
restarts = {k: 0 for k in procs}
done_at_start = {k: shard_count(k) for k in procs}
stalled = {k: 0 for k in procs}
failed = []
t0 = last_snap = time.time()
while procs:
    time.sleep(60)
    print(f"[{(time.time() - t0) / 60:.0f} min] {progress()} / {n_total} | RAM trống {mem_avail_gb():.1f} GB", flush=True)
    for k, p in list(procs.items()):
        rc = p.poll()
        if rc is None:
            continue
        del procs[k]
        n = shard_count(k)
        print(f"shard{k}: exit {rc}, đã chấm {n} mẫu")
        if rc == 0:
            continue
        # chết mà không thêm mẫu nào hai lần liền: có thể chính một video làm tràn RAM → dừng thay vì lặp mãi
        stalled[k] = stalled[k] + 1 if n == done_at_start[k] else 0
        if rc in RESTART_CODES and restarts[k] < MAX_RESTARTS and stalled[k] < 2:
            restarts[k] += 1; done_at_start[k] = n
            print(f"shard{k}: khởi động lại lần {restarts[k]}")
            procs[k] = start(k)
        else:
            print(tail(f"{LOG_DIR}/shard{k}.log")); failed.append(k)
    if time.time() - last_snap > SNAPSHOT_EVERY_MIN * 60:
        snapshot_and_zip(); last_snap = time.time()
snapshot_and_zip()
print("số lần khởi động lại:", restarts)
assert not failed, f"shard {failed} lỗi — xem log ở trên; chạy lại cell này sẽ tiếp tục phần còn thiếu"
print("xong:", progress())
""")

    md("""
## 8. Kết quả

- Bảng tổng: mọi method có mặt trong `RESULTS_DIR` (method của notebook này + các zip đã Add Input), cùng `RUN_TAG`.
- `accuracy`: lỗi (OOM, video hỏng...) tính là sai, như `eval/eval_shortvideobench.py`; `accuracy_no_errors` bỏ mẫu lỗi.
- Ghép cặp chỉ trên câu hỏi cả hai method đều chấm: hiệu accuracy, khoảng tin cậy bootstrap 95%, McNemar exact,
  `answer_agreement` = tỉ lệ hai method cùng đúng/cùng sai. Chỉ coi là khác nhau thật khi khoảng tin cậy không chứa 0.
  So nén với nén chỉ công bằng khi `retained_av_ratio` của hai bên xấp xỉ nhau.
- `gen_seconds`: thời gian `generate()` (prefill + bộ nén + decode 1–2 token, không gồm đọc video).
- Mẫu lỗi: bảng cuối in loại lỗi, số token và dòng cuối traceback; traceback đầy đủ nằm trong `__records.csv` và log.
- """ + B["paper"] + """
""")
    code(r"""
import itertools
recs = svb_eval.load_records([RESULTS_DIR], RUN_TAG)
rows, by_type = svb_eval.summarize(recs)
summary = pd.DataFrame(rows).set_index("method")
if BASELINE in summary.index:
    summary["speedup_vs_baseline"] = summary.loc[BASELINE, "gen_seconds_median"] / summary["gen_seconds_median"]
display(summary.round(4))

types = pd.DataFrame(by_type).pivot(index="problem_type", columns="method", values="accuracy")
display(types.round(4))

present = sorted(summary.index, key=lambda m: (m != BASELINE, m))  # baseline đứng trước trong mỗi cặp
paired = pd.DataFrame([c for a, b in itertools.combinations(present, 2) if (c := svb_eval.paired_compare(recs, a, b))])
display(paired.round(4) if len(paired) else "chưa có method thứ hai để ghép cặp — Add Input zip của notebook khác")

df = pd.DataFrame(recs)
stages = [c for c in ["t_vision", "t_audio", "t_omnizip", "t_llm_prefill", "t_lm_head", "t_llm_decode", "ttft_seconds",
                      "gen_seconds", "peak_mem_gb", "n_llm_tokens"] if c in df]
if stages:
    display(df[df["error"].isna()].groupby("method")[stages].median().round(3))

err = df[df["error"].notna()].copy()
if len(err):
    err["last_tb_line"] = err.get("traceback", pd.Series(dtype=str)).fillna("").str.strip().str.split("\n").str[-1]
    display(err[[c for c in ["method", "qid", "video", "has_audio", "n_input_tokens", "n_video_tokens", "n_audio_tokens",
                             "peak_mem_gb", "error", "last_tb_line"] if c in err]])

out = f"{RESULTS_DIR}/{RUN_TAG}__{'_'.join(METHODS)}"
df.to_csv(out + "__records.csv", index=False)
summary.to_csv(out + "__summary.csv"); types.to_csv(out + "__by_type.csv"); paired.to_csv(out + "__paired.csv", index=False)
json.dump({"run_tag": RUN_TAG, "commit": CODE_COMMIT, "model": MODEL_ID, "methods": METHODS,
           "summary": summary.reset_index().to_dict("records"), "paired": paired.to_dict("records")},
          open(out + "__summary.json", "w"), indent=1, default=float)
snapshot_and_zip()
print("đã lưu:", sorted(os.listdir(RESULTS_DIR)))
print("tải về:", SNAPSHOT_ZIP)
""")

    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for c in nb["cells"]:
        src = c["source"]
        c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    out_path = os.path.join(HERE, nb_name + ".ipynb")
    json.dump(nb, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out_path, len(cells), "cells,", os.path.getsize(out_path) // 1024, "KB")


if __name__ == "__main__":  # build_fix_notebook.py imports the embedded code from here
    for name, (method, mcfg, desc) in VARIANTS.items():
        build(name, method, mcfg, desc)
    for name, (method, mcfg, desc) in WS_VARIANTS.items():
        build(name, method, mcfg, desc, bench="ws")
