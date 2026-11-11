# Cell "Dữ liệu WorldSense" của ws_*.ipynb; build_notebook.py chèn nguyên văn vào notebook (with_retry, DATA_DIR... định nghĩa ở cell trước)
REPO = "honglyhly/WorldSense"
qa_path = with_retry(lambda: hf_hub_download(REPO, "worldsense_qa.json", repo_type="dataset", local_dir=DATA_DIR))
qa = json.load(open(qa_path, encoding="utf-8"))
dur = {v: int(str(qa[v]["video_duration"]).rstrip("s")) for v in qa}
pool = sorted(v for v in qa if MAX_DURATION_S is None or dur[v] <= MAX_DURATION_S)
# mẫu ngẫu nhiên cố định theo SEED (đại diện cho cả phân bố độ dài); hai notebook cùng SEED/N_VIDEOS chọn cùng video
chosen = sorted(random.Random(SEED).sample(pool, N_VIDEOS)) if N_VIDEOS and N_VIDEOS < len(pool) else pool
samples = []
for v in chosen:
    for key in sorted(k for k in qa[v] if k.startswith("task")):
        t = qa[v][key]
        samples.append({"qid": f"{v}__{key}", "video": v + ".mp4", "question": t["question"], "candidates": t["candidates"],
                        "answer": t["answer"], "problem_type": t["task_type"], "task_domain": t["task_domain"],
                        "domain": qa[v]["domain"], "duration": qa[v]["duration"], "video_duration_s": dur[v]})
data_json = f"{DATA_DIR}/ws_samples.json"
json.dump(samples, open(data_json, "w", encoding="utf-8"), ensure_ascii=False)
n_total = len(samples)

# Video nằm trong 11 zip (~18 GB): đọc mục lục từng zip qua HTTP rồi chỉ trích các video đã chọn
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
os.makedirs(f"{DATA_DIR}/videos", exist_ok=True)
todo = {s["video"] for s in samples if not os.path.exists(f"{DATA_DIR}/videos/{s['video']}")}

def extract_from(zp):
    with fs.open(zp, "rb", block_size=16 * 2**20) as f, zipfile.ZipFile(f) as z:
        for info in z.infolist():
            name = os.path.basename(info.filename)
            if name in todo:
                part = f"{DATA_DIR}/videos/{name}.part"
                with z.open(info) as src, open(part, "wb") as dst:
                    shutil.copyfileobj(src, dst, 8 * 2**20)
                os.replace(part, f"{DATA_DIR}/videos/{name}"); todo.discard(name)

t_dl = time.time()
for zp in sorted(p for p in fs.ls(f"datasets/{REPO}", detail=False) if "worldsense_videos_" in p):
    if todo:
        with_retry(lambda: extract_from(zp))
assert not todo, f"thiếu {len(todo)} video, vd {sorted(todo)[:3]} — chạy lại cell này"
print(f"{len(chosen)} video / {n_total} câu hỏi; tải video {time.time() - t_dl:.0f}s;",
      "độ dài:", dict(collections.Counter(qa[v]["duration"] for v in chosen)))
