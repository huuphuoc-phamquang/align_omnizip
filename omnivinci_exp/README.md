# omnivinci_exp — chạy OmniZip trên OmniVinci

Toàn bộ phần thêm cho thí nghiệm này nằm gọn trong thư mục này, **không sửa file nào
của repo upstream**. Nhờ vậy `git pull` từ KD-TAO/OmniZip về sau vẫn sạch.

## Thứ tự chạy

```bash
cd /mnt/hdd2/trung/projects/new_users/Nghia/OmniZip/omnivinci_exp

bash setup_env.sh          # 1. kiểm tra env omnivinci có sẵn + smoke test
bash download_data.sh      # 2. tải 3 dataset (~54 GB) về ../../data/
python probe_omnivinci.py --video <video_mẫu>.mp4 --frames 32    # 3. đo cấu trúc token
```

**Về conda env:** script dùng **env đang active**. Cách chạy khuyến nghị:

```bash
conda activate omnivinci
bash setup_env.sh
```

Nếu chưa activate gì (hoặc đang ở `base`), script sẽ tự thử activate `$CONDA_ENV_NAME`
(mặc định `omnivinci`, đổi trong `paths.sh`). Nhưng nên tự activate hơn: hook của conda
(`~cuda-nvcc_deactivate.sh` tham chiếu `CUDAARCHS_BACKUP`) hay báo lỗi khi gọi từ trong
script, dễ gây hiểu nhầm là env hỏng.

Bước 3 là bước **quan trọng nhất và dễ bị bỏ qua nhất**. Đọc kỹ phần dưới.

## Vì sao phải chạy `probe_omnivinci.py` trước khi viết một dòng code nén nào

Paper OmniZip dùng cặp `ρ_a = 0.3, ρ_v = 0.6` cho mức giữ lại ~45%. Cặp số đó **không
phải hằng số của phương pháp** — nó được tính ngược từ tỉ lệ token cụ thể của
Qwen2.5-Omni:

```
mỗi cửa sổ 2 giây = 288 video token + 50 audio token
audio chiếm 50/338 = 14.8% chuỗi
(50×0.7 + 288×0.4) / 338 = 44.4%   ✓ khớp con số 45% trong bài
```

OmniVinci có encoder khác hoàn toàn (SigLIP-so400m + AF3 thay vì Qwen2.5-VL + Whisper),
cửa sổ 30 giây thay vì 2 giây. Tỉ lệ video:audio gần như chắc chắn khác. Bê nguyên cặp
`0.3/0.6` sang là **so sánh ở hai mức FLOPs khác nhau** — bảng kết quả sẽ vô nghĩa.

Script in ra `tokens_per_frame`, `audio_tokens_per_sec`, `audio_share_of_av_tokens` để
tính lại cặp `ρ` đúng cho OmniVinci.

Có một khả năng cần đặc biệt để ý: nếu audio chỉ chiếm dưới 5% chuỗi, script sẽ cảnh
báo. Khi đó phần "nén audio" của OmniZip gần như không tiết kiệm được gì, và toàn bộ
ngân sách phải dồn vào video — điều này **thay đổi cách diễn giải kết quả**, không chỉ
thay đổi tham số.

## Chọn GPU

Server có 5 card nhưng không phải lúc nào cũng rảnh cả 5. Chọn card bằng
`CUDA_VISIBLE_DEVICES` (đúng tên biến — viết sai sẽ bị bỏ qua âm thầm và tiến trình
chiếm GPU 0):

```bash
CUDA_VISIBLE_DEVICES=2 python probe_omnivinci.py --video <...> --frames 32
```

Sau khi đặt biến này, card được đánh số lại từ 0 **bên trong tiến trình** — GPU vật lý
số 2 trở thành `cuda:0`. Nên giữ `--device-map cuda:0` (mặc định), đừng viết `cuda:2`.

**Luôn dùng 1 card cho probe và cho mọi phép đo tốc độ.** Nhiều card làm latency và
peak-mem bị pha tạp bởi chi phí giao tiếp, mà đó là baseline để so speedup về sau.
Kiểm tra card rảnh thật: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`

## Script làm gì

| File | Vai trò |
|---|---|
| `paths.sh` | Đường dẫn dùng chung, override bằng biến môi trường |
| `setup_env.sh` | Kiểm tra env có sẵn đủ chưa, cảnh báo nếu torch là build CUDA 13 |
| `download_data.sh` | Tải WorldSense → AVUT → ShortVid, kiểm tra dung lượng trống trước |
| `probe_omnivinci.py` | Đo cấu trúc token + xác thực API + đo latency/memory baseline |

## Nguyên tắc với môi trường: đừng động vào env đang chạy tốt

Env `omnivinci` đã chạy được model — đó là tài sản, không phải thứ để cài đè. Nên
`setup_env.sh` **chỉ kiểm tra và báo cáo**, không tự `pip install` gì (trừ khi truyền
`--install-optional`). Thiếu gói nào thì nó in ra lệnh pip để bạn tự quyết định.

**Không cần cài thêm gì cho phần đọc video/audio.** `media.py` của OmniVinci đã import
sẵn `decord`, `cv2`, `librosa`, `soundfile`, `whisper`, `PIL` — `probe_omnivinci.py` dùng
lại đúng những thư viện đó (decord trước, cv2 dự phòng).

**Đừng chạy `setup.sh` của OmniZip trong env này.** Nó ghim `transformers==4.52.3`,
trong khi snapshot OmniVinci build với 4.46.x. Phần OmniZip ta thực sự cần —
`omnizip/omnizip_core.py` — chỉ `import torch`, không đụng transformers.

**Nếu buộc phải cài gì đó**, luôn kèm `--upgrade-strategy only-if-needed` và kiểm tra
lại phiên bản `torch` / `transformers` sau khi cài. `pip install` mặc định có thể kéo
torch mới nhất kèm `cuda-toolkit 13.x`, mà CUDA 13 cần driver r580+ nên **không chạy
trên driver 12.8**. `setup_env.sh` in phiên bản CUDA build của torch ở mỗi lần chạy.

## Sau khi probe xong

Số đo từ `probe_omnivinci.py` là đầu vào cho:

- Tính cặp `ρ_a, ρ_v` mới → điền vào `EXPERIMENTS.md` §2
- Quyết định chia cửa sổ con trong segment 30s → `PORT_PLAN.md` giai đoạn P4
- Cột "Token vào LLM" của bảng hiệu năng → `EXPERIMENTS.md` §4.3
