#!/usr/bin/env bash
# Transcode AV1-encoded ShortVid-Bench clips to H.264.
#
# Why this is needed: the opencv-python build in this environment has no AV1 decoder.
# cv2.VideoCapture opens those files but reports zero frames, and OmniVinci's media.py
# then fails with "Video ... has no frames". Around 80% of ShortVid-Bench is AV1, so
# most of the benchmark is unusable without this step. WorldSense (h264) and AVUT
# (vp9 + h264) decode fine and are left alone.
#
# Why transcode instead of upgrading opencv: this env is fragile. Reinstalling opencv
# risks pulling NumPy 2 back in and breaking cv2 for a different reason (see the ABI
# trap documented in setup_env.sh). ffmpeg in the same env already has libdav1d, so a
# one-off transcode fixes the problem without touching any Python package.
#
# Originals are kept. Output goes to videos_h264/, where non-AV1 files are symlinked
# so the directory is a complete drop-in replacement for videos/.
#
# Usage:
#   bash transcode_shortvid.sh            # 8 parallel jobs
#   JOBS=16 bash transcode_shortvid.sh    # more parallelism
set -o pipefail
cd "$(dirname "$0")"
source ./paths.sh

SRC="$SHORTVID_DIR/videos"
DST="$SHORTVID_DIR/videos_h264"
JOBS="${JOBS:-8}"
CRF="${CRF:-20}"
LIST="$SHORTVID_DIR/.codec_scan.txt"

if [ ! -d "$SRC" ]; then
  echo "ERROR: $SRC not found"
  exit 1
fi
command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg not found"; exit 1; }

mkdir -p "$DST"

echo "======================================================================"
echo "[1] Scanning video codecs"
echo "======================================================================"
echo "  source: $SRC"
find "$SRC" -name '*.mp4' -print0 | xargs -0 -P "$JOBS" -I{} sh -c '
  c=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$1" 2>/dev/null)
  echo "${c:-unknown}|$1"
' _ {} > "$LIST"

echo "  codec distribution:"
cut -d'|' -f1 "$LIST" | sort | uniq -c | sed 's/^/    /'

grep '^av1|' "$LIST" | cut -d'|' -f2- > "$LIST.av1"
grep -v '^av1|' "$LIST" | cut -d'|' -f2- > "$LIST.ok"
N_AV1=$(wc -l < "$LIST.av1")
N_OK=$(wc -l < "$LIST.ok")
echo "  av1 to transcode: $N_AV1"
echo "  already decodable: $N_OK"

echo
echo "======================================================================"
echo "[2] Linking non-AV1 files"
echo "======================================================================"
while IFS= read -r f; do
  [ -n "$f" ] || continue
  ln -sf "$f" "$DST/$(basename "$f")"
done < "$LIST.ok"
echo "  symlinked $N_OK files"

echo
echo "======================================================================"
echo "[3] Transcoding AV1 -> H.264 ($JOBS parallel jobs, crf $CRF)"
echo "======================================================================"
echo "  Audio is copied untouched when the codec allows it, so the audio track the"
echo "  model hears is bit-identical to the original. Only the video is re-encoded."

export DST CRF
cat "$LIST.av1" | xargs -P "$JOBS" -I{} sh -c '
  src="$1"
  out="$DST/$(basename "$src")"
  if [ -f "$out" ] && [ ! -L "$out" ]; then exit 0; fi   # already transcoded
  # First attempt: stream-copy the audio. Falls back to AAC if the source audio
  # codec cannot be stored in mp4.
  if ! ffmpeg -y -loglevel error -i "$src" \
       -c:v libx264 -crf "$CRF" -preset veryfast -pix_fmt yuv420p \
       -c:a copy "$out" 2>/dev/null; then
    ffmpeg -y -loglevel error -i "$src" \
       -c:v libx264 -crf "$CRF" -preset veryfast -pix_fmt yuv420p \
       -c:a aac -b:a 128k "$out" || echo "FAILED: $src"
  fi
' _ {}

echo
echo "======================================================================"
echo "[4] Verification"
echo "======================================================================"
N_OUT=$(find "$DST" \( -name '*.mp4' -o -type l \) | wc -l)
N_SRC=$(find "$SRC" -name '*.mp4' | wc -l)
echo "  source files : $N_SRC"
echo "  output files : $N_OUT"

echo "  checking decodability with the same OpenCV the model uses..."
python - <<'PYCHECK'
import os, random, sys
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2
d = os.path.join(os.environ["SHORTVID_DIR"], "videos_h264")
files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".mp4")]
random.seed(0)
sample = random.sample(files, min(40, len(files)))
bad = []
for f in sample:
    cap = cv2.VideoCapture(f)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        bad.append(f)
print("    checked %d files, %d still undecodable" % (len(sample), len(bad)))
for f in bad[:5]:
    print("      %s" % f)
sys.exit(1 if bad else 0)
PYCHECK
RC=$?

echo
if [ "$RC" = "0" ] && [ "$N_OUT" -ge "$N_SRC" ]; then
  echo "DONE. eval_omnivinci.py picks up videos_h264/ automatically."
else
  echo "INCOMPLETE - re-run this script; finished files are skipped."
  exit 1
fi
