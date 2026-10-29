#!/usr/bin/env bash
# Download the 3 benchmark datasets into $DATA_DIR (outside the repo, so rsync
# never touches them).
# VideoMME was dropped: 389.4 GB and a video-only benchmark, of little value for an
# audio-guided method. See EXPERIMENTS.md section 1.
set -euo pipefail
cd "$(dirname "$0")"
source ./paths.sh

# Only enable hf_transfer when the package is actually installed.
# - Setting the flag without the package makes huggingface_hub fail immediately.
# - hf_transfer CANNOT resume a partially downloaded file: it deletes and restarts it.
if python -c "import hf_transfer" 2>/dev/null; then
  HFT_ON=1
  echo "hf_transfer: available (parallel byte-ranges within a single file)"
  echo "             Partial files get restarted from scratch; finished files are skipped."
else
  HFT_ON=0
  echo "hf_transfer: not installed -> default backend, can resume mid-file"
fi

# Warn when not inside tmux/screen: an SSH drop kills the download process.
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ]; then
  echo
  echo "WARNING: not running inside tmux/screen. An SSH drop will kill the download."
  echo "         Better: tmux new -s dl   then re-run this script."
  echo "         (Downloaded data survives; re-running resumes - you only lose time.)"
  echo
fi

# 54.1 GB downloaded + ~18 GB for unzipping WorldSense (11 zip files) + headroom
need_gb=90
avail_gb=$(df -BG --output=avail "$(dirname "$DATA_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
echo "Free space: ${avail_gb} GB (need >= ${need_gb} GB)"
if [ "${avail_gb:-0}" -lt "$need_gb" ]; then
  echo "ERROR: not enough space. Free some up and re-run."; exit 1
fi

# Parallelism is set per mode; see the notes inside dl().
# If /mnt/hdd2 is a spinning disk and iostat shows %util near 100, drop MAX_WORKERS_MANY to 4.
echo "workers: big=${MAX_WORKERS_BIG:-4}  many=${MAX_WORKERS_MANY:-8}"
echo "         (override: MAX_WORKERS_MANY=4 bash download_data.sh)"

# HF quota is 1000 API requests per 5 minutes. Each file costs a few requests, so for a
# repo with thousands of files THE QUOTA is the hard ceiling, not bandwidth:
#   ~3300 files x ~3 requests = ~10000 requests -> at least ~50 minutes.
# More workers does not shorten this, it only hits 429 sooner.
#
# The observed 429 came from the 'xet-read-token' endpoint. Disabling xet removes the
# per-file token refresh and cuts the request count. Set to 0 to use xet again.
DISABLE_XET="${DISABLE_XET:-1}"

# 'huggingface-cli' is deprecated in favour of 'hf'. Use 'hf' when available.
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
else
  HF_CLI="huggingface-cli"
fi
echo "cli: $HF_CLI"

mkdir -p "$DATA_DIR"

dl () {  # dl <hf_repo_id> <dest_dir> <label> <big|many>
  echo
  echo "=== $3 -> $2 ==="

  # Pick the mode from the SHAPE of the data instead of enabling hf_transfer everywhere:
  #
  #   big  = few files, each very large (WorldSense: 11 zips of ~2 GB).
  #          hf_transfer splits one file into parallel byte-ranges -> much faster.
  #          Trade-off: an interrupted file is deleted and restarted.
  #
  #   many = thousands of small files (AVUT ~3287, ShortVid ~1003, each 2-7 MB).
  #          hf_transfer gains nothing here (files too small to split) and costs resume.
  #          The bottleneck is per-file latency -> more files in parallel is what helps.
  local hft workers
  if [ "$4" = "big" ] && [ "$HFT_ON" = "1" ]; then
    hft=1; workers="${MAX_WORKERS_BIG:-4}"
  else
    hft=0; workers="${MAX_WORKERS_MANY:-8}"
  fi
  echo "    mode: $4 | hf_transfer=$hft | workers=$workers | disable_xet=$DISABLE_XET"

  local log tries max_tries rc wait_s
  log="$(mktemp)"
  tries=0
  max_tries=12

  while true; do
    set +e
    HF_HUB_ENABLE_HF_TRANSFER=$hft HF_HUB_DISABLE_XET=$DISABLE_XET \
      $HF_CLI download "$1" --repo-type dataset --local-dir "$2" \
      --max-workers "$workers" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e

    # Do NOT trust the exit code. When rate-limited, `hf download` prints
    # "Returning existing local_dir ... as remote repo cannot be accessed"
    # and exits 0 - reporting success while having downloaded nothing.
    if [ "$rc" = "0" ] && ! grep -q "cannot be accessed" "$log"; then
      echo "  DONE: $3"
      rm -f "$log"
      return 0
    fi

    tries=$((tries + 1))
    if [ "$tries" -ge "$max_tries" ]; then
      echo "  GIVING UP after $max_tries attempts. Re-run later; downloaded data is kept."
      rm -f "$log"
      return 1
    fi

    if grep -qE "429|Too Many Requests|rate limit" "$log"; then
      # The HF quota window is 5 minutes. Retrying after 30s only burns more quota.
      wait_s=360
      echo
      echo "  === RATE LIMITED (429) ==="
      echo "  Quota: 1000 API requests / 5 min. Waiting ${wait_s}s for the window to reset."
      echo "  If this keeps happening: MAX_WORKERS_MANY=4 bash download_data.sh"
      echo "  Also check you are logged in: hf auth whoami  (anonymous limits are tighter)"
    else
      wait_s=30
    fi
    echo "  Attempt $tries/$max_tries failed. Waiting ${wait_s}s before retry..."
    sleep "$wait_s"
  done
}

# WorldSense first: it is the only benchmark with a published OmniVinci number (48.2),
# so it is what validates the baseline.
dl honglyhly/WorldSense        "$WORLDSENSE_DIR" "WorldSense  18.1 GB" big
dl tsinghua-ee/AVUTBenchmark   "$AVUT_DIR"       "AVUT        24.0 GB" many
dl TencentARC/ShortVid-Bench   "$SHORTVID_DIR"   "ShortVid    12.0 GB" many

echo
echo "======================================================================"
echo "Verifying actual file counts"
echo "======================================================================"
# Count real files instead of trusting the "DONE" lines above: under a 429,
# `hf download` can exit 0 without having downloaded anything new.
verify () {  # verify <dir> <expected_count> <label>
  local n
  n=$(find "$1" -type f -not -path "*/.cache/*" 2>/dev/null | wc -l)
  printf "  %-12s %6d / %6d files" "$3" "$n" "$2"
  if [ "$n" -lt "$2" ]; then
    echo "   <-- $(( $2 - n )) MISSING, re-run this script"
    return 1
  fi
  echo "   OK"
}

INCOMPLETE=0
verify "$WORLDSENSE_DIR" 15   "WorldSense" || INCOMPLETE=1
verify "$AVUT_DIR"       3287 "AVUT"       || INCOMPLETE=1
verify "$SHORTVID_DIR"   1003 "ShortVid"   || INCOMPLETE=1
echo "  (expected counts come from the HF API; treat them as a reference, not gospel)"

if [ "$INCOMPLETE" = "1" ]; then
  echo
  echo "INCOMPLETE. Wait out the rate-limit window (5 min), then re-run this script."
  echo "Skipping the WorldSense unzip step to avoid extracting partial archives."
  exit 1
fi

echo
echo "======================================================================"
echo "Extracting WorldSense"
echo "======================================================================"
# WorldSense ships as 11 zip files (~1.6-2 GB each) and is not directly usable.
# They must be extracted into videos/.
if ls "$WORLDSENSE_DIR"/worldsense_videos_*.zip >/dev/null 2>&1; then
  mkdir -p "$WORLDSENSE_DIR/videos"
  for z in "$WORLDSENSE_DIR"/worldsense_videos_*.zip; do
    marker="$WORLDSENSE_DIR/videos/.done_$(basename "$z" .zip)"
    if [ -f "$marker" ]; then
      echo "  skip (already extracted): $(basename "$z")"
      continue
    fi
    echo "  extracting $(basename "$z") ..."
    if unzip -q -o "$z" -d "$WORLDSENSE_DIR/videos"; then
      touch "$marker"
    else
      echo "  ERROR extracting $(basename "$z") - the archive may be incomplete."
      echo "  Re-run this script to resume the download first."
      exit 1
    fi
  done
  # subtitles (small, 3 MB)
  [ -f "$WORLDSENSE_DIR/worldsense_subtitles.zip" ] && \
    unzip -q -o "$WORLDSENSE_DIR/worldsense_subtitles.zip" -d "$WORLDSENSE_DIR/subtitles"
  echo "  Videos after extraction: $(find "$WORLDSENSE_DIR/videos" -name '*.mp4' | wc -l)"
  echo
  echo "  Once verified, the archives can be deleted to reclaim ~18 GB:"
  echo "    rm $WORLDSENSE_DIR/worldsense_videos_*.zip"
  echo "  (only delete when you are sure you will not need to re-download)"
else
  echo "  No zip files found - WorldSense may not be downloaded yet."
fi

echo
echo "=== Annotation files ==="
# File names expected by the three eval scripts in this repo:
for f in "$WORLDSENSE_DIR/worldsense_qa.json" \
         "$AVUT_DIR/AV_Human_data.json" \
         "$SHORTVID_DIR/ShortVid-Bench-1k.json"; do
  if [ -f "$f" ]; then echo "  OK      $f"
  else echo "  MISSING $f  <- find the real name: find $(dirname "$f") -name '*.json' | head"; fi
done

echo
echo "DONE. Next: python probe_omnivinci.py --video <path_to_video>"
