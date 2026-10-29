#!/usr/bin/env bash
# Re-run the two omnizip runs with the fixed video compressor.
# random and baseline are unaffected by the fix, so their numbers still stand.
#
# Output goes to log files: two parallel progress bars on one terminal are unreadable.
#
# Usage:
#   bash run_all_v2.sh
set -e
cd "$(dirname "$0")"

SUBSET="${SUBSET:-500}"

echo "Starting 2 runs at $(date +%H:%M), ~2 hours."

CUDA_VISIBLE_DEVICES=0 python eval_omnivinci.py \
  --benchmark worldsense --method omnizip --target-retained 0.35 \
  --subset "$SUBSET" --seed 0 --resume \
  --run-name omnizip_r35_v2_n500 > log_omnizip_r35_v2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python eval_omnivinci.py \
  --benchmark worldsense --method omnizip --target-retained 0.45 \
  --subset "$SUBSET" --seed 0 --resume \
  --run-name omnizip_r45_v2_n500 > log_omnizip_r45_v2.txt 2>&1 &

wait

echo "Done at $(date +%H:%M)."
echo "  results/omnivinci-9b/worldsense/omnizip_r35_v2_n500/"
echo "  results/omnivinci-9b/worldsense/omnizip_r45_v2_n500/"
