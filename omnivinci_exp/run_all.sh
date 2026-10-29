#!/usr/bin/env bash
# Run the 4 remaining WorldSense runs one after another on a single GPU.
#
# Order is random/omnizip at 45% first, then both at 35%, so if the script is stopped
# halfway there is still a complete matched pair to compare.
#
# --resume is on: re-running the script picks up where an interrupted run stopped
# instead of starting it over.
#
# Usage:
#   bash run_all.sh          # GPU 3
#   GPU=0 bash run_all.sh    # another card
set -e
cd "$(dirname "$0")"

GPU="${GPU:-3}"
SUBSET="${SUBSET:-500}"

export CUDA_VISIBLE_DEVICES="$GPU"

run () {   # run <method> <target_retained>
  echo
  echo "=================================================================="
  echo "  $1 @ retained $2   (GPU $GPU)   started $(date +%H:%M)"
  echo "=================================================================="
  python eval_omnivinci.py \
    --benchmark worldsense \
    --method "$1" \
    --target-retained "$2" \
    --subset "$SUBSET" \
    --seed 0 \
    --resume
}

run random  0.45
run omnizip 0.45
run random  0.35
run omnizip 0.35

echo
echo "All 4 runs finished at $(date +%H:%M)."
