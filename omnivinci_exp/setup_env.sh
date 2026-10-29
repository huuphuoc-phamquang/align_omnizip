#!/usr/bin/env bash
# Verify the existing OmniVinci conda env is ready for the experiments.
#
# Principle: this env already runs OmniVinci -> do NOT install over it.
# The script only REPORTS what is missing and prints the pip command, leaving the
# decision to you. A careless `pip install` can bump torch/transformers to a
# different version and break a working env.
#
# Usage:
#   bash setup_env.sh                    # use the active env (or 'omnivinci')
#   CONDA_ENV_NAME=abc bash setup_env.sh # target a different env
#   bash setup_env.sh --install-optional # also install hf_transfer
#
# Do NOT use `set -u`: conda's activate/deactivate hooks (e.g.
# ~cuda-nvcc_deactivate.sh) reference unset variables such as CUDAARCHS_BACKUP.
# Under `set -u` they raise "unbound variable" and make `conda activate` return a
# non-zero code even though the env is perfectly fine.
set -o pipefail
cd "$(dirname "$0")"
source ./paths.sh

INSTALL_OPTIONAL=0
[ "${1:-}" = "--install-optional" ] && INSTALL_OPTIONAL=1

echo "======================================================================"
echo "[0] GPUs and paths"
echo "======================================================================"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || \
  echo "  (nvidia-smi unavailable)"
echo "  env       : $CONDA_ENV_NAME"
echo "  model dir : $OV_MODEL_DIR"

if [ ! -f "$OV_MODEL_DIR/modeling_vila.py" ]; then
  echo "  ERROR: $OV_MODEL_DIR/modeling_vila.py not found"
  echo "         The model dir must be the HF snapshot of nvidia/omnivinci."
  exit 1
fi
echo "  OK: modeling_vila.py found"

echo
echo "======================================================================"
echo "[1] Python environment"
echo "======================================================================"
# Prefer whatever env is already active. This avoids calling `conda activate`
# altogether, since conda's hooks tend to fail inside scripts.
CUR_ENV="${CONDA_DEFAULT_ENV:-}"

if [ -n "$CUR_ENV" ] && [ "$CUR_ENV" != "base" ]; then
  echo "  Using active env: $CUR_ENV  (script does not call conda activate)"
  if [ "$CUR_ENV" != "$CONDA_ENV_NAME" ]; then
    echo "  Note: differs from CONDA_ENV_NAME='$CONDA_ENV_NAME' set in paths.sh."
    echo "        Continuing with '$CUR_ENV'."
  fi
else
  [ "$CUR_ENV" = "base" ] && echo "  Currently in 'base' - not the env to run the model."
  echo "  Trying to activate '$CONDA_ENV_NAME'..."
  CONDA_BASE="$(conda info --base 2>/dev/null)" || { echo "ERROR: conda not found"; exit 1; }
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  set +u
  conda activate "$CONDA_ENV_NAME"
  set -o pipefail
  if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]; then
    echo
    echo "ERROR: could not enter env '$CONDA_ENV_NAME'."
    echo "Simplest fix: run 'conda activate $CONDA_ENV_NAME' in your own shell,"
    echo "then re-run this script - it will use the active env."
    echo "Available envs:"; conda env list
    exit 1
  fi
fi
echo "  python : $(which python)"
echo "  version: $(python --version 2>&1)"

echo
echo "======================================================================"
echo "[2] Package check"
echo "======================================================================"
python - <<'PYCHECK'
import importlib.util as iu
import sys

# core = required by OmniVinci itself; any of these missing means the env is not ready
core = ["torch", "transformers", "decord", "librosa", "soundfile",
        "cv2", "whisper", "PIL", "numpy", "kaldiio", "accelerate"]
# extra = additionally needed by the experiment scripts
extra = ["tqdm", "huggingface_hub"]
# optional = nice to have, everything still runs without them
optional = ["hf_transfer", "flash_attn"]


def check(names, label):
    """Three states: OK / MISSING (not installed) / BROKEN (installed, import fails).

    The distinction matters: running `pip install` on a BROKEN package fixes nothing,
    because the usual cause is an ABI mismatch (module built against NumPy 1.x being
    imported under NumPy 2.x).
    """
    missing, broken = [], []
    print("  -- %s --" % label)
    for n in names:
        if iu.find_spec(n) is None:
            print("    MISSING  %-18s (not installed)" % n)
            missing.append(n)
            continue
        try:
            m = __import__(n)
            print("    OK       %-18s %s" % (n, getattr(m, "__version__", "")))
        except Exception as e:
            msg = str(e).replace("\n", " ")[:120]
            print("    BROKEN   %-18s installed but import failed: %s" % (n, msg))
            broken.append((n, str(e)))
    return missing, broken


miss_core, broke_core = check(core, "core (OmniVinci)")
miss_extra, broke_extra = check(extra, "extra (experiment scripts)")
miss_opt, broke_opt = check(optional, "optional")
all_broken = broke_core + broke_extra

print()
import torch
print("  torch %s | cuda build %s | available %s | %d GPU(s)" % (
    torch.__version__, torch.version.cuda, torch.cuda.is_available(),
    torch.cuda.device_count()))
if torch.version.cuda:
    major = int(torch.version.cuda.split(".")[0])
    if major >= 13:
        print("  WARNING: torch is a CUDA 13 build -> needs driver r580+, will not run on 12.8.")
    else:
        print("  OK: CUDA 12.x build, compatible with driver 12.8.")

import numpy

# --- NumPy ABI mismatch diagnosis ---
# Compiled modules (cv2, decord, torch...) built against NumPy 1.x fail under NumPy 2.x
# with "_ARRAY_API not found" or "numpy.core.multiarray failed to import".
ABI_SIGNS = ("_ARRAY_API", "numpy.core.multiarray failed",
             "numpy.dtype size changed", "compiled using NumPy 1.x")
abi_hits = [n for n, e in all_broken if any(s in e for s in ABI_SIGNS)]

np_major = int(numpy.__version__.split(".")[0])
tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])

if abi_hits or (np_major >= 2 and tv < (2, 4)):
    print()
    print("  === NUMPY ABI MISMATCH ===")
    print("  numpy is %s." % numpy.__version__)
    if abi_hits:
        print("  Modules failing due to ABI mismatch: %s" % ", ".join(abi_hits))
    if tv < (2, 4):
        print("  torch %s is also a NumPy 1.x build (torch supports NumPy 2 only from 2.4)."
              % torch.__version__)
    print()
    print("  Fix by DOWNGRADING numpy, not by reinstalling the modules:")
    print("    pip install \"numpy==1.26.4\"")
    print()
    print("  Reason: all the modules above were compiled in the NumPy 1.x era. Upgrading")
    print("  one module to a NumPy 2 build fixes only that one; torch stays mismatched.")
    sys.exit(4)

if all_broken:
    print()
    print("  %d package(s) installed but failing to import (not an ABI mismatch):"
          % len(all_broken))
    for n, e in all_broken:
        print("    %s: %s" % (n, str(e).replace("\n", " ")[:200]))
    sys.exit(4)

# module name -> pip name
pipname = {"cv2": "opencv-python", "PIL": "pillow", "whisper": "openai-whisper",
           "huggingface_hub": "huggingface_hub", "flash_attn": "flash-attn"}
need = miss_core + miss_extra
if need:
    print()
    print("  %d package(s) MISSING. Install command (run it yourself, read it first):"
          % len(need))
    print("    pip install --upgrade-strategy only-if-needed " +
          " ".join(pipname.get(n, n) for n in need))
    print("  Afterwards re-run this script and re-check the torch/transformers versions.")
    sys.exit(3)

print()
print("  All required packages present.")
if miss_opt:
    print("  Optional still missing: " + ", ".join(miss_opt))
PYCHECK
CHECK_RC=$?

if [ "$CHECK_RC" = "3" ]; then
  echo
  echo "STOP: missing packages. See the pip command above."
  exit 1
elif [ "$CHECK_RC" = "4" ]; then
  echo
  echo "STOP: some packages are installed but fail to import."
  echo "If it is a NumPy ABI mismatch: run the pip command above, then re-run this script."
  echo "The model must load before going further - OmniVinci's media.py imports cv2 on"
  echo "its very first lines."
  exit 1
elif [ "$CHECK_RC" != "0" ]; then
  echo "STOP: package check failed (rc=$CHECK_RC)."
  exit 1
fi

if [ "$INSTALL_OPTIONAL" = "1" ]; then
  echo
  echo "======================================================================"
  echo "[2b] Installing optional packages (--install-optional)"
  echo "======================================================================"
  BEFORE=$(pip freeze 2>/dev/null | grep -E '^(torch|transformers|numpy)==' | sort)
  pip install --upgrade-strategy only-if-needed hf_transfer
  AFTER=$(pip freeze 2>/dev/null | grep -E '^(torch|transformers|numpy)==' | sort)
  if [ "$BEFORE" != "$AFTER" ]; then
    echo
    echo "  WARNING: pip changed core package versions:"
    diff <(echo "$BEFORE") <(echo "$AFTER") || true
    echo "  Re-run the baseline to make sure the env is still correct."
  else
    echo "  OK: torch/transformers/numpy untouched."
  fi
fi

echo
echo "======================================================================"
echo "[3] Smoke test: load OmniVinci"
echo "======================================================================"
python ./probe_omnivinci.py --smoke-only || {
  echo "Smoke test FAILED - read the log above before continuing."; exit 1; }

echo
echo "DONE. Next: bash download_data.sh"
