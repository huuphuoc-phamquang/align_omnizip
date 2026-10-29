#!/usr/bin/env bash
# Shared paths. Override any of them via environment variables.
# Usage: source paths.sh

export NGHIA_ROOT="${NGHIA_ROOT:-/mnt/hdd2/trung/projects/new_users/Nghia}"

export OV_MODEL_DIR="${OV_MODEL_DIR:-$NGHIA_ROOT/omnivinci}"   # HF snapshot of nvidia/omnivinci
export OZ_REPO_DIR="${OZ_REPO_DIR:-$NGHIA_ROOT/OmniZip}"        # this repo on the server
export DATA_DIR="${DATA_DIR:-$NGHIA_ROOT/data}"                 # datasets, OUTSIDE the repo
export RESULTS_DIR="${RESULTS_DIR:-$OZ_REPO_DIR/results}"

export WORLDSENSE_DIR="${WORLDSENSE_DIR:-$DATA_DIR/worldsense}"
export AVUT_DIR="${AVUT_DIR:-$DATA_DIR/avut}"
export SHORTVID_DIR="${SHORTVID_DIR:-$DATA_DIR/shortvid}"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-omnivinci}"
