#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${DCP_VENV_DIR:-}" && -x "${DCP_VENV_DIR}/bin/python" ]]; then
    PYTHON_BIN="${DCP_VENV_DIR}/bin/python"
  elif [[ -x "$SCRIPT_DIR/.venv-aistation/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv-aistation/bin/python"
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
MMIMDB_DATA_ROOT="${MMIMDB_DATA_ROOT:-../../data/mmimdb/arrow}"
MISSING_TABLE_ROOT="${MISSING_TABLE_ROOT:-../../data/mmimdb/missing_tables}"
CLIP_CACHE_ROOT="${CLIP_CACHE_ROOT:-../../cache/clip}"
ORIGINAL_DCP_PATH="${ORIGINAL_DCP_PATH:-../../experiments/dcp_mmimdb_reproduction/checkpoints/dcp_mmimdb_reproduction_seed0_seed0/version_2/checkpoints/epoch=3-step=507.ckpt}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-../../outputs/direct_task_full_logs}"
EXP_NAME="${EXP_NAME:-relative_bounded_direct_task_20ep}"

exec "$PYTHON_BIN" run.py with \
  task_finetune_mmimdb \
  reliability_learning \
  "data_root=$MMIMDB_DATA_ROOT" \
  "missing_table_root=$MISSING_TABLE_ROOT" \
  "clip_cache_root=$CLIP_CACHE_ROOT" \
  "original_dcp_path=$ORIGINAL_DCP_PATH" \
  "log_dir=$TRAIN_LOG_DIR" \
  "exp_name=$EXP_NAME" \
  reliability_importance_mode=relative \
  gate_supervision_mode=direct_task \
  adapter_train_epochs=10 \
  reliability_predictor_lr=0.001 \
  reliability_adapter_lr=0.003 \
  reliability_gate_lr=0.001 \
  reliability_weight_decay=0.0001 \
  num_gpus=1 \
  num_nodes=1 \
  num_workers=0 \
  per_gpu_batchsize=4 \
  batch_size=64 \
  precision=16 \
  max_epoch=20 \
  max_steps=None \
  fast_dev_run=False \
  val_check_interval=1.0 \
  seed=0 \
  "$@"
