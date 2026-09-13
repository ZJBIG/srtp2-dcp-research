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
MMIMDB_DATA_ROOT="${MMIMDB_DATA_ROOT:-../../data/mmimdb/arrow}"
MISSING_TABLE_ROOT="${MISSING_TABLE_ROOT:-../../data/mmimdb/missing_tables}"
CLIP_CACHE_ROOT="${CLIP_CACHE_ROOT:-../../cache/clip}"
ORIGINAL_DCP_PATH="${ORIGINAL_DCP_PATH:-../../experiments/dcp_mmimdb_reproduction/checkpoints/dcp_mmimdb_reproduction_seed0_seed0/version_2/checkpoints/epoch=3-step=507.ckpt}"

exec "$PYTHON_BIN" tools/aistation_preflight.py \
  --data-root "$MMIMDB_DATA_ROOT" \
  --missing-table-root "$MISSING_TABLE_ROOT" \
  --clip-cache-root "$CLIP_CACHE_ROOT" \
  --original-dcp-path "$ORIGINAL_DCP_PATH" \
  "$@"
