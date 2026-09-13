#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
DCP_VENV_DIR="${DCP_VENV_DIR:-.venv-aistation}"

if [[ "$DCP_VENV_DIR" != /* ]]; then
  DCP_VENV_DIR="$SCRIPT_DIR/$DCP_VENV_DIR"
fi

PYTHON_VERSION="$($PYTHON_BOOTSTRAP -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')"
if [[ "$PYTHON_VERSION" != "3.8" && "${DCP_ALLOW_UNTESTED_PYTHON:-0}" != "1" ]]; then
  echo "Expected Python 3.8, found $PYTHON_VERSION." >&2
  echo "Choose an AIStation PyTorch image with Python 3.8, or explicitly set DCP_ALLOW_UNTESTED_PYTHON=1." >&2
  exit 1
fi

"$PYTHON_BOOTSTRAP" -c 'import torch, torchvision; print("BASE_TORCH={} BASE_TORCHVISION={}".format(torch.__version__, torchvision.__version__))'

if [[ ! -x "$DCP_VENV_DIR/bin/python" ]]; then
  "$PYTHON_BOOTSTRAP" -m venv --system-site-packages "$DCP_VENV_DIR"
fi

VENV_PYTHON="$DCP_VENV_DIR/bin/python"
PIP_SOURCE_ARGS=()
if [[ -n "${DCP_WHEELHOUSE:-}" ]]; then
  PIP_SOURCE_ARGS+=(--no-index --find-links "$DCP_WHEELHOUSE")
fi

"$VENV_PYTHON" -m pip install "${PIP_SOURCE_ARGS[@]}" --upgrade "pip<25" wheel
"$VENV_PYTHON" -m pip install "${PIP_SOURCE_ARGS[@]}" -r requirements_aistation.txt
"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" -c 'import numpy, PIL, pyarrow, pytorch_lightning, sacred, timm, torch, torchmetrics, torchvision, transformers; print("ENV_READY python={} torch={} torchvision={} lightning={} cuda={}".format(__import__("sys").version.split()[0], torch.__version__, torchvision.__version__, pytorch_lightning.__version__, torch.cuda.is_available()))'

echo "DCP virtual environment is ready: $DCP_VENV_DIR"
echo "For this shell: export DCP_VENV_DIR=\"$DCP_VENV_DIR\""
echo "Then run: bash aistation_preflight.sh"
