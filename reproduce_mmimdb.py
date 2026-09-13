"""Run the configured MM-IMDb reproduction from PyCharm or PowerShell.

This launcher does not modify the DCP source code. Choose this file for the
recommended RTX 4060 configuration (FP16, per-GPU micro-batch 2).
"""

from __future__ import print_function

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("DCP_PROJECT_ROOT", Path(__file__).resolve().parent)
).expanduser()
CODE_ROOT = Path(
    os.environ.get(
        "DCP_CODE_ROOT", PROJECT_ROOT / "code" / "Deep_Correlated_Prompting-main"
    )
).expanduser()
ARROW_ROOT = Path(
    os.environ.get("MMIMDB_DATA_ROOT", PROJECT_ROOT / "data" / "mmimdb" / "arrow")
).expanduser()
MISSING_TABLE_ROOT = Path(
    os.environ.get(
        "MISSING_TABLE_ROOT",
        PROJECT_ROOT / "data" / "mmimdb" / "missing_tables",
    )
).expanduser()
EXPERIMENT_ROOT = Path(
    os.environ.get(
        "DCP_EXPERIMENT_ROOT",
        PROJECT_ROOT / "experiments" / "dcp_mmimdb_reproduction",
    )
).expanduser()
RUN_LOG_DIR = EXPERIMENT_ROOT / "checkpoints"

PROFILE_NAME = "dcp_mmimdb_reproduction_seed0"
MICRO_BATCH = 2


def main():
    for path in (CODE_ROOT, ARROW_ROOT, MISSING_TABLE_ROOT):
        if not path.is_dir():
            raise FileNotFoundError(path)

    log_root = EXPERIMENT_ROOT / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / (PROFILE_NAME + ".txt")
    cache_root = Path(
        os.environ.get("DCP_CACHE_ROOT", PROJECT_ROOT / "cache")
    ).expanduser()

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(cache_root / "huggingface"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
    env.setdefault(
        "TRANSFORMERS_CACHE", str(cache_root / "huggingface" / "transformers")
    )
    env.setdefault("TORCH_HOME", str(cache_root / "torch"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("GIT_PYTHON_REFRESH", "quiet")

    args = [
        sys.executable,
        "run.py",
        "with",
        "task_finetune_mmimdb",
        "data_root=" + str(ARROW_ROOT),
        "missing_table_root=" + str(MISSING_TABLE_ROOT),
        "num_gpus=1",
        "num_nodes=1",
        "batch_size=256",
        "per_gpu_batchsize=" + str(MICRO_BATCH),
        "num_workers=0",
        "seed=0",
        "precision=16",
        "max_epoch=20",
        "max_steps=None",
        "fast_dev_run=False",
        "val_check_interval=0.2",
        "log_dir=" + str(RUN_LOG_DIR),
        "exp_name=" + PROFILE_NAME,
    ]

    print("DCP MM-IMDb reproduction")
    print("Profile:       ", PROFILE_NAME)
    print("Micro-batch:   ", MICRO_BATCH)
    print("Effective batch: 256")
    print("Epochs:        20")
    print("Progress:      Lightning percentage progress bar")
    print("Log file:      ", log_path)
    print("Checkpoint root:", EXPERIMENT_ROOT / "checkpoints")
    print("Started:       ", datetime.now().isoformat(timespec="seconds"))

    # Append so a later retry preserves the previous startup error and run log.
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n===== RUN " + datetime.now().isoformat(timespec="seconds") + " =====\n")
        process = subprocess.Popen(
            args,
            cwd=str(CODE_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    print("Finished with exit code:", return_code)
    print("Log file:", log_path)
    print("Checkpoints:", EXPERIMENT_ROOT / "checkpoints")
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
