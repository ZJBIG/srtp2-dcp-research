import hashlib
import math
import re
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code" / "Deep_Correlated_Prompting-main"
METRICS = CODE / "pilot_logs" / "dcp_mmimdb_epoch1_pilot_002_seed0" / "version_0" / "metrics.csv"
CKPT = CODE / "pilot_logs" / "dcp_mmimdb_epoch1_pilot_002_seed0" / "version_0" / "checkpoints" / "last.ckpt"
LOG = ROOT / "experiments" / "dcp_mmimdb_epoch1_pilot_002" / "train_output.txt"


def finite_stats(series):
    nonempty = series.dropna()
    numeric = pd.to_numeric(nonempty, errors="coerce")
    finite = numeric[ numeric.map(math.isfinite) ]
    return {
        "rows": len(series),
        "nonempty": len(nonempty),
        "blank_as_pandas_nan": int(series.isna().sum()),
        "finite": len(finite),
        "nan": int(numeric.isna().sum()),
        "inf": int((numeric.abs() == float("inf")).sum()),
        "first": finite.iloc[0] if len(finite) else None,
        "last": finite.iloc[-1] if len(finite) else None,
        "min": finite.min() if len(finite) else None,
        "max": finite.max() if len(finite) else None,
    }


def tensor_nonfinite(obj):
    tensors = []
    if isinstance(obj, dict):
        for value in obj.values():
            tensors.extend(tensor_nonfinite(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            tensors.extend(tensor_nonfinite(value))
    elif torch.is_tensor(obj):
        return [not bool(torch.isfinite(obj.detach().float()).all())]
    return tensors


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def main():
    frame = pd.read_csv(METRICS)
    print(f"metrics_rows={len(frame)}")
    for column in frame.columns:
        lower = column.lower()
        if "loss" in lower or "lr" in lower:
            print(f"METRIC_COLUMN {column} {finite_stats(frame[column])}")
    print("STEP_EPOCH_COLUMNS")
    for column in frame.columns:
        if column in {"step", "epoch"}:
            print(column, finite_stats(frame[column]))

    checkpoint = torch.load(str(CKPT), map_location="cpu")
    state_dict = checkpoint.get("state_dict", {})
    optimizer_states = checkpoint.get("optimizer_states", [])
    print(f"checkpoint_epoch={checkpoint.get('epoch')}")
    print(f"checkpoint_global_step={checkpoint.get('global_step')}")
    print(f"checkpoint_keys={list(checkpoint.keys())}")
    print(f"state_dict_tensors={sum(torch.is_tensor(v) for v in state_dict.values())}")
    print(f"state_dict_nonfinite={sum(tensor_nonfinite(state_dict))}")
    print(f"optimizer_state_count={len(optimizer_states)}")
    print(f"optimizer_state_nonfinite={sum(tensor_nonfinite(optimizer_states))}")
    scaler = checkpoint.get("native_amp_scaling_state")
    print(f"amp_scaler_state={scaler}")

    raw_bytes = LOG.read_bytes()
    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        raw = raw_bytes.decode("utf-16", errors="replace")
    else:
        raw = raw_bytes.decode("utf-8", errors="replace")
    print(f"progress_loss_nan_lines={len(re.findall(r'loss=nan', raw, flags=re.I))}")
    print(f"has_traceback={bool(re.search(r'Traceback', raw, flags=re.I))}")
    print(f"has_cuda_error={bool(re.search(r'CUDA error|out of memory', raw, flags=re.I))}")

    paths = [
        CODE / "run.py",
        CODE / "clip" / "datamodules" / "multitask_datamodule.py",
        ROOT / "data" / "mmimdb" / "missing_tables" / "mmimdb_dev_missing_both_07.pt",
        ROOT / "data" / "mmimdb" / "missing_tables" / "mmimdb_test_missing_both_07.pt",
        ROOT / "data" / "mmimdb" / "missing_tables" / "mmimdb_train_missing_both_07.pt",
        ROOT / "cache" / "runtime_profile" / ".cache" / "clip" / "ViT-B-16.pt",
    ]
    for path in paths:
        print(f"SHA256 {path} {sha256(path)}")


if __name__ == "__main__":
    main()
