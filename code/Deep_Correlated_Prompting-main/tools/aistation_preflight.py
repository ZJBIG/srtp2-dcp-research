"""Validate an AIStation mount and Python image without loading the model."""

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SOURCE_ROOT.parents[1]
EXPECTED_ARROW = {
    "mmimdb_train.arrow": 15552,
    "mmimdb_dev.arrow": 2608,
    "mmimdb_test.arrow": 7799,
}
EXPECTED_COLUMNS = ["image", "plots", "label", "genres", "image_id", "split"]
EXPECTED_MISSING_TABLES = (
    "mmimdb_train_missing_both_07.pt",
    "mmimdb_dev_missing_both_07.pt",
    "mmimdb_test_missing_both_07.pt",
)


def _default_path(environment_name, relative_path):
    configured = os.environ.get(environment_name)
    if configured:
        return Path(configured).expanduser()
    return WORKSPACE_ROOT / relative_path


def _require_file(path, label):
    if not path.is_file():
        raise FileNotFoundError("{} not found: {}".format(label, path))
    if path.stat().st_size <= 0:
        raise RuntimeError("{} is empty: {}".format(label, path))
    print("{}={}".format(label, path.resolve()))


def _package_version(module_name):
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")


def _check_arrow_files(data_root):
    import pyarrow as pa

    for filename, expected_rows in EXPECTED_ARROW.items():
        path = data_root / filename
        _require_file(path, "ARROW_FILE")
        with pa.memory_map(str(path), "r") as source:
            reader = pa.ipc.RecordBatchFileReader(source)
            rows = sum(
                reader.get_batch(index).num_rows
                for index in range(reader.num_record_batches)
            )
            columns = reader.schema.names
        if rows != expected_rows:
            raise RuntimeError(
                "{} row mismatch: actual {}, expected {}".format(
                    filename, rows, expected_rows
                )
            )
        if columns != EXPECTED_COLUMNS:
            raise RuntimeError(
                "{} columns mismatch: actual {}, expected {}".format(
                    filename, columns, EXPECTED_COLUMNS
                )
            )
        print("ARROW_OK={} rows={}".format(filename, rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_default_path("MMIMDB_DATA_ROOT", Path("data/mmimdb/arrow")),
    )
    parser.add_argument(
        "--missing-table-root",
        type=Path,
        default=_default_path(
            "MISSING_TABLE_ROOT", Path("data/mmimdb/missing_tables")
        ),
    )
    parser.add_argument(
        "--clip-cache-root",
        type=Path,
        default=_default_path("CLIP_CACHE_ROOT", Path("cache/clip")),
    )
    parser.add_argument(
        "--original-dcp-path",
        type=Path,
        default=_default_path(
            "ORIGINAL_DCP_PATH",
            Path(
                "experiments/dcp_mmimdb_reproduction/checkpoints/"
                "dcp_mmimdb_reproduction_seed0_seed0/version_2/checkpoints/"
                "epoch=3-step=507.ckpt"
            ),
        ),
    )
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    missing_root = args.missing_table_root.expanduser()
    clip_root = args.clip_cache_root.expanduser()
    original_checkpoint = args.original_dcp_path.expanduser()

    if not data_root.is_dir():
        raise FileNotFoundError("MMIMDB data root not found: {}".format(data_root))
    if not missing_root.is_dir():
        raise FileNotFoundError(
            "Missing-table root not found: {}".format(missing_root)
        )
    if not clip_root.is_dir():
        raise FileNotFoundError("CLIP cache root not found: {}".format(clip_root))

    print("PYTHON={}".format(sys.executable))
    print("PYTHON_VERSION={}".format(platform.python_version()))
    print("PLATFORM={}".format(platform.platform()))
    for module_name in (
        "numpy",
        "PIL",
        "pyarrow",
        "pytorch_lightning",
        "sacred",
        "timm",
        "torchmetrics",
        "transformers",
    ):
        print(
            "PACKAGE_{}={}".format(
                module_name.upper(), _package_version(module_name)
            )
        )

    import torch
    import torchvision

    print("TORCH={}".format(torch.__version__))
    print("TORCHVISION={}".format(torchvision.__version__))
    print("CUDA_BUILD={}".format(torch.version.cuda))
    print("CUDA_AVAILABLE={}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("CUDA_DEVICE_COUNT={}".format(torch.cuda.device_count()))
        print("CUDA_DEVICE_NAME={}".format(torch.cuda.get_device_name(0)))
        print("CUDA_CAPABILITY={}".format(torch.cuda.get_device_capability(0)))
    elif not args.allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable. Choose a GPU-enabled AIStation PyTorch image "
            "or use --allow-cpu only for file validation."
        )

    _check_arrow_files(data_root)
    for filename in EXPECTED_MISSING_TABLES:
        _require_file(missing_root / filename, "MISSING_TABLE")
    _require_file(clip_root / "ViT-B-16.pt", "CLIP_WEIGHT")
    _require_file(original_checkpoint, "ORIGINAL_DCP_CHECKPOINT")
    _require_file(SOURCE_ROOT / "run.py", "RUN_ENTRY")
    _require_file(SOURCE_ROOT / "run_aistation.sh", "AISTATION_LAUNCHER")
    print("AISTATION_PREFLIGHT=OK")


if __name__ == "__main__":
    main()
