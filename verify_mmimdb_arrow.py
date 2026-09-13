import argparse
import os
from pathlib import Path

import pyarrow as pa

DEFAULT_ROOT = Path(
    os.environ.get(
        "MMIMDB_DATA_ROOT",
        Path(__file__).resolve().parent / "data" / "mmimdb" / "arrow",
    )
).expanduser()

EXPECTED = {
    "mmimdb_train.arrow": 15552,
    "mmimdb_dev.arrow": 2608,
    "mmimdb_test.arrow": 7799,
}

EXPECTED_COLUMNS = ["image", "plots", "label", "genres", "image_id", "split"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    root = parser.parse_args().data_root.expanduser()

    for filename, expected_rows in EXPECTED.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)

        source = pa.memory_map(str(path), "r")
        reader = pa.ipc.RecordBatchFileReader(source)
        table = reader.read_all()

        print("=" * 60)
        print("file:", filename)
        print("size:", path.stat().st_size)
        print("rows:", table.num_rows)
        print("columns:", table.num_columns)
        print("fields:", table.column_names)

        if table.num_rows != expected_rows:
            raise RuntimeError(
                f"{filename} row count mismatch: "
                f"actual {table.num_rows}, expected {expected_rows}"
            )
        if table.column_names != EXPECTED_COLUMNS:
            raise RuntimeError(
                f"{filename} fields mismatch: "
                f"actual {table.column_names}, expected {EXPECTED_COLUMNS}"
            )

    print("=" * 60)
    print("All three Arrow files verified successfully")


if __name__ == "__main__":
    main()
