"""Deterministic, traceable missing-modality table management."""

import math
import os
import random

import torch


def _number_token(value):
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def missing_table_filename(
    dataset_name, missing_type, missing_ratio, both_ratio, seed
):
    return (
        "{}_missing_{}_r{}_both{}_seed{}.pt".format(
            dataset_name,
            missing_type,
            _number_token(missing_ratio),
            _number_token(both_ratio),
            int(seed),
        )
    )


def _generation_seed(dataset_name, split, seed):
    dataset_offset = sum(
        (index + 1) * ord(character)
        for index, character in enumerate(str(dataset_name))
    )
    split_offset = {"train": 0, "val": 1000003, "test": 2000003}[split]
    return (int(seed) * 3000017 + dataset_offset + split_offset) % (2 ** 32)


def _expected_metadata(
    dataset_name,
    split,
    total_num,
    missing_ratio,
    missing_type,
    both_ratio,
    seed,
):
    identity = missing_table_filename(
        dataset_name, missing_type, missing_ratio, both_ratio, seed
    )
    return {
        "identity": identity,
        "dataset": str(dataset_name),
        "split": str(split),
        "total_num": int(total_num),
        "missing_ratio": float(missing_ratio),
        "missing_type": str(missing_type),
        "both_ratio": float(both_ratio),
        "seed": int(seed),
        "generation_seed": _generation_seed(dataset_name, split, seed),
    }


def _validate_configuration(total_num, missing_ratio, missing_type, both_ratio):
    if int(total_num) < 0:
        raise ValueError("total_num must be non-negative")
    if not 0.0 <= float(missing_ratio) <= 1.0:
        raise ValueError("missing_ratio must lie in [0, 1]")
    if missing_type not in ("text", "image", "both"):
        raise ValueError("missing_type must be text, image, or both")
    if not 0.0 <= float(both_ratio) <= 1.0:
        raise ValueError("both_ratio must lie in [0, 1]")


def _validate_table(table, total_num, path):
    if not torch.is_tensor(table):
        raise RuntimeError("Missing table is not a Tensor: {}".format(path))
    if table.dim() != 1 or table.numel() != int(total_num):
        raise RuntimeError(
            "Missing table length mismatch at {}: expected {}, got {}".format(
                path, int(total_num), int(table.numel())
            )
        )
    values = set(float(value) for value in torch.unique(table).cpu().tolist())
    if not values.issubset({0.0, 1.0, 2.0}):
        raise RuntimeError(
            "Missing table contains values outside {{0, 1, 2}}: {}".format(path)
        )


def _validate_metadata(actual, expected, path):
    if not isinstance(actual, dict):
        raise RuntimeError("Missing table metadata is absent at {}".format(path))
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            matches = isinstance(actual_value, (int, float)) and math.isclose(
                float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = actual_value == expected_value
        if not matches:
            mismatches.append((key, expected_value, actual_value))
    if mismatches:
        raise RuntimeError(
            "Missing table metadata mismatch at {}: {}".format(path, mismatches)
        )


def _legacy_filename(dataset_name, missing_type, missing_ratio):
    ratio = str(missing_ratio).replace(".", "")
    return "{}_missing_{}_{}.pt".format(dataset_name, missing_type, ratio)


def _generate_table(total_num, missing_ratio, missing_type, both_ratio, generator):
    table = torch.zeros(int(total_num), dtype=torch.long)
    missing_count = int(int(total_num) * float(missing_ratio))
    if missing_count == 0:
        return table
    missing_indices = generator.sample(range(int(total_num)), missing_count)
    if missing_type == "text":
        table[missing_indices] = 1
    elif missing_type == "image":
        table[missing_indices] = 2
    else:
        table[missing_indices] = 1
        image_count = int(len(missing_indices) * float(both_ratio))
        image_indices = generator.sample(missing_indices, image_count)
        table[image_indices] = 2
    return table


def load_or_create_missing_table(
    root,
    dataset_name,
    split,
    total_num,
    missing_ratio,
    missing_type,
    both_ratio,
    seed,
    allow_legacy_seed0=True,
):
    """Load a canonical table, explicitly reuse legacy seed 0, or generate one."""

    _validate_configuration(total_num, missing_ratio, missing_type, both_ratio)
    expected = _expected_metadata(
        dataset_name,
        split,
        total_num,
        missing_ratio,
        missing_type,
        both_ratio,
        seed,
    )
    root = os.path.abspath(root)
    canonical_path = os.path.join(root, expected["identity"])
    if os.path.isfile(canonical_path):
        payload = torch.load(canonical_path, map_location="cpu")
        if not isinstance(payload, dict) or "table" not in payload:
            raise RuntimeError(
                "Canonical missing table must contain table and metadata: {}".format(
                    canonical_path
                )
            )
        _validate_metadata(payload.get("metadata"), expected, canonical_path)
        table = payload["table"]
        _validate_table(table, total_num, canonical_path)
        metadata = dict(expected, path=canonical_path, legacy=False)
        return table, metadata

    legacy_path = os.path.join(
        root, _legacy_filename(dataset_name, missing_type, missing_ratio)
    )
    legacy_allowed = (
        bool(allow_legacy_seed0)
        and int(seed) == 0
        and math.isclose(float(both_ratio), 0.5, rel_tol=0.0, abs_tol=1e-12)
    )
    if legacy_allowed and os.path.isfile(legacy_path):
        table = torch.load(legacy_path, map_location="cpu")
        _validate_table(table, total_num, legacy_path)
        metadata = dict(
            expected,
            identity="legacy:" + os.path.basename(legacy_path),
            path=legacy_path,
            legacy=True,
        )
        return table, metadata

    os.makedirs(root, exist_ok=True)
    generator = random.Random(expected["generation_seed"])
    table = _generate_table(
        total_num, missing_ratio, missing_type, both_ratio, generator
    )
    torch.save({"table": table, "metadata": expected}, canonical_path)
    metadata = dict(expected, path=canonical_path, legacy=False)
    return table, metadata
