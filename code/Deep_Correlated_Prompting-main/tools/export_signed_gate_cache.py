"""Export deterministic frozen features and endpoint losses for Gate studies."""

import argparse
import hashlib
import json
import os
import sys

import torchmetrics  # Import first for the legacy transformers/torchmetrics stack.
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clip.datamodules.multitask_datamodule import MTDataModule
from clip.modules import CLIPransformerSS
from clip.modules.reliability_diagnostics import gate_distribution, mmimdb_metrics
from clip.modules.signed_gate_offline import (
    deterministic_sample_indices,
    validate_signed_gate_cache,
)


NEW_PREFIXES = (
    "model.reliability_predictor.",
    "model.prompt_learner.prompt_adapter.",
    "model.utility_gate.",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _verify_frozen_base(final_state, original_state):
    final_base = {
        key: value for key, value in final_state.items()
        if not key.startswith(NEW_PREFIXES)
    }
    if set(final_base) != set(original_state):
        raise RuntimeError("Final checkpoint base keys differ from Original DCP")
    changed = [
        key for key in original_state
        if not torch.equal(final_base[key].cpu(), original_state[key].cpu())
    ]
    if changed:
        raise RuntimeError(
            "Frozen Original DCP tensors changed: {}".format(changed[:10])
        )


def _sample_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    ).mean(dim=-1)


def _dataset_and_metadata(config, split, sample_count, seed, batch_size):
    datamodule = MTDataModule(config, dist=False)
    datamodule.prepare_data()
    datamodule.setup("fit")
    dataset = {
        "train": datamodule.train_dataset,
        "val": datamodule.val_dataset,
    }[split]
    indices = deterministic_sample_indices(
        len(dataset), sample_count=sample_count, seed=seed
    )
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=datamodule.collate,
    )
    provenance_dataset = dataset
    nested = getattr(provenance_dataset, "datasets", None)
    if nested is not None and len(nested) == 1:
        provenance_dataset = nested[0]
    metadata = getattr(provenance_dataset, "missing_table_metadata", None)
    return loader, indices, len(dataset), metadata


def _cache_summary(cache):
    choose_g1 = cache["signed_advantage"].gt(0)
    oracle_logits = torch.where(
        choose_g1.unsqueeze(-1), cache["g1_logits"], cache["g0_logits"]
    )
    return {
        "sample_count": int(cache["sample_id"].numel()),
        "unique_sample_ids": int(torch.unique(cache["sample_id"]).numel()),
        "positive_ratio": float(choose_g1.float().mean()),
        "signed_advantage": gate_distribution(cache["signed_advantage"]),
        "metrics": {
            "g0": mmimdb_metrics(cache["g0_logits"], cache["labels"]),
            "g1": mmimdb_metrics(cache["g1_logits"], cache["labels"]),
            "binary_oracle": mmimdb_metrics(oracle_logits, cache["labels"]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("final_checkpoint")
    parser.add_argument("original_checkpoint")
    parser.add_argument("output_pt")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--missing-table-root")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    final_path = os.path.abspath(args.final_checkpoint)
    original_path = os.path.abspath(args.original_checkpoint)
    output_path = os.path.abspath(args.output_pt)
    final_checkpoint = torch.load(final_path, map_location="cpu")
    original_checkpoint = torch.load(original_path, map_location="cpu")
    _verify_frozen_base(
        final_checkpoint["state_dict"], original_checkpoint["state_dict"]
    )
    config = dict(final_checkpoint["hyper_parameters"]["config"])
    config["test_only"] = True
    config["load_path"] = final_path
    config["num_workers"] = 0
    config.setdefault("missing_table_legacy_seed0", True)
    if args.data_root:
        config["data_root"] = os.path.abspath(args.data_root)
    if args.missing_table_root:
        config["missing_table_root"] = os.path.abspath(args.missing_table_root)

    model_seed = int(config.get("seed", 0))
    pl.seed_everything(model_seed)
    model = CLIPransformerSS(config).cuda().eval()
    if model.model.reliability_importance_mode != "relative":
        raise RuntimeError("signed Gate cache requires relative importance mode")
    loader, selected_indices, dataset_length, missing_metadata = (
        _dataset_and_metadata(
            config,
            args.split,
            sample_count=args.sample_count,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    )

    parts = {
        key: [] for key in (
            "sample_id",
            "missing_type",
            "availability",
            "reliability",
            "base_context",
            "labels",
            "g0_logits",
            "g1_logits",
            "g0_loss",
            "g1_loss",
        )
    }
    with torch.inference_mode(), torch.cuda.amp.autocast():
        for batch_index, batch in enumerate(loader):
            image = batch["image"][0].cuda(non_blocking=True)
            labels = torch.tensor(batch["label"]).float().cuda()
            state = model.model.prepare_reliability_state(
                image, batch["text"], batch["missing_type"]
            )
            g0_logits = model.mmimdb_classifier(state["base_cls_feats"])
            g1_gate = torch.ones(
                (image.shape[0], 1),
                device=image.device,
                dtype=state["reliability"].dtype,
            )
            g1_features = model.model.encode_reliability_state(state, g1_gate)
            g1_logits = model.mmimdb_classifier(g1_features)
            parts["sample_id"].append(
                torch.as_tensor(batch["sample_id"]).long().cpu()
            )
            parts["missing_type"].append(
                torch.as_tensor(batch["missing_type"]).long().cpu()
            )
            parts["availability"].append(
                state["availability_mask"].float().cpu()
            )
            parts["reliability"].append(state["reliability"].float().cpu())
            parts["base_context"].append(state["base_cls_feats"].cpu())
            parts["labels"].append(labels.cpu())
            parts["g0_logits"].append(g0_logits.float().cpu())
            parts["g1_logits"].append(g1_logits.float().cpu())
            parts["g0_loss"].append(_sample_loss(g0_logits, labels).float().cpu())
            parts["g1_loss"].append(_sample_loss(g1_logits, labels).float().cpu())
            if (batch_index + 1) % 100 == 0:
                print(
                    "CACHE_PROGRESS split={} batches={} samples={}".format(
                        args.split,
                        batch_index + 1,
                        sum(value.shape[0] for value in parts["sample_id"]),
                    ),
                    flush=True,
                )

    cache = {key: torch.cat(value) for key, value in parts.items()}
    cache["signed_advantage"] = cache["g0_loss"] - cache["g1_loss"]
    cache["dataset_index"] = selected_indices.clone()
    cache["metadata"] = {
        "format_version": 1,
        "split": args.split,
        "dataset_length": int(dataset_length),
        "sample_count_requested": int(args.sample_count),
        "selection_seed": int(args.seed),
        "model_seed": model_seed,
        "final_checkpoint": final_path,
        "final_checkpoint_sha256": _sha256(final_path),
        "original_checkpoint": original_path,
        "original_checkpoint_sha256": _sha256(original_path),
        "importance_mode": model.model.reliability_importance_mode,
        "missing_table": missing_metadata,
        "exporter_sha256": _sha256(os.path.abspath(__file__)),
        "offline_helpers_sha256": _sha256(
            os.path.join(
                PROJECT_ROOT, "clip", "modules", "signed_gate_offline.py"
            )
        ),
    }
    validate_signed_gate_cache(cache)
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    torch.save(cache, output_path)
    summary = _cache_summary(cache)
    summary["cache"] = {
        "path": output_path,
        "sha256": _sha256(output_path),
        "metadata": cache["metadata"],
    }
    summary_path = output_path + ".json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
