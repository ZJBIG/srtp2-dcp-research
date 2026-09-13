"""Evaluate final Reliability-DCP ablations on one deterministic MM-IMDb loader."""

import argparse
import json
import os
import sys

import torchmetrics  # Import first for the legacy transformers/torchmetrics stack.
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clip.datamodules.multitask_datamodule import MTDataModule
from clip.modules import CLIPransformerSS
from clip.modules.reliability_diagnostics import (
    binary_gate_score_diagnostics,
    condition_preserving_shuffle,
    condition_prior_reliability,
    continuous_target_metrics,
    gate_distribution,
    gate_temperature_sweep,
    importance_diagnostics,
    mmimdb_metrics,
    relative_importance_metrics,
)
from clip.modules.reliability_learning import (
    PromptAdapter,
    endpoint_gate_target,
    reliability_targets_from_delta,
    soft_gate_target,
)


NEW_PREFIXES = (
    "model.reliability_predictor.",
    "model.prompt_learner.prompt_adapter.",
    "model.utility_gate.",
)


def verify_frozen_base(final_state, original_state):
    final_base = {
        key: value for key, value in final_state.items()
        if not key.startswith(NEW_PREFIXES)
    }
    if set(final_base) != set(original_state):
        raise RuntimeError("Final checkpoint base keys differ from Original DCP")
    differences = [
        key for key in original_state
        if not torch.equal(final_base[key].cpu(), original_state[key].cpu())
    ]
    if differences:
        raise RuntimeError(
            "Frozen Original DCP tensors changed: {}".format(differences[:10])
        )


def loader_for(config, split):
    dm = MTDataModule(config, dist=False)
    dm.prepare_data()
    dm.setup("fit")
    dataset = {"val": dm.val_dataset, "test": dm.test_dataset}[split]
    return DataLoader(
        dataset,
        batch_size=dm.batch_size,
        shuffle=False,
        num_workers=dm.num_workers,
        collate_fn=dm.collate,
    )


def _sample_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    ).mean(dim=-1)


def _select_text(text, indices):
    return ([text[0][int(index)] for index in indices],)


def _prompt_max_abs(first, second):
    if len(first) != len(second):
        raise RuntimeError("Prompt depth changed between equivalent paths")
    return max(
        (
            float((left - right).abs().float().max())
            for left, right in zip(first, second)
        ),
        default=0.0,
    )


def _state_with_reliability(clip_model, state, reliability):
    image_offsets, text_offsets, _ = (
        clip_model.prompt_learner.prompt_adapter.compute_offsets(
            state["image_prompts"],
            state["text_prompts"],
            reliability,
            state["availability_mask"],
        )
    )
    adjusted = dict(state)
    adjusted.update(
        {
            "reliability": reliability,
            "image_offsets": image_offsets,
            "text_offsets": text_offsets,
        }
    )
    return adjusted


def _difference_accumulator():
    return {"max_abs": 0.0, "sum_abs": 0.0, "count": 0}


def _accumulate_difference(accumulator, first, second):
    if isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise RuntimeError("Compared prompt depths do not match")
        for left, right in zip(first, second):
            _accumulate_difference(accumulator, left, right)
        return
    difference = (first - second).detach().abs().float()
    accumulator["max_abs"] = max(
        accumulator["max_abs"], float(difference.max())
    )
    accumulator["sum_abs"] += float(difference.sum())
    accumulator["count"] += difference.numel()


def _finalize_difference(accumulator):
    return {
        "count": int(accumulator["count"]),
        "max_abs": float(accumulator["max_abs"]),
        "mean_abs": float(
            accumulator["sum_abs"] / max(accumulator["count"], 1)
        ),
    }


def _candidate_ablation_metrics(fixed_logits, labels, candidates):
    candidate_labels = tuple(str(value) for value in candidates)
    fixed_metrics = {
        label: mmimdb_metrics(fixed_logits[label], labels)
        for label in candidate_labels
    }
    best_f1_label = max(
        candidate_labels, key=lambda label: fixed_metrics[label]["macro_f1"]
    )
    best_loss_label = min(
        candidate_labels, key=lambda label: fixed_metrics[label]["loss"]
    )
    fixed_stack = torch.stack(
        [fixed_logits[label] for label in candidate_labels], dim=1
    )
    expanded_labels = labels.unsqueeze(1).expand_as(fixed_stack)
    candidate_losses = F.binary_cross_entropy_with_logits(
        fixed_stack, expanded_labels, reduction="none"
    ).mean(dim=-1)
    hard_best_index = candidate_losses.argmin(dim=-1)
    row_index = torch.arange(labels.shape[0])
    multistrength_oracle_logits = fixed_stack[row_index, hard_best_index]
    binary_losses = candidate_losses[:, [0, len(candidates) - 1]]
    binary_best_index = binary_losses.argmin(dim=-1)
    binary_stack = fixed_stack[:, [0, len(candidates) - 1], :]
    binary_oracle_logits = binary_stack[row_index, binary_best_index]
    return (
        {
            "fixed": fixed_metrics,
            "best_global_fixed": dict(
                {"gate": float(best_f1_label)},
                **fixed_metrics[best_f1_label]
            ),
            "best_global_fixed_loss": dict(
                {"gate": float(best_loss_label)},
                **fixed_metrics[best_loss_label]
            ),
            "binary_oracle": mmimdb_metrics(binary_oracle_logits, labels),
            "multistrength_oracle": mmimdb_metrics(
                multistrength_oracle_logits, labels
            ),
        },
        candidate_losses,
        hard_best_index,
    )


def _hard_gate_summary(
    gate_values,
    g0_logits,
    g1_logits,
    labels,
    candidate_losses,
    binary_target,
    endpoint_regret,
):
    gate_values = gate_values.detach().float().reshape(-1)
    decision = gate_values.ge(0.5)
    selected_logits = torch.where(
        decision.unsqueeze(-1), g1_logits, g0_logits
    )
    selected_loss = torch.where(
        decision, candidate_losses[:, -1], candidate_losses[:, 0]
    )
    best_endpoint_loss = torch.minimum(
        candidate_losses[:, 0], candidate_losses[:, -1]
    )
    target_decision = binary_target.ge(0.5)
    correct = decision.eq(target_decision).float()
    weighted_accuracy = (
        float((correct * endpoint_regret).sum() / endpoint_regret.sum())
        if float(endpoint_regret.sum()) > 0.0 else None
    )
    return {
        "gate": gate_distribution(gate_values),
        "hard_on_ratio": float(decision.float().mean()),
        "binary_accuracy": float(correct.mean()),
        "regret_weighted_accuracy": weighted_accuracy,
        "mean_endpoint_regret": float(
            (selected_loss - best_endpoint_loss).mean()
        ),
        "metrics": mmimdb_metrics(selected_logits, labels),
    }


def _importance_targets(model, batch, image, state, labels, temperature):
    missing = torch.as_tensor(batch["missing_type"], device=image.device).long()
    current_loss = _sample_loss(
        model.mmimdb_classifier(state["base_cls_feats"]), labels
    )
    null_loss = _sample_loss(
        model.mmimdb_classifier(torch.zeros_like(state["base_cls_feats"])),
        labels,
    )
    loss_without_image = null_loss.clone()
    loss_without_text = null_loss.clone()
    complete = missing.eq(0).nonzero(as_tuple=False).flatten()
    if complete.numel():
        indices = complete.detach().cpu().tolist()
        selected_image = image.index_select(0, complete)
        selected_text = _select_text(batch["text"], indices)
        selected_labels = labels.index_select(0, complete)
        text_only = model.model.base_view(
            torch.ones_like(selected_image), selected_text, [2] * len(indices)
        )
        image_only = model.model.base_view(
            selected_image, ([""] * len(indices),), [1] * len(indices)
        )
        loss_without_image.index_copy_(
            0,
            complete,
            _sample_loss(
                model.mmimdb_classifier(text_only["cls_feats"]), selected_labels
            ),
        )
        loss_without_text.index_copy_(
            0,
            complete,
            _sample_loss(
                model.mmimdb_classifier(image_only["cls_feats"]), selected_labels
            ),
        )
    delta = torch.stack(
        [loss_without_image - current_loss, loss_without_text - current_loss],
        dim=-1,
    )
    target, supervision = reliability_targets_from_delta(
        delta,
        state["availability_mask"],
        temperature,
        importance_mode=model.model.reliability_importance_mode,
    )
    unsupervised = supervision.le(0.0)
    delta = delta.masked_fill(unsupervised, float("nan"))
    target = target.masked_fill(unsupervised, float("nan"))
    return delta, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("final_checkpoint")
    parser.add_argument("original_checkpoint")
    parser.add_argument("output_json")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--data-root")
    parser.add_argument("--missing-table-root")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    final_path = os.path.abspath(args.final_checkpoint)
    original_path = os.path.abspath(args.original_checkpoint)
    final_checkpoint = torch.load(final_path, map_location="cpu")
    original_checkpoint = torch.load(original_path, map_location="cpu")
    verify_frozen_base(
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
    loader = loader_for(config, args.split)
    provenance_dataset = loader.dataset
    nested_datasets = getattr(provenance_dataset, "datasets", None)
    if nested_datasets is not None and len(nested_datasets) == 1:
        provenance_dataset = nested_datasets[0]
    missing_table_metadata = getattr(
        provenance_dataset, "missing_table_metadata", None
    )
    candidates = tuple(float(value) for value in model.model.gate_candidates)
    candidate_labels = tuple(str(value) for value in candidates)
    original_logits, learned_logits = [], []
    fixed_logits = {label: [] for label in candidate_labels}
    labels_all, gates_all, sample_ids = [], [], []
    reliability_all, target_all, delta_all = [], [], []
    availability_all, missing_type_all = [], []
    g0_feature_max = 0.0
    g0_feature_sum = 0.0
    g0_feature_count = 0
    g0_logit_max = 0.0
    g0_logit_sum = 0.0
    g0_logit_count = 0
    zero_gate_prompt_max = 0.0
    regenerated_prompt_max = 0.0
    raw_zero_base_max = 0.0
    raw_zero_base_sum = 0.0
    raw_zero_base_count = 0
    same_prompt_repeat_max = 0.0
    same_prompt_repeat_sum = 0.0
    same_prompt_repeat_count = 0
    task_temperatures = config.get("reliability_temperature_by_task", {})
    importance_temperature = task_temperatures.get("mmimdb")
    if importance_temperature is None:
        importance_temperature = config.get("reliability_temperature", 0.1)
    importance_temperature = float(importance_temperature)

    with torch.inference_mode(), torch.cuda.amp.autocast():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            image = batch["image"][0].cuda(non_blocking=True)
            labels = torch.tensor(batch["label"]).float().cuda()
            state = model.model.prepare_reliability_state(
                image, batch["text"], batch["missing_type"]
            )
            learned_gate = model.model.utility_gate(
                state["reliability"],
                state["availability_mask"],
                state["base_cls_feats"],
            )
            zero_gate = torch.zeros_like(learned_gate)
            zero_image_prompts = PromptAdapter._apply_gate(
                state["image_prompts"], state["image_offsets"], zero_gate
            )
            zero_text_prompts = PromptAdapter._apply_gate(
                state["text_prompts"], state["text_offsets"], zero_gate
            )
            zero_gate_prompt_max = max(
                zero_gate_prompt_max,
                _prompt_max_abs(zero_image_prompts, state["image_prompts"]),
                _prompt_max_abs(zero_text_prompts, state["text_prompts"]),
            )
            regenerated_image_prompts, regenerated_text_prompts = (
                model.model.prompt_learner(batch["missing_type"])
            )
            regenerated_prompt_max = max(
                regenerated_prompt_max,
                _prompt_max_abs(
                    regenerated_image_prompts, state["image_prompts"]
                ),
                _prompt_max_abs(
                    regenerated_text_prompts, state["text_prompts"]
                ),
            )
            raw_image, raw_text = model.model._encode_from_prompts(
                state["image"],
                state["tokenized_texts"],
                state["missing_type"],
                zero_image_prompts,
                zero_text_prompts,
            )
            raw_image, raw_text = model.model._mask_features(
                raw_image, raw_text, state["availability_mask"]
            )
            raw_zero_features = torch.cat([raw_image, raw_text], dim=-1)
            repeated_image, repeated_text = model.model._encode_from_prompts(
                state["image"],
                state["tokenized_texts"],
                state["missing_type"],
                zero_image_prompts,
                zero_text_prompts,
            )
            repeated_image, repeated_text = model.model._mask_features(
                repeated_image, repeated_text, state["availability_mask"]
            )
            repeated_features = torch.cat(
                [repeated_image, repeated_text], dim=-1
            )
            raw_zero_difference = (
                raw_zero_features - state["base_cls_feats"]
            ).abs().float()
            repeat_difference = (
                repeated_features - raw_zero_features
            ).abs().float()
            raw_zero_base_max = max(
                raw_zero_base_max, float(raw_zero_difference.max())
            )
            raw_zero_base_sum += float(raw_zero_difference.sum())
            raw_zero_base_count += raw_zero_difference.numel()
            same_prompt_repeat_max = max(
                same_prompt_repeat_max, float(repeat_difference.max())
            )
            same_prompt_repeat_sum += float(repeat_difference.sum())
            same_prompt_repeat_count += repeat_difference.numel()
            base_logits = model.mmimdb_classifier(state["base_cls_feats"])
            original_logits.append(base_logits.float().cpu())
            for candidate, label in zip(candidates, candidate_labels):
                gate = torch.full_like(learned_gate, candidate)
                candidate_features = model.model.encode_reliability_state(
                    state, gate
                )
                candidate_logits = model.mmimdb_classifier(candidate_features)
                fixed_logits[label].append(candidate_logits.float().cpu())
                if candidate == 0.0:
                    feature_difference = (
                        candidate_features - state["base_cls_feats"]
                    ).abs().float()
                    logit_difference = (
                        candidate_logits - base_logits
                    ).abs().float()
                    g0_feature_max = max(
                        g0_feature_max, float(feature_difference.max())
                    )
                    g0_feature_sum += float(feature_difference.sum())
                    g0_feature_count += feature_difference.numel()
                    g0_logit_max = max(g0_logit_max, float(logit_difference.max()))
                    g0_logit_sum += float(logit_difference.sum())
                    g0_logit_count += logit_difference.numel()
            learned_logits.append(
                model.mmimdb_classifier(
                    model.model.encode_reliability_state(state, learned_gate)
                ).float().cpu()
            )
            delta, importance_target = _importance_targets(
                model,
                batch,
                image,
                state,
                labels,
                importance_temperature,
            )
            labels_all.append(labels.cpu())
            gates_all.append(learned_gate.float().cpu())
            reliability_all.append(state["reliability"].float().cpu())
            target_all.append(importance_target.float().cpu())
            delta_all.append(delta.float().cpu())
            availability_all.append(state["availability_mask"].float().cpu())
            missing_type_all.append(
                torch.as_tensor(batch["missing_type"]).long().cpu()
            )
            sample_ids.extend(int(value) for value in batch["sample_id"])

    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Ablation loader contains duplicate sample_id values")
    labels_all = torch.cat(labels_all)
    gates_all = torch.cat(gates_all)
    reliability_all = torch.cat(reliability_all)
    target_all = torch.cat(target_all)
    delta_all = torch.cat(delta_all)
    availability_all = torch.cat(availability_all)
    missing_type_all = torch.cat(missing_type_all)
    original_logits = torch.cat(original_logits)
    learned_logits = torch.cat(learned_logits)
    fixed_logits = {
        label: torch.cat(values) for label, values in fixed_logits.items()
    }
    predicted_candidate_metrics, candidate_losses, hard_best_index = (
        _candidate_ablation_metrics(fixed_logits, labels_all, candidates)
    )
    binary_gate_target, endpoint_regret = endpoint_gate_target(
        candidate_losses, candidates
    )
    soft_target, _, candidate_loss_range = soft_gate_target(
        candidate_losses,
        candidates,
        config.get("gate_target_temperature", 0.025),
    )
    temperature_sweep = gate_temperature_sweep(
        candidate_losses,
        candidates,
        temperatures=(0.005, 0.01, 0.025, 0.05),
    )

    source_reliability = {
        "predicted": reliability_all,
        "neutral": torch.full_like(reliability_all, 0.5),
        "condition_prior": condition_prior_reliability(
            reliability_all, availability_all
        ),
        "condition_shuffled": condition_preserving_shuffle(
            reliability_all, availability_all, seed=args.seed
        ),
        "oracle": torch.nan_to_num(target_all, nan=0.5),
    }
    available_entries = availability_all.gt(0.5)
    for source_name, source_values in tuple(source_reliability.items()):
        source_values = torch.where(
            available_entries,
            source_values,
            torch.full_like(source_values, 0.5),
        )
        if not torch.isfinite(source_values).all():
            raise RuntimeError(
                "Non-finite available reliability in source {}".format(
                    source_name
                )
            )
        source_reliability[source_name] = source_values

    generator = torch.Generator().manual_seed(args.seed)
    shuffled_gates = gates_all[torch.randperm(len(gates_all), generator=generator)]
    additional_sources = tuple(
        name for name in source_reliability if name != "predicted"
    )
    source_fixed_parts = {
        source_name: {
            label: []
            for candidate, label in zip(candidates, candidate_labels)
            if candidate != 0.0
        }
        for source_name in additional_sources
    }
    source_difference_accumulators = {
        source_name: {
            "offset": _difference_accumulator(),
            "g1_feature": _difference_accumulator(),
            "g1_logit": _difference_accumulator(),
        }
        for source_name in additional_sources
    }
    gate_source_parts = {source_name: [] for source_name in additional_sources}
    g1_repeat_accumulator = {
        "feature": _difference_accumulator(),
        "logit": _difference_accumulator(),
    }
    shuffled_logits, repeated_ids, offset = [], [], 0
    with torch.inference_mode(), torch.cuda.amp.autocast():
        for batch_index, batch in enumerate(loader_for(config, args.split)):
            if args.max_batches and batch_index >= args.max_batches:
                break
            image = batch["image"][0].cuda(non_blocking=True)
            batch_size = image.shape[0]
            state = model.model.prepare_reliability_state(
                image, batch["text"], batch["missing_type"]
            )
            predicted_g1_gate = torch.ones(
                (batch_size, 1),
                device=image.device,
                dtype=state["reliability"].dtype,
            )
            predicted_g1_features = model.model.encode_reliability_state(
                state, predicted_g1_gate
            )
            predicted_g1_logits = model.mmimdb_classifier(
                predicted_g1_features
            )
            repeated_g1_features = model.model.encode_reliability_state(
                state, predicted_g1_gate
            )
            repeated_g1_logits = model.mmimdb_classifier(
                repeated_g1_features
            )
            _accumulate_difference(
                g1_repeat_accumulator["feature"],
                repeated_g1_features,
                predicted_g1_features,
            )
            _accumulate_difference(
                g1_repeat_accumulator["logit"],
                repeated_g1_logits,
                predicted_g1_logits,
            )
            for source_name in additional_sources:
                source_values = source_reliability[source_name][
                    offset:offset + batch_size
                ].to(state["reliability"])
                source_state = _state_with_reliability(
                    model.model, state, source_values
                )
                gate_source_parts[source_name].append(
                    model.model.utility_gate(
                        source_values,
                        state["availability_mask"],
                        state["base_cls_feats"],
                    ).float().cpu()
                )
                _accumulate_difference(
                    source_difference_accumulators[source_name]["offset"],
                    source_state["image_offsets"] + source_state["text_offsets"],
                    state["image_offsets"] + state["text_offsets"],
                )
                for candidate, label in zip(candidates, candidate_labels):
                    if candidate == 0.0:
                        continue
                    candidate_gate = torch.full(
                        (batch_size, 1),
                        candidate,
                        device=image.device,
                        dtype=state["reliability"].dtype,
                    )
                    source_features = model.model.encode_reliability_state(
                        source_state, candidate_gate
                    )
                    source_logits = model.mmimdb_classifier(source_features)
                    source_fixed_parts[source_name][label].append(
                        source_logits.float().cpu()
                    )
                    if candidate == 1.0:
                        _accumulate_difference(
                            source_difference_accumulators[source_name][
                                "g1_feature"
                            ],
                            source_features,
                            predicted_g1_features,
                        )
                        _accumulate_difference(
                            source_difference_accumulators[source_name][
                                "g1_logit"
                            ],
                            source_logits,
                            predicted_g1_logits,
                        )
            gate = shuffled_gates[offset:offset + batch_size].cuda()
            shuffled_logits.append(
                model.mmimdb_classifier(
                    model.model.encode_reliability_state(state, gate)
                ).float().cpu()
            )
            repeated_ids.extend(int(value) for value in batch["sample_id"])
            offset += batch_size
    if repeated_ids != sample_ids:
        raise RuntimeError("Ablation loader order changed between deterministic passes")

    source_fixed_logits = {"predicted": fixed_logits}
    zero_label = candidate_labels[0]
    for source_name in additional_sources:
        source_fixed_logits[source_name] = {zero_label: fixed_logits[zero_label]}
        for candidate, label in zip(candidates, candidate_labels):
            if candidate != 0.0:
                source_fixed_logits[source_name][label] = torch.cat(
                    source_fixed_parts[source_name][label]
                )

    gate_source_values = {"predicted": gates_all}
    for source_name in additional_sources:
        gate_source_values[source_name] = torch.cat(
            gate_source_parts[source_name]
        )
    gate_source_ablation = {
        source_name: _hard_gate_summary(
            gate_values,
            fixed_logits[candidate_labels[0]],
            fixed_logits[candidate_labels[-1]],
            labels_all,
            candidate_losses,
            binary_gate_target,
            endpoint_regret,
        )
        for source_name, gate_values in gate_source_values.items()
    }

    source_ablation = {}
    source_output_differences = {}
    for source_name, source_values in source_reliability.items():
        source_candidate_metrics, _, _ = _candidate_ablation_metrics(
            source_fixed_logits[source_name], labels_all, candidates
        )
        source_ablation[source_name] = {
            "metrics": source_candidate_metrics,
            "relative_target_probability": relative_importance_metrics(
                source_values, target_all, availability_all
            ),
            "relative_raw_delta": relative_importance_metrics(
                source_values, delta_all, availability_all
            ),
        }
        reliability_difference = (
            source_values[available_entries]
            - source_reliability["predicted"][available_entries]
        ).abs()
        if source_name == "predicted":
            source_output_differences[source_name] = {
                "available_reliability_abs_difference": gate_distribution(
                    reliability_difference
                ),
                "offset": {"count": 0, "max_abs": 0.0, "mean_abs": 0.0},
                "g1_feature": {
                    "count": 0, "max_abs": 0.0, "mean_abs": 0.0
                },
                "g1_logit": {
                    "count": 0, "max_abs": 0.0, "mean_abs": 0.0
                },
            }
        else:
            source_output_differences[source_name] = {
                "available_reliability_abs_difference": gate_distribution(
                    reliability_difference
                ),
                **{
                    key: _finalize_difference(value)
                    for key, value in source_difference_accumulators[
                        source_name
                    ].items()
                },
            }

    metrics = {
        "original_dcp": mmimdb_metrics(original_logits, labels_all),
        **predicted_candidate_metrics,
        "learned": mmimdb_metrics(learned_logits, labels_all),
        "shuffled": mmimdb_metrics(torch.cat(shuffled_logits), labels_all),
        "learned_hard": _hard_gate_summary(
            gates_all,
            fixed_logits[candidate_labels[0]],
            fixed_logits[candidate_labels[-1]],
            labels_all,
            candidate_losses,
            binary_gate_target,
            endpoint_regret,
        )["metrics"],
        "shuffled_hard": _hard_gate_summary(
            shuffled_gates,
            fixed_logits[candidate_labels[0]],
            fixed_logits[candidate_labels[-1]],
            labels_all,
            candidate_losses,
            binary_gate_target,
            endpoint_regret,
        )["metrics"],
    }
    importance = importance_diagnostics(
        reliability_all,
        target_all,
        availability_all,
        missing_type_all,
    )
    delta_correlation = {}
    for index, name in enumerate(("image", "text")):
        supervised = availability_all[:, index].gt(0.5) & torch.isfinite(
            delta_all[:, index]
        )
        correlation = continuous_target_metrics(
            reliability_all[supervised, index], delta_all[supervised, index]
        )
        delta_correlation[name] = {
            "count": correlation["count"],
            "pearson": correlation["pearson"],
            "spearman": correlation["spearman"],
        }
    hard_distribution = {
        label: float(hard_best_index.eq(index).float().mean())
        for index, label in enumerate(candidate_labels)
    }
    gate_score_diagnostics = {
        "learned": binary_gate_score_diagnostics(
            gates_all,
            fixed_logits[candidate_labels[0]],
            fixed_logits[candidate_labels[-1]],
            labels_all,
            candidate_losses,
            binary_gate_target,
            endpoint_regret,
        ),
        "shuffled": binary_gate_score_diagnostics(
            shuffled_gates,
            fixed_logits[candidate_labels[0]],
            fixed_logits[candidate_labels[-1]],
            labels_all,
            candidate_losses,
            binary_gate_target,
            endpoint_regret,
        ),
    }
    result = {
        "split": args.split,
        "sample_count": len(sample_ids),
        "unique_sample_ids": len(set(sample_ids)),
        "base_tensors_equal_original": True,
        "checkpoint": {
            "final": final_path,
            "original_dcp": original_path,
            "model_seed": model_seed,
        },
        "missing_table": missing_table_metadata,
        "metrics": metrics,
        "g0_equivalence": {
            "feature_max_abs": g0_feature_max,
            "feature_mean_abs": g0_feature_sum / max(g0_feature_count, 1),
            "logit_max_abs": g0_logit_max,
            "logit_mean_abs": g0_logit_sum / max(g0_logit_count, 1),
        },
        "numerical_path_diagnostics": {
            "zero_gate_prompt_max_abs": zero_gate_prompt_max,
            "regenerated_prompt_max_abs": regenerated_prompt_max,
            "raw_zero_vs_base_feature_max_abs": raw_zero_base_max,
            "raw_zero_vs_base_feature_mean_abs": (
                raw_zero_base_sum / max(raw_zero_base_count, 1)
            ),
            "same_prompt_repeat_feature_max_abs": same_prompt_repeat_max,
            "same_prompt_repeat_feature_mean_abs": (
                same_prompt_repeat_sum / max(same_prompt_repeat_count, 1)
            ),
        },
        "learned_gate": gate_distribution(gates_all),
        "soft_oracle_target": gate_distribution(soft_target),
        "gate_temperature_sweep": temperature_sweep,
        "candidate_loss_range": gate_distribution(candidate_loss_range),
        "hard_g_star_distribution": hard_distribution,
        "importance_temperature": importance_temperature,
        "importance_mode": model.model.reliability_importance_mode,
        "gate_supervision_mode": config.get(
            "gate_supervision_mode", "soft_scalar"
        ),
        "importance": importance,
        "reliability_delta_correlation": delta_correlation,
        "importance_source_ablation": source_ablation,
        "importance_source_output_differences": source_output_differences,
        "gate_importance_source_ablation": gate_source_ablation,
        "gate_score_diagnostics": gate_score_diagnostics,
        "binary_gate_target": {
            "positive_ratio": float(binary_gate_target.gt(0.5).float().mean()),
            "tie_ratio": float(binary_gate_target.eq(0.5).float().mean()),
            "endpoint_regret": gate_distribution(endpoint_regret),
        },
        "g1_repeat_numerical_baseline": {
            key: _finalize_difference(value)
            for key, value in g1_repeat_accumulator.items()
        },
        "shuffle_seed": args.seed,
    }
    output_path = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
