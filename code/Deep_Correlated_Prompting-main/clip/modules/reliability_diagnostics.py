"""Final-model MM-IMDb ablation metric helpers."""

import torch
import torch.nn.functional as F


def _safe_f1(true_positive, predicted_positive, actual_positive):
    denominator = predicted_positive + actual_positive
    return torch.where(
        denominator > 0,
        2.0 * true_positive / denominator.clamp_min(1.0),
        torch.zeros_like(denominator),
    )


def mmimdb_metrics(logits, labels):
    """Compute the repository's zero-threshold multilabel metrics."""

    predictions = logits.gt(0)
    targets = labels.bool()
    true_positive = (predictions & targets).float()
    class_f1 = _safe_f1(
        true_positive.sum(0), predictions.float().sum(0), targets.float().sum(0)
    )
    sample_f1 = _safe_f1(
        true_positive.sum(1), predictions.float().sum(1), targets.float().sum(1)
    )
    micro_f1 = _safe_f1(
        true_positive.sum(), predictions.float().sum(), targets.float().sum()
    )
    return {
        "loss": float(F.binary_cross_entropy_with_logits(logits, labels)),
        "micro_f1": float(micro_f1),
        "macro_f1": float(class_f1.mean()),
        "samples_f1": float(sample_f1.mean()),
    }


def gate_distribution(values):
    values = values.detach().float().reshape(-1)
    quantiles = torch.quantile(values, torch.tensor([.1, .25, .5, .75, .9]))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
    }


def _value_distribution(values):
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"mean": None, "std": None, "p10": None, "p50": None, "p90": None}
    quantiles = torch.quantile(values, torch.tensor([.1, .5, .9]))
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p10": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p90": float(quantiles[2]),
    }


def _pearson(first, second):
    first = first.detach().float().reshape(-1)
    second = second.detach().float().reshape(-1)
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    if float(denominator) == 0.0:
        return None
    return float((first * second).sum() / denominator)


def _average_ranks(values):
    values = values.detach().float().reshape(-1)
    sorted_values, order = torch.sort(values)
    sorted_ranks = torch.empty_like(sorted_values)
    start = 0
    while start < sorted_values.numel():
        end = start + 1
        while end < sorted_values.numel() and bool(
            sorted_values[end] == sorted_values[start]
        ):
            end += 1
        sorted_ranks[start:end] = (start + end - 1) / 2.0
        start = end
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def binary_gate_score_diagnostics(
    gate_values,
    g0_logits,
    g1_logits,
    labels,
    candidate_losses,
    binary_target,
    endpoint_regret,
):
    """Diagnose endpoint ranking and same-split threshold headroom."""

    scores = gate_values.detach().float().reshape(-1)
    g0_logits = g0_logits.detach().float()
    g1_logits = g1_logits.detach().float()
    labels = labels.detach().float()
    candidate_losses = candidate_losses.detach().float()
    binary_target = binary_target.detach().float().reshape(-1)
    endpoint_regret = endpoint_regret.detach().float().reshape(-1)
    sample_count = scores.numel()
    if sample_count == 0:
        raise ValueError("binary Gate diagnostics require at least one sample")
    if g0_logits.shape != g1_logits.shape or g0_logits.shape != labels.shape:
        raise ValueError("g0_logits, g1_logits, and labels must share shape")
    if g0_logits.shape[0] != sample_count:
        raise ValueError("Gate scores and logits must share sample count")
    if candidate_losses.dim() != 2 or candidate_losses.shape[0] != sample_count:
        raise ValueError("candidate_losses must have shape [N, K]")
    if candidate_losses.shape[1] < 2:
        raise ValueError("candidate_losses must include g0 and g1 endpoints")
    if binary_target.numel() != sample_count or endpoint_regret.numel() != sample_count:
        raise ValueError("Gate targets and regret must share sample count")

    strict = binary_target.ne(0.5)
    positive = binary_target.gt(0.5) & strict
    negative = binary_target.lt(0.5) & strict
    if positive.any() and negative.any():
        ranks = _average_ranks(scores[strict])
        strict_positive = binary_target[strict].gt(0.5)
        positive_count = int(strict_positive.sum())
        negative_count = int((~strict_positive).sum())
        positive_rank_sum = ranks[strict_positive].sum()
        roc_auc = float(
            (
                positive_rank_sum
                - positive_count * (positive_count - 1) / 2.0
            )
            / (positive_count * negative_count)
        )
    else:
        roc_auc = None

    endpoint_advantage = candidate_losses[:, 0] - candidate_losses[:, -1]
    unique_scores = torch.unique(scores, sorted=True)
    scale = max(float(unique_scores.abs().max()), 1.0)
    epsilon = scale * 1e-6
    if unique_scores.numel() == 1:
        thresholds = torch.stack(
            [unique_scores[0] - epsilon, unique_scores[0] + epsilon]
        )
    else:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        thresholds = torch.cat(
            [
                unique_scores[:1] - epsilon,
                midpoints,
                unique_scores[-1:] + epsilon,
            ]
        )

    best_endpoint_loss = torch.minimum(
        candidate_losses[:, 0], candidate_losses[:, -1]
    )
    target_decision = binary_target.ge(0.5)
    regret_sum = endpoint_regret.sum()

    def threshold_summary(threshold):
        decision = scores.ge(threshold)
        selected_logits = torch.where(
            decision.unsqueeze(-1), g1_logits, g0_logits
        )
        selected_loss = torch.where(
            decision, candidate_losses[:, -1], candidate_losses[:, 0]
        )
        correct = decision.eq(target_decision).float()
        return {
            "threshold": float(threshold),
            "hard_on_ratio": float(decision.float().mean()),
            "binary_accuracy": float(correct.mean()),
            "regret_weighted_accuracy": (
                float((correct * endpoint_regret).sum() / regret_sum)
                if float(regret_sum) > 0.0 else None
            ),
            "mean_endpoint_regret": float(
                (selected_loss - best_endpoint_loss).mean()
            ),
            "metrics": mmimdb_metrics(selected_logits, labels),
        }

    summaries = [threshold_summary(value) for value in thresholds]
    best_task_loss = min(
        summaries,
        key=lambda item: (
            item["metrics"]["loss"], -item["metrics"]["macro_f1"]
        ),
    )
    best_macro_f1 = max(
        summaries,
        key=lambda item: (
            item["metrics"]["macro_f1"], -item["metrics"]["loss"]
        ),
    )
    best_weighted_accuracy = max(
        summaries,
        key=lambda item: (
            -1.0 if item["regret_weighted_accuracy"] is None
            else item["regret_weighted_accuracy"],
            -item["mean_endpoint_regret"],
        ),
    )
    return {
        "score": gate_distribution(scores),
        "target_positive_ratio": float(positive.float().mean()),
        "regret_weighted_positive_ratio": (
            float((positive.float() * endpoint_regret).sum() / regret_sum)
            if float(regret_sum) > 0.0 else None
        ),
        "roc_auc": roc_auc,
        "endpoint_advantage_pearson": _pearson(scores, endpoint_advantage),
        "endpoint_advantage_spearman": _pearson(
            _average_ranks(scores), _average_ranks(endpoint_advantage)
        ),
        "threshold_sweep": {
            "candidate_count": int(thresholds.numel()),
            "best_task_loss": best_task_loss,
            "best_macro_f1": best_macro_f1,
            "best_regret_weighted_accuracy": best_weighted_accuracy,
        },
    }


def _validate_importance_pair(reliability, availability):
    reliability = reliability.detach().float()
    availability = availability.detach().to(
        device=reliability.device, dtype=torch.float32
    )
    if reliability.shape != availability.shape:
        raise ValueError("reliability and availability must share shape")
    if reliability.dim() != 2 or reliability.shape[1] != 2:
        raise ValueError("importance tensors must have shape [N, 2]")
    return reliability, availability


def _availability_group_ids(availability):
    available = availability.gt(0.5).long()
    return available[:, 0] * 2 + available[:, 1]


def condition_prior_reliability(reliability, availability):
    """Replace each R pair by its availability-condition prediction mean."""

    reliability, availability = _validate_importance_pair(
        reliability, availability
    )
    result = reliability.clone()
    group_ids = _availability_group_ids(availability)
    for group_id in torch.unique(group_ids, sorted=True).tolist():
        selected = group_ids.eq(int(group_id))
        result[selected] = reliability[selected].mean(dim=0, keepdim=True)
    return result


def condition_preserving_shuffle(reliability, availability, seed=0):
    """Shuffle complete R pairs only within the same availability condition."""

    reliability, availability = _validate_importance_pair(
        reliability, availability
    )
    result = reliability.clone()
    group_ids = _availability_group_ids(availability)
    generator = torch.Generator().manual_seed(int(seed))
    for group_id in torch.unique(group_ids, sorted=True).tolist():
        indices = group_ids.eq(int(group_id)).nonzero(as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        order = torch.randperm(indices.numel(), generator=generator).to(
            indices.device
        )
        source_indices = indices.index_select(0, order)
        result.index_copy_(
            0, indices, reliability.index_select(0, source_indices)
        )
    return result


def _relative_metrics(prediction_difference, target_difference):
    prediction_difference = prediction_difference.detach().float().reshape(-1)
    target_difference = target_difference.detach().float().reshape(-1)
    valid = torch.isfinite(prediction_difference) & torch.isfinite(
        target_difference
    )
    prediction_difference = prediction_difference[valid]
    target_difference = target_difference[valid]
    if prediction_difference.numel() == 0:
        return {
            "count": 0,
            "mae": None,
            "pearson": None,
            "spearman": None,
            "sign_accuracy": None,
            "prediction_difference": _value_distribution(prediction_difference),
            "target_difference": _value_distribution(target_difference),
        }
    return {
        "count": int(prediction_difference.numel()),
        "mae": float(
            (prediction_difference - target_difference).abs().mean()
        ),
        "pearson": _pearson(prediction_difference, target_difference),
        "spearman": _pearson(
            _average_ranks(prediction_difference),
            _average_ranks(target_difference),
        ),
        "sign_accuracy": float(
            prediction_difference.ge(0).eq(target_difference.ge(0)).float().mean()
        ),
        "prediction_difference": _value_distribution(prediction_difference),
        "target_difference": _value_distribution(target_difference),
    }


def relative_importance_metrics(reliability, target, availability):
    """Measure image-vs-text importance only when both modalities are available."""

    reliability, availability = _validate_importance_pair(
        reliability, availability
    )
    target = target.detach().to(device=reliability.device, dtype=torch.float32)
    if target.shape != reliability.shape:
        raise ValueError("reliability, target, and availability must share shape")
    complete = availability.gt(0.5).all(dim=1) & torch.isfinite(target).all(dim=1)
    prediction_difference = (
        reliability[complete, 0] - reliability[complete, 1]
    )
    target_difference = target[complete, 0] - target[complete, 1]
    result = _relative_metrics(prediction_difference, target_difference)
    result["baselines"] = {
        "zero_difference": _relative_metrics(
            torch.zeros_like(target_difference), target_difference
        )
    }
    return result


def gate_temperature_sweep(candidate_losses, candidates, temperatures):
    """Summarize how temperature smooths fixed candidate-loss supervision."""

    candidate_losses = candidate_losses.detach().float()
    candidate_values = torch.as_tensor(
        tuple(float(value) for value in candidates),
        device=candidate_losses.device,
        dtype=candidate_losses.dtype,
    )
    if candidate_losses.dim() != 2:
        raise ValueError("candidate_losses must have shape [N, K]")
    if candidate_losses.shape[1] != candidate_values.numel():
        raise ValueError("candidate_losses and candidates must agree")
    if candidate_losses.shape[0] == 0 or candidate_values.numel() < 2:
        raise ValueError("temperature diagnostics require non-empty candidates")

    minimum = candidate_losses.min(dim=-1, keepdim=True).values
    loss_range = (
        candidate_losses.max(dim=-1).values
        - candidate_losses.min(dim=-1).values
    )
    hard_index = candidate_losses.argmin(dim=-1)
    hard_target = candidate_values.index_select(0, hard_index)
    endpoint = hard_index.eq(0) | hard_index.eq(candidate_values.numel() - 1)
    exact_tie = loss_range.eq(0)
    normalizer = torch.log(
        candidate_losses.new_tensor(float(candidate_values.numel()))
    )

    result = {}
    for temperature in temperatures:
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError("gate temperatures must be greater than zero")
        probabilities = torch.softmax(
            -(candidate_losses - minimum) / temperature, dim=-1
        )
        target = (probabilities * candidate_values).sum(dim=-1)
        normalized_entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1) / normalizer
        near_tie = loss_range.le(temperature)
        result[str(temperature)] = {
            "target": gate_distribution(target),
            "normalized_entropy": gate_distribution(normalized_entropy),
            "hard_target_mae": float((target - hard_target).abs().mean()),
            "hard_target_pearson": _pearson(target, hard_target),
            "hard_target_spearman": _pearson(
                _average_ranks(target), _average_ranks(hard_target)
            ),
            "endpoint_count": int(endpoint.sum()),
            "endpoint_target_mae": (
                float((target[endpoint] - hard_target[endpoint]).abs().mean())
                if endpoint.any() else None
            ),
            "exact_tie_count": int(exact_tie.sum()),
            "exact_tie_target_mean": (
                float(target[exact_tie].mean()) if exact_tie.any() else None
            ),
            "loss_range_le_temperature": {
                "count": int(near_tie.sum()),
                "fraction": float(near_tie.float().mean()),
                "center_abs_mean": (
                    float((target[near_tie] - 0.5).abs().mean())
                    if near_tie.any() else None
                ),
            },
        }
    return result


def continuous_target_metrics(
    prediction, target, calibration_bins=10
):
    """Metrics for a probability prediction against a continuous target."""

    prediction = prediction.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    prediction = prediction[valid]
    target = target[valid]
    if prediction.numel() == 0:
        return {
            "count": 0,
            "mae": None,
            "brier": None,
            "pearson": None,
            "spearman": None,
            "calibration_error": None,
            "prediction": _value_distribution(prediction),
            "target": _value_distribution(target),
        }
    calibration_bins = int(calibration_bins)
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be at least 1")
    clipped = prediction.clamp(0.0, 1.0)
    bin_index = torch.floor(clipped * calibration_bins).long().clamp(
        max=calibration_bins - 1
    )
    calibration_error = prediction.new_tensor(0.0)
    for index in range(calibration_bins):
        selected = bin_index.eq(index)
        if selected.any():
            calibration_error += (
                selected.float().mean()
                * (prediction[selected].mean() - target[selected].mean()).abs()
            )
    return {
        "count": int(prediction.numel()),
        "mae": float((prediction - target).abs().mean()),
        "brier": float((prediction - target).square().mean()),
        "pearson": _pearson(prediction, target),
        "spearman": _pearson(
            _average_ranks(prediction), _average_ranks(target)
        ),
        "calibration_error": float(calibration_error),
        "prediction": _value_distribution(prediction),
        "target": _value_distribution(target),
    }


def importance_diagnostics(
    reliability,
    target,
    availability,
    missing_type,
    calibration_bins=10,
):
    """Summarize image/text importance quality on supervised entries only."""

    reliability = reliability.detach().float()
    target = target.detach().float()
    availability = availability.detach().float()
    missing_type = missing_type.detach().long().reshape(-1)
    if reliability.shape != target.shape or reliability.shape != availability.shape:
        raise ValueError("reliability, target, and availability must share shape")
    if reliability.dim() != 2 or reliability.shape[1] != 2:
        raise ValueError("importance tensors must have shape [N, 2]")
    if reliability.shape[0] != missing_type.numel():
        raise ValueError("missing_type length must match importance tensors")

    condition_names = {0: "complete", 1: "image_only", 2: "text_only"}
    result = {}
    for modality_index, modality_name in enumerate(("image", "text")):
        supervised = availability[:, modality_index].gt(0.5) & torch.isfinite(
            target[:, modality_index]
        )
        prediction_values = reliability[supervised, modality_index]
        target_values = target[supervised, modality_index]
        groups = {}
        for condition, condition_name in condition_names.items():
            selected = supervised & missing_type.eq(condition)
            if selected.any():
                groups[condition_name] = continuous_target_metrics(
                    reliability[selected, modality_index],
                    target[selected, modality_index],
                    calibration_bins=calibration_bins,
                )

        constant_prediction = torch.full_like(target_values, 0.5)
        group_mean_prediction = torch.empty_like(target_values)
        supervised_conditions = missing_type[supervised]
        for condition in torch.unique(supervised_conditions).tolist():
            selected = supervised_conditions.eq(int(condition))
            group_mean_prediction[selected] = target_values[selected].mean()
        result[modality_name] = {
            "overall": continuous_target_metrics(
                prediction_values,
                target_values,
                calibration_bins=calibration_bins,
            ),
            "groups": groups,
            "baselines": {
                "constant_0_5": continuous_target_metrics(
                    constant_prediction,
                    target_values,
                    calibration_bins=calibration_bins,
                ),
                "group_mean": continuous_target_metrics(
                    group_mean_prediction,
                    target_values,
                    calibration_bins=calibration_bins,
                ),
            },
        }
    return result
