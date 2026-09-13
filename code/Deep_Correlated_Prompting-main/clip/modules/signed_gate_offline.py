"""Pure helpers for frozen-feature signed Gate experiments."""

import torch


REQUIRED_CACHE_TENSORS = (
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
    "signed_advantage",
)


def deterministic_sample_indices(total_count, sample_count, seed):
    """Return a reproducible random subset, or the full ordered dataset."""

    total_count = int(total_count)
    sample_count = int(sample_count)
    if total_count < 1:
        raise ValueError("total_count must be positive")
    if sample_count <= 0 or sample_count >= total_count:
        return torch.arange(total_count, dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(total_count, generator=generator)[:sample_count]


def validate_signed_gate_cache(cache):
    """Validate tensor identity and the endpoint-advantage invariant."""

    missing = [key for key in REQUIRED_CACHE_TENSORS if key not in cache]
    if missing:
        raise ValueError("signed Gate cache is missing {}".format(missing))
    if not all(torch.is_tensor(cache[key]) for key in REQUIRED_CACHE_TENSORS):
        raise ValueError("all signed Gate cache entries must be tensors")
    sample_count = int(cache["sample_id"].reshape(-1).numel())
    if sample_count < 1:
        raise ValueError("signed Gate cache must contain samples")
    for key in REQUIRED_CACHE_TENSORS:
        if cache[key].shape[0] != sample_count:
            raise ValueError("{} has inconsistent sample count".format(key))
    if torch.unique(cache["sample_id"].reshape(-1)).numel() != sample_count:
        raise ValueError("sample_id values must be unique within a split")
    if cache["availability"].shape != (sample_count, 2):
        raise ValueError("availability must have shape [N, 2]")
    if cache["reliability"].shape != (sample_count, 2):
        raise ValueError("reliability must have shape [N, 2]")
    if cache["base_context"].dim() != 2:
        raise ValueError("base_context must have shape [N, D]")
    if cache["labels"].shape != cache["g0_logits"].shape:
        raise ValueError("labels and g0_logits must share shape")
    if cache["labels"].shape != cache["g1_logits"].shape:
        raise ValueError("labels and g1_logits must share shape")
    for key in ("g0_loss", "g1_loss", "signed_advantage"):
        if cache[key].reshape(-1).numel() != sample_count:
            raise ValueError("{} must contain one value per sample".format(key))
        if not torch.isfinite(cache[key].float()).all():
            raise ValueError("{} contains non-finite values".format(key))
    expected = cache["g0_loss"].float() - cache["g1_loss"].float()
    actual = cache["signed_advantage"].float()
    if not torch.allclose(actual, expected, atol=1e-7, rtol=1e-6):
        raise ValueError("signed_advantage must equal g0_loss - g1_loss")
    return sample_count


def compose_gate_inputs(cache, source):
    """Compose one explicit input source without mixing attribution roles."""

    availability = cache["availability"].detach().float()
    reliability = cache["reliability"].detach().float()
    context = cache["base_context"].detach().float()
    if availability.shape != reliability.shape or availability.shape[1] != 2:
        raise ValueError("availability and reliability must have shape [N, 2]")
    if context.dim() != 2 or context.shape[0] != availability.shape[0]:
        raise ValueError("base_context must have shape [N, D]")
    importance = torch.cat(
        [(reliability - 0.5) * availability, availability], dim=-1
    )
    source = str(source)
    if source == "availability":
        return availability
    if source == "importance":
        return importance
    if source == "context":
        return context
    if source == "context_importance":
        return torch.cat([importance, context], dim=-1)
    raise ValueError("unknown signed Gate input source: {}".format(source))
