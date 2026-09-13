"""Final availability-conditioned Reliability-DCP components."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_reliability_importance_mode(importance_mode):
    importance_mode = str(importance_mode)
    if importance_mode not in ("absolute", "relative"):
        raise ValueError(
            "reliability_importance_mode must be 'absolute' or 'relative'"
        )
    return importance_mode


def validate_gate_supervision_mode(supervision_mode):
    supervision_mode = str(supervision_mode)
    if supervision_mode not in (
        "soft_scalar", "binary_regret", "direct_task"
    ):
        raise ValueError(
            "gate_supervision_mode must be 'soft_scalar', "
            "'binary_regret', or 'direct_task'"
        )
    return supervision_mode


def availability_conditioned_target(
    loss_without_modality, loss_with_modality, temperature
):
    """Return delta_m and sigmoid(delta_m / temperature)."""

    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("Reliability temperature must be greater than zero")
    delta = loss_without_modality - loss_with_modality
    return delta, torch.sigmoid(delta / temperature)


def reliability_targets_from_delta(
    delta, availability_mask, temperature, importance_mode="absolute"
):
    """Build absolute or complete-only relative Reliability supervision."""

    importance_mode = validate_reliability_importance_mode(importance_mode)
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("Reliability temperature must be greater than zero")
    if delta.dim() != 2 or delta.shape[1] != 2:
        raise ValueError("delta must have shape [N, 2]")
    availability = availability_mask.to(delta)
    if availability.shape != delta.shape:
        raise ValueError("delta and availability_mask must share shape")

    if importance_mode == "absolute":
        supervision = availability
        target = torch.sigmoid(delta / temperature)
        target = torch.where(
            supervision.gt(0.5), target, torch.full_like(target, 0.5)
        )
        return target, supervision

    complete = availability.gt(0.5).all(dim=1, keepdim=True)
    image_probability = torch.sigmoid(
        (delta[:, 0:1] - delta[:, 1:2]) / temperature
    )
    relative_target = torch.cat(
        [image_probability, 1.0 - image_probability], dim=-1
    )
    target = torch.where(
        complete, relative_target, torch.full_like(relative_target, 0.5)
    )
    supervision = complete.to(delta).expand_as(delta)
    return target, supervision


def validate_gate_candidates(candidates):
    values = tuple(float(value) for value in candidates)
    if values != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("gate_candidates must be [0, 0.25, 0.5, 0.75, 1]")
    return values


def soft_gate_target(candidate_losses, candidates, temperature):
    """Return a tie-aware continuous target from candidate task losses."""

    values = validate_gate_candidates(candidates)
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("gate_target_temperature must be greater than zero")
    if candidate_losses.dim() < 1 or candidate_losses.shape[-1] != len(values):
        raise ValueError(
            "candidate_losses last dimension must match gate_candidates"
        )
    candidate_values = torch.as_tensor(
        values, device=candidate_losses.device, dtype=candidate_losses.dtype
    )
    minimum = candidate_losses.min(dim=-1, keepdim=True).values
    probabilities = torch.softmax(
        -(candidate_losses - minimum) / temperature, dim=-1
    )
    target = (probabilities * candidate_values).sum(dim=-1)
    loss_range = (
        candidate_losses.max(dim=-1).values
        - candidate_losses.min(dim=-1).values
    )
    return target, probabilities, loss_range


def endpoint_gate_target(candidate_losses, candidates):
    """Return the better endpoint and the cost of choosing the wrong one."""

    values = validate_gate_candidates(candidates)
    if candidate_losses.dim() < 1 or candidate_losses.shape[-1] != len(values):
        raise ValueError(
            "candidate_losses last dimension must match gate_candidates"
        )
    loss_g0 = candidate_losses[..., 0]
    loss_g1 = candidate_losses[..., -1]
    regret = (loss_g1 - loss_g0).abs()
    target = loss_g1.lt(loss_g0).to(candidate_losses)
    target = torch.where(
        regret.eq(0), torch.full_like(target, 0.5), target
    )
    return target, regret


def regret_weighted_binary_gate_loss(logits, target, regret):
    """Binary endpoint loss weighted by the per-sample wrong-choice regret."""

    logits = logits.reshape(-1)
    target = target.to(logits).reshape(-1)
    regret = regret.to(logits).detach().reshape(-1)
    if logits.shape != target.shape or logits.shape != regret.shape:
        raise ValueError("logits, target, and regret must share shape")
    element_loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    return (element_loss * regret).sum() / regret.sum().clamp_min(1e-8)


def availability_from_missing_type(missing_type, device, dtype=torch.float32):
    """Convert DCP missing types to [image, text] availability masks."""

    missing = torch.as_tensor(missing_type, device=device, dtype=torch.long)
    if missing.dim() == 0:
        missing = missing.unsqueeze(0)
    if ((missing < 0) | (missing > 2)).any():
        raise ValueError("missing_type must contain only 0, 1, or 2")
    availability = torch.ones(missing.shape[0], 2, device=device, dtype=dtype)
    availability[:, 0] = missing.ne(2).to(dtype=dtype)
    availability[:, 1] = missing.ne(1).to(dtype=dtype)
    return availability


class ReliabilityPredictor(nn.Module):
    """Predict independent image and text reliability logits."""

    def __init__(
        self,
        image_feature_dim,
        text_feature_dim,
        hidden_dim=256,
        importance_mode="absolute",
    ):
        super().__init__()
        self.importance_mode = validate_reliability_importance_mode(
            importance_mode
        )
        self.image_norm = nn.LayerNorm(int(image_feature_dim))
        self.text_norm = nn.LayerNorm(int(text_feature_dim))
        input_dim = int(image_feature_dim) + int(text_feature_dim) + 2
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, image_features, text_features, availability_mask):
        availability = availability_mask.to(image_features)
        image_features = self.image_norm(image_features) * availability[:, 0:1]
        text_features = self.text_norm(text_features) * availability[:, 1:2]
        raw_logits = self.predictor(
            torch.cat([image_features, text_features, availability], dim=-1)
        )
        if self.importance_mode == "absolute":
            return raw_logits, torch.sigmoid(raw_logits)

        relative_logit = raw_logits[:, 0:1] - raw_logits[:, 1:2]
        logits = torch.cat([relative_logit, -relative_logit], dim=-1)
        reliability = torch.sigmoid(logits)
        complete = availability.gt(0.5).all(dim=1, keepdim=True)
        reliability = torch.where(
            complete, reliability, torch.full_like(reliability, 0.5)
        )
        return logits, reliability


class PromptAdapter(nn.Module):
    """Produce bounded, depth-specific prompt adjustments from Reliability."""

    def __init__(
        self,
        prompt_depth,
        image_prompt_dim,
        text_prompt_dim,
        hidden_dim=64,
        scale=1.0,
        use_availability=False,
    ):
        super().__init__()
        self.prompt_depth = int(prompt_depth)
        self.scale = float(scale)
        self.use_availability = bool(use_availability)
        if self.scale <= 0:
            raise ValueError("adapter_scale must be positive")
        input_dim = 4 if self.use_availability else 2
        self.shared = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)), nn.GELU()
        )
        self.image_projections = nn.ModuleList(
            [nn.Linear(int(hidden_dim), int(image_prompt_dim)) for _ in range(self.prompt_depth)]
        )
        self.text_projections = nn.ModuleList(
            [nn.Linear(int(hidden_dim), int(text_prompt_dim)) for _ in range(self.prompt_depth)]
        )
        for projection in list(self.image_projections) + list(self.text_projections):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    @staticmethod
    def _apply_gate(prompts, offsets, gate):
        strength = gate.view(-1, 1, 1)
        return [
            prompt + strength.to(prompt) * offset.to(prompt)
            for prompt, offset in zip(prompts, offsets)
        ]

    def _bounded_offset(self, raw_offset, prompt):
        prompt_rms = (
            prompt.detach().float().pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
        )
        cap = (self.scale * prompt_rms).clamp_min(1e-6).to(raw_offset)
        bounded = cap * torch.tanh(raw_offset / (cap + 1e-6))
        return bounded, cap

    @staticmethod
    def _diagnostics(depth, modality, prompt, raw, bounded, cap):
        raw_expanded = raw.expand_as(prompt)
        bounded_expanded = bounded.expand_as(prompt)
        prompt_norm = prompt.float().norm(dim=(1, 2))
        raw_norm = raw_expanded.float().norm(dim=(1, 2))
        bounded_norm = bounded_expanded.float().norm(dim=(1, 2))
        saturation = raw.abs().ge(cap).float().mean(dim=(1, 2))
        return {
            "depth": depth,
            "modality": modality,
            "prompt_norm": prompt_norm,
            "raw_offset_norm": raw_norm,
            "bounded_offset_norm": bounded_norm,
            "raw_offset_prompt_ratio": raw_norm / prompt_norm.clamp_min(1e-8),
            "bounded_offset_prompt_ratio": bounded_norm / prompt_norm.clamp_min(1e-8),
            "cap_saturation_ratio": saturation,
        }

    def compute_offsets(
        self,
        image_prompts,
        text_prompts,
        reliability,
        availability_mask,
        return_diagnostics=False,
    ):
        if len(image_prompts) != self.prompt_depth or len(text_prompts) != self.prompt_depth:
            raise ValueError("Prompt list length must equal configured prompt_depth")
        availability = availability_mask.to(reliability)
        centered_reliability = (reliability.detach() - 0.5) * availability
        adapter_input = centered_reliability
        if self.use_availability:
            adapter_input = torch.cat(
                [centered_reliability, availability], dim=-1
            )
        hidden = self.shared(adapter_input)
        image_offsets, text_offsets, diagnostics = [], [], []
        for depth in range(self.prompt_depth):
            raw_image = self.image_projections[depth](hidden).unsqueeze(1)
            raw_text = self.text_projections[depth](hidden).unsqueeze(1)
            image_offset, image_cap = self._bounded_offset(
                raw_image, image_prompts[depth]
            )
            text_offset, text_cap = self._bounded_offset(raw_text, text_prompts[depth])
            image_offsets.append(image_offset)
            text_offsets.append(text_offset)
            if return_diagnostics:
                diagnostics.extend(
                    [
                        self._diagnostics(
                            depth,
                            "image",
                            image_prompts[depth],
                            raw_image,
                            image_offset,
                            image_cap,
                        ),
                        self._diagnostics(
                            depth,
                            "text",
                            text_prompts[depth],
                            raw_text,
                            text_offset,
                            text_cap,
                        ),
                    ]
                )
        return image_offsets, text_offsets, diagnostics

    def forward(
        self,
        image_prompts,
        text_prompts,
        reliability,
        availability_mask,
        utility_gate,
        return_diagnostics=False,
    ):
        image_offsets, text_offsets, diagnostics = self.compute_offsets(
            image_prompts,
            text_prompts,
            reliability,
            availability_mask,
            return_diagnostics=return_diagnostics,
        )
        result = (
            self._apply_gate(image_prompts, image_offsets, utility_gate),
            self._apply_gate(text_prompts, text_offsets, utility_gate),
        )
        if return_diagnostics:
            return result[0], result[1], diagnostics
        return result


class UtilityGate(nn.Module):
    """Continuous contextual strength controller for bounded prompt offsets."""

    def __init__(self, context_dim, projector_dim=64, hidden_dim=64):
        super().__init__()
        self.context_projector = nn.Sequential(
            nn.LayerNorm(int(context_dim)),
            nn.Linear(int(context_dim), int(projector_dim)),
            nn.GELU(),
        )
        gate_input_dim = 2 + 2 + int(projector_dim)
        self.input_norm = nn.LayerNorm(gate_input_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def logits(self, reliability, availability_mask, base_context):
        reliability = reliability.detach()
        availability = availability_mask.to(reliability)
        base_context = base_context.detach()
        projected_context = self.context_projector(base_context.to(reliability))
        centered_reliability = (reliability - 0.5) * availability
        gate_input = torch.cat(
            [centered_reliability, availability, projected_context], dim=-1
        )
        return self.gate_mlp(self.input_norm(gate_input))

    def forward(self, reliability, availability_mask, base_context):
        return torch.sigmoid(
            self.logits(reliability, availability_mask, base_context)
        )
