"""Counterfactual proxy features for missing-modality learning.

This module implements a trainable first approximation to a conditional
plausible set.  A support proxy supplies the task-relevant missing feature,
while a bounded challenge proxy is trained adversarially through gradient
reversal.  The downstream classifier therefore learns against a hard but
plausible completion without requiring alternating optimizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def gradient_reverse(x, scale=1.0):
    return _GradientReverse.apply(x, float(scale))


class DirectionalProxyGenerator(nn.Module):
    """Generate support/challenge proxies for one directed modality edge."""

    def __init__(self, feature_dim, hidden_dim, identity_dim, challenge_radius):
        super().__init__()
        self.challenge_radius = float(challenge_radius)
        condition_dim = feature_dim + 2 * identity_dim

        self.source_norm = nn.LayerNorm(feature_dim)
        self.message = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.target_prior = nn.Parameter(torch.empty(feature_dim))
        self.challenge = nn.Sequential(
            nn.Linear(condition_dim + feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.challenge_scale = nn.Sequential(
            nn.Linear(condition_dim + feature_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.target_prior, std=0.02)

    def forward(self, source, direction_identity, missing_identity):
        source = self.source_norm(source.float())
        condition = torch.cat([source, direction_identity, missing_identity], dim=-1)
        message = self.message(condition)
        gate = self.gate(condition)
        support = self.target_prior.unsqueeze(0) + gate * message

        # Detaching the support here isolates the adversary: the support proxy
        # remains optimized by task/alignment losses, while only the bounded
        # residual is trained to challenge the classifier.
        challenge_input = torch.cat([condition, support.detach()], dim=-1)
        delta = self.challenge(challenge_input)
        delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta * self.challenge_scale(challenge_input) * self.challenge_radius
        return support, delta, gate


class CounterfactualProxyGame(nn.Module):
    """Bidirectional support/challenge proxies for image-text features.

    Missing type convention follows DCP:
      0: complete, 1: missing text, 2: missing image.
    """

    def __init__(
        self,
        feature_dim=512,
        hidden_dim=512,
        identity_dim=32,
        challenge_radius=1.0,
        adversary_strength=1.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.adversary_strength = float(adversary_strength)
        self.direction_identity = nn.Embedding(2, identity_dim)
        self.missing_identity = nn.Embedding(3, identity_dim)
        self.text_from_image = DirectionalProxyGenerator(
            feature_dim, hidden_dim, identity_dim, challenge_radius
        )
        self.image_from_text = DirectionalProxyGenerator(
            feature_dim, hidden_dim, identity_dim, challenge_radius
        )

    @staticmethod
    def _alignment_loss(prediction, target):
        prediction = F.normalize(prediction.float(), dim=-1)
        target = F.normalize(target.detach().float(), dim=-1)
        return (1.0 - (prediction * target).sum(dim=-1)).mean()

    @staticmethod
    def _masked_mean(values, mask):
        if mask.any():
            return values[mask].mean()
        return values.sum() * 0.0

    def forward(self, image_features, text_features, missing_type):
        device = image_features.device
        missing_type = torch.as_tensor(missing_type, device=device, dtype=torch.long)
        if ((missing_type < 0) | (missing_type > 2)).any():
            raise ValueError("missing_type must contain only 0 (complete), 1 (text), or 2 (image)")

        batch_size = image_features.shape[0]
        direction_ids = torch.arange(2, device=device).view(2, 1).expand(2, batch_size)
        direction_emb = self.direction_identity(direction_ids)
        missing_emb = self.missing_identity(missing_type)

        text_support, text_delta, text_gate = self.text_from_image(
            image_features, direction_emb[0], missing_emb
        )
        image_support, image_delta, image_gate = self.image_from_text(
            text_features, direction_emb[1], missing_emb
        )
        text_challenge = text_support.detach() + text_delta
        image_challenge = image_support.detach() + image_delta

        complete = missing_type.eq(0)
        missing_text = missing_type.eq(1)
        missing_image = missing_type.eq(2)

        point_image = torch.where(missing_image[:, None], image_support, image_features.float())
        point_text = torch.where(missing_text[:, None], text_support, text_features.float())

        # The classifier receives an ordinary gradient, while proxy generators
        # receive its sign-reversed gradient and learn hard counterfactuals.
        adversarial_image = image_support + gradient_reverse(
            image_delta, self.adversary_strength
        )
        adversarial_text = text_support + gradient_reverse(
            text_delta, self.adversary_strength
        )
        challenge_image_out = torch.where(missing_image[:, None], adversarial_image, point_image)
        challenge_text_out = torch.where(missing_text[:, None], adversarial_text, point_text)

        zero = image_features.float().sum() * 0.0
        if complete.any():
            alignment_loss = self._alignment_loss(text_support[complete], text_features[complete])
            alignment_loss = alignment_loss + self._alignment_loss(
                image_support[complete], image_features[complete]
            )
        else:
            alignment_loss = zero

        # A bounded perturbation alone is not enough: this term keeps challenge
        # proxies close to the learned conditional support representation.
        text_challenge_distance = 1.0 - F.cosine_similarity(
            text_challenge, text_support.detach(), dim=-1
        )
        image_challenge_distance = 1.0 - F.cosine_similarity(
            image_challenge, image_support.detach(), dim=-1
        )
        active_missing = missing_text | missing_image
        challenge_distance = torch.where(
            missing_text, text_challenge_distance, image_challenge_distance
        )
        plausibility_loss = self._masked_mean(challenge_distance, active_missing)

        return {
            "cls_feats": torch.cat([point_image, point_text], dim=-1),
            "challenge_cls_feats": torch.cat(
                [challenge_image_out, challenge_text_out], dim=-1
            ),
            "counterfactual_mask": active_missing,
            "proxy_alignment_loss": alignment_loss,
            "proxy_plausibility_loss": plausibility_loss,
            "proxy_gate_mean": torch.stack([image_gate.mean(), text_gate.mean()]).mean(),
            "text_support_proxy": text_support,
            "image_support_proxy": image_support,
            # Raw features are exposed only for complete-pair regularization.
            # Features from a missing branch still encode the dataset dummy and
            # must never be treated as observed modality evidence.
            "raw_image_features": image_features.float(),
            "raw_text_features": text_features.float(),
        }
