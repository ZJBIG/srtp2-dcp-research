import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "clip"
    / "modules"
    / "reliability_learning.py"
)
SPEC = importlib.util.spec_from_file_location("reliability_learning", MODULE_PATH)
reliability_learning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reliability_learning)


class ReliabilityLearningTests(unittest.TestCase):
    def test_availability_masks_follow_missing_types(self):
        result = reliability_learning.availability_from_missing_type(
            [0, 1, 2], torch.device("cpu")
        )
        expected = torch.tensor([[1., 1.], [1., 0.], [0., 1.]])
        self.assertTrue(torch.equal(result, expected))

    def test_predictor_outputs_independent_sigmoids(self):
        predictor = reliability_learning.ReliabilityPredictor(3, 5, hidden_dim=4)
        torch.nn.init.zeros_(predictor.predictor[-1].weight)
        torch.nn.init.constant_(predictor.predictor[-1].bias, 2.0)
        logits, reliability = predictor(
            torch.randn(2, 3), torch.randn(2, 5), torch.ones(2, 2)
        )
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.all(reliability.sum(-1) > 1.0))

    def test_relative_predictor_is_complementary_and_neutral_when_missing(self):
        predictor = reliability_learning.ReliabilityPredictor(
            3, 5, hidden_dim=4, importance_mode="relative"
        )
        torch.nn.init.zeros_(predictor.predictor[-1].weight)
        predictor.predictor[-1].bias.data.copy_(torch.tensor([2.0, -1.0]))
        logits, reliability = predictor(
            torch.randn(3, 3),
            torch.randn(3, 5),
            torch.tensor([[1., 1.], [1., 0.], [0., 1.]]),
        )
        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertAlmostEqual(reliability[0].sum().item(), 1.0, places=6)
        self.assertGreater(reliability[0, 0].item(), reliability[0, 1].item())
        self.assertTrue(
            torch.equal(reliability[1:], torch.full((2, 2), .5))
        )

    def test_unknown_importance_mode_fails_explicitly(self):
        with self.assertRaises(ValueError):
            reliability_learning.ReliabilityPredictor(
                3, 5, hidden_dim=4, importance_mode="unknown"
            )

    def test_relative_targets_supervise_complete_samples_only(self):
        target_fn = getattr(
            reliability_learning, "reliability_targets_from_delta"
        )
        delta = torch.tensor([[.4, .1], [.8, .2], [.3, .7]])
        availability = torch.tensor([[1., 1.], [1., 0.], [0., 1.]])
        target, supervision = target_fn(
            delta, availability, temperature=.1, importance_mode="relative"
        )
        expected_image = torch.sigmoid(torch.tensor(3.0))
        self.assertTrue(
            torch.allclose(
                target[0], torch.stack([expected_image, 1.0 - expected_image])
            )
        )
        self.assertTrue(torch.equal(target[1:], torch.full((2, 2), .5)))
        self.assertTrue(
            torch.equal(
                supervision,
                torch.tensor([[1., 1.], [0., 0.], [0., 0.]])
            )
        )

    def test_absolute_targets_keep_available_modality_formula(self):
        target_fn = getattr(
            reliability_learning, "reliability_targets_from_delta"
        )
        delta = torch.tensor([[.2, -.1], [.4, .7]])
        availability = torch.tensor([[1., 1.], [1., 0.]])
        target, supervision = target_fn(
            delta, availability, temperature=.1, importance_mode="absolute"
        )
        self.assertTrue(
            torch.allclose(target[0], torch.sigmoid(delta[0] / .1))
        )
        self.assertAlmostEqual(target[1, 0].item(), torch.sigmoid(torch.tensor(4.)).item())
        self.assertEqual(target[1, 1].item(), .5)
        self.assertTrue(torch.equal(supervision, availability))

    def test_reliability_target_uses_only_frozen_base_losses(self):
        loss_base = torch.tensor([.2, .8])
        loss_without = torch.tensor([.8, .2])
        _, target_before = reliability_learning.availability_conditioned_target(
            loss_without, loss_base, .1
        )
        adapter = reliability_learning.PromptAdapter(1, 3, 3, hidden_dim=2)
        for parameter in adapter.parameters():
            torch.nn.init.normal_(parameter)
        _, target_after = reliability_learning.availability_conditioned_target(
            loss_without, loss_base, .1
        )
        self.assertTrue(torch.equal(target_before, target_after))

    def test_missing_modality_is_excluded_from_reliability_gradient(self):
        logits = torch.zeros(2, 2, requires_grad=True)
        targets = torch.tensor([[1., 0.], [0., 1.]])
        availability = torch.tensor([[1., 0.], [0., 1.]])
        element = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = (element * availability).sum() / availability.sum()
        loss.backward()
        self.assertEqual(logits.grad[0, 1].item(), 0.0)
        self.assertEqual(logits.grad[1, 0].item(), 0.0)
        self.assertNotEqual(logits.grad[0, 0].item(), 0.0)

    def test_bounded_adapter_formula_and_saturation(self):
        adapter = reliability_learning.PromptAdapter(
            1, 3, 3, hidden_dim=2, scale=.5
        )
        for projection in list(adapter.image_projections) + list(adapter.text_projections):
            torch.nn.init.zeros_(projection.weight)
            torch.nn.init.constant_(projection.bias, 100.0)
        prompts = [torch.ones(2, 2, 3)]
        result = adapter(
            prompts,
            prompts,
            torch.ones(2, 2),
            torch.ones(2, 2),
            torch.ones(2, 1),
            return_diagnostics=True,
        )
        for item in result[2]:
            self.assertTrue(torch.all(item["bounded_offset_prompt_ratio"] <= .5 + 1e-5))
            self.assertTrue(torch.equal(item["cap_saturation_ratio"], torch.ones(2)))

    def test_availability_aware_adapter_has_an_explicit_condition_path(self):
        adapter = reliability_learning.PromptAdapter(
            1, 2, 2, hidden_dim=1, use_availability=True
        )
        torch.nn.init.zeros_(adapter.shared[0].weight)
        torch.nn.init.zeros_(adapter.shared[0].bias)
        adapter.shared[0].weight.data[0, 3] = 1.0
        for projection in list(adapter.image_projections) + list(adapter.text_projections):
            torch.nn.init.ones_(projection.weight)
            torch.nn.init.zeros_(projection.bias)
        prompts = [torch.ones(2, 1, 2)]
        image_offsets, _, _ = adapter.compute_offsets(
            prompts,
            prompts,
            torch.full((2, 2), .5),
            torch.tensor([[1., 1.], [1., 0.]]),
        )
        self.assertEqual(adapter.shared[0].in_features, 4)
        self.assertFalse(torch.equal(image_offsets[0][0], image_offsets[0][1]))

    def test_task_loss_updates_adapter_only(self):
        predictor = reliability_learning.ReliabilityPredictor(3, 3, hidden_dim=4)
        adapter = reliability_learning.PromptAdapter(1, 3, 3, hidden_dim=4)
        for projection in list(adapter.image_projections) + list(adapter.text_projections):
            torch.nn.init.normal_(projection.weight)
        _, reliability = predictor(
            torch.randn(2, 3), torch.randn(2, 3), torch.ones(2, 2)
        )
        prompts = [torch.randn(2, 2, 3)]
        adapted = adapter(
            prompts, prompts, reliability.detach(), torch.ones(2, 2), torch.ones(2, 1)
        )
        sum(value.sum() for value in adapted[0] + adapted[1]).backward()
        self.assertTrue(any(p.grad is not None for p in adapter.parameters()))
        self.assertTrue(all(p.grad is None for p in predictor.parameters()))

    def test_reliability_loss_updates_predictor_only(self):
        predictor = reliability_learning.ReliabilityPredictor(3, 3, hidden_dim=4)
        adapter = reliability_learning.PromptAdapter(1, 3, 3, hidden_dim=4)
        logits, _ = predictor(
            torch.randn(2, 3), torch.randn(2, 3), torch.ones(2, 2)
        )
        F.binary_cross_entropy_with_logits(logits, torch.rand_like(logits)).backward()
        self.assertTrue(any(p.grad is not None for p in predictor.parameters()))
        self.assertTrue(all(p.grad is None for p in adapter.parameters()))

    def test_gate_trains_projector_but_detaches_context_and_reliability(self):
        gate = reliability_learning.UtilityGate(8, projector_dim=4, hidden_dim=4)
        reliability = torch.rand(3, 2, requires_grad=True)
        context = torch.randn(3, 8, requires_grad=True)
        optimizer = torch.optim.SGD(gate.parameters(), lr=.1)
        prediction = gate(reliability, torch.ones(3, 2), context)
        F.smooth_l1_loss(prediction.reshape(-1), torch.tensor([0., .5, 1.])).backward()
        optimizer.step()
        optimizer.zero_grad()
        prediction = gate(reliability, torch.ones(3, 2), context)
        F.smooth_l1_loss(prediction.reshape(-1), torch.tensor([0., .5, 1.])).backward()
        projector = gate.context_projector[1]
        self.assertIsNotNone(projector.weight.grad)
        self.assertGreater(projector.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(reliability.grad)
        self.assertIsNone(context.grad)

    def test_continuous_gate_shape_range_and_neutral_initialization(self):
        gate = reliability_learning.UtilityGate(8, projector_dim=4, hidden_dim=4)
        output = gate(torch.rand(5, 2), torch.ones(5, 2), torch.randn(5, 8))
        self.assertEqual(tuple(output.shape), (5, 1))
        self.assertTrue(torch.equal(output, torch.full_like(output, .5)))

    def test_gate_ignores_reliability_of_unavailable_modality(self):
        torch.manual_seed(7)
        gate = reliability_learning.UtilityGate(8, projector_dim=4, hidden_dim=4)
        for parameter in gate.parameters():
            if parameter.dim() > 1:
                torch.nn.init.normal_(parameter)
            else:
                torch.nn.init.uniform_(parameter, -.3, .3)
        availability = torch.tensor([[1., 0.], [0., 1.]])
        context = torch.randn(2, 8)
        first = torch.tensor([[.8, .1], [.2, .7]])
        changed_only_when_missing = torch.tensor([[.8, .9], [.9, .7]])
        first_logits = gate.logits(first, availability, context)
        changed_logits = gate.logits(
            changed_only_when_missing, availability, context
        )
        self.assertTrue(torch.allclose(first_logits, changed_logits))

    def test_soft_gate_target_is_neutral_for_exact_and_near_ties(self):
        target_fn = getattr(reliability_learning, "soft_gate_target")
        losses = torch.tensor(
            [[1., 1., 1., 1., 1.], [1., 1.001, 1.002, 1.001, 1.]]
        )
        target, probabilities, loss_range = target_fn(
            losses, [0., .25, .5, .75, 1.], temperature=.025
        )
        self.assertTrue(torch.allclose(target[0], torch.tensor(.5)))
        self.assertTrue(
            torch.allclose(probabilities[0], torch.full((5,), .2))
        )
        self.assertEqual(loss_range[0].item(), 0.0)
        self.assertLess(abs(target[1].item() - .5), .02)

    def test_soft_gate_target_tracks_clear_optimum(self):
        target_fn = getattr(reliability_learning, "soft_gate_target")
        losses = torch.tensor(
            [[0., 1., 2., 3., 4.], [4., 3., 0., 3., 4.], [4., 3., 2., 1., 0.]]
        )
        target, _, _ = target_fn(
            losses, [0., .25, .5, .75, 1.], temperature=.025
        )
        self.assertTrue(torch.allclose(target, torch.tensor([0., .5, 1.]), atol=1e-4))

    def test_gate_candidates_are_supervision_only(self):
        candidates = reliability_learning.validate_gate_candidates([0, .25, .5, .75, 1])
        losses = torch.tensor([[3., 2., 1., 2., 3.], [1., 2., 3., 4., 5.]])
        values = torch.tensor(candidates)
        g_star = values.index_select(0, losses.argmin(-1))
        self.assertTrue(torch.equal(g_star, torch.tensor([.5, 0.])))

    def test_endpoint_gate_target_uses_g0_g1_and_neutral_ties(self):
        target_fn = getattr(reliability_learning, "endpoint_gate_target")
        losses = torch.tensor(
            [
                [0., 1., 2., 3., 4.],
                [4., 3., 2., 1., 0.],
                [1., 2., 3., 2., 1.],
            ]
        )
        target, regret = target_fn(
            losses, [0., .25, .5, .75, 1.]
        )
        self.assertTrue(torch.equal(target, torch.tensor([0., 1., .5])))
        self.assertTrue(torch.equal(regret, torch.tensor([4., 4., 0.])))

    def test_regret_weighted_gate_loss_ignores_zero_regret_ties(self):
        loss_fn = getattr(
            reliability_learning, "regret_weighted_binary_gate_loss"
        )
        logits = torch.tensor([2., -9.], requires_grad=True)
        target = torch.tensor([0., .5])
        regret = torch.tensor([3., 0.])
        loss = loss_fn(logits, target, regret)
        expected = F.binary_cross_entropy_with_logits(logits[:1], target[:1])
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertEqual(logits.grad[1].item(), 0.0)

    def test_gate_supervision_mode_is_explicit(self):
        validator = getattr(
            reliability_learning, "validate_gate_supervision_mode"
        )
        self.assertEqual(validator("soft_scalar"), "soft_scalar")
        self.assertEqual(validator("binary_regret"), "binary_regret")
        self.assertEqual(validator("direct_task"), "direct_task")
        with self.assertRaises(ValueError):
            validator("categorical")


if __name__ == "__main__":
    unittest.main()
