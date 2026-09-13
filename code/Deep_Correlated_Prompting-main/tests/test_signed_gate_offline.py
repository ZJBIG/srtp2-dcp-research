import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "clip"
    / "modules"
    / "signed_gate_offline.py"
)
if MODULE_PATH.exists():
    SPEC = importlib.util.spec_from_file_location(
        "signed_gate_offline", MODULE_PATH
    )
    offline = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(offline)
else:
    offline = None


class SignedGateOfflineTests(unittest.TestCase):
    def test_offline_module_exists(self):
        self.assertIsNotNone(offline)

    @unittest.skipIf(offline is None, "offline module not implemented")
    def test_deterministic_indices_are_reproducible_and_unique(self):
        select = getattr(offline, "deterministic_sample_indices")
        first = select(total_count=20, sample_count=7, seed=13)
        second = select(total_count=20, sample_count=7, seed=13)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.numel(), 7)
        self.assertEqual(torch.unique(first).numel(), 7)
        self.assertTrue(torch.equal(select(5, 0, 13), torch.arange(5)))

    @unittest.skipIf(offline is None, "offline module not implemented")
    def test_cache_validation_checks_signed_advantage(self):
        validate = getattr(offline, "validate_signed_gate_cache")
        cache = {
            "sample_id": torch.tensor([3, 8]),
            "missing_type": torch.tensor([0, 1]),
            "availability": torch.tensor([[1., 1.], [1., 0.]]),
            "reliability": torch.tensor([[.6, .4], [.5, .5]]),
            "base_context": torch.ones(2, 4),
            "labels": torch.ones(2, 3),
            "g0_logits": torch.zeros(2, 3),
            "g1_logits": torch.ones(2, 3),
            "g0_loss": torch.tensor([.4, .2]),
            "g1_loss": torch.tensor([.1, .3]),
            "signed_advantage": torch.tensor([.3, -.1]),
        }
        self.assertEqual(validate(cache), 2)
        cache["signed_advantage"] = torch.tensor([.3, .1])
        with self.assertRaises(ValueError):
            validate(cache)

    @unittest.skipIf(offline is None, "offline module not implemented")
    def test_input_sources_keep_importance_and_context_separate(self):
        compose = getattr(offline, "compose_gate_inputs")
        cache = {
            "availability": torch.tensor([[1., 1.], [1., 0.]]),
            "reliability": torch.tensor([[.7, .3], [.5, .5]]),
            "base_context": torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
        }
        availability = compose(cache, "availability")
        importance = compose(cache, "importance")
        context = compose(cache, "context")
        combined = compose(cache, "context_importance")
        self.assertEqual(tuple(availability.shape), (2, 2))
        self.assertEqual(tuple(importance.shape), (2, 4))
        self.assertEqual(tuple(context.shape), (2, 3))
        self.assertEqual(tuple(combined.shape), (2, 7))
        self.assertTrue(torch.equal(combined[:, :4], importance))
        self.assertTrue(torch.equal(combined[:, 4:], context))


if __name__ == "__main__":
    unittest.main()
