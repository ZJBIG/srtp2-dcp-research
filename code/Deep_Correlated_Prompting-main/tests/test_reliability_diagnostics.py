import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "clip"
    / "modules"
    / "reliability_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location("reliability_diagnostics", MODULE_PATH)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class ContinuousTargetMetricTests(unittest.TestCase):
    def test_perfect_prediction_has_zero_error_and_unit_correlations(self):
        metric = getattr(diagnostics, "continuous_target_metrics")
        target = torch.tensor([.1, .4, .9])
        result = metric(target, target, calibration_bins=5)
        self.assertEqual(result["count"], 3)
        self.assertAlmostEqual(result["mae"], 0.0, places=7)
        self.assertAlmostEqual(result["brier"], 0.0, places=7)
        self.assertAlmostEqual(result["calibration_error"], 0.0, places=7)
        self.assertAlmostEqual(result["pearson"], 1.0, places=6)
        self.assertAlmostEqual(result["spearman"], 1.0, places=6)

    def test_reverse_order_has_negative_rank_correlation(self):
        metric = getattr(diagnostics, "continuous_target_metrics")
        prediction = torch.tensor([.9, .6, .1])
        target = torch.tensor([.1, .4, .9])
        result = metric(prediction, target)
        self.assertAlmostEqual(result["spearman"], -1.0, places=6)

    def test_constant_prediction_has_defined_error_and_no_correlation(self):
        metric = getattr(diagnostics, "continuous_target_metrics")
        prediction = torch.full((4,), .5)
        target = torch.tensor([0., .25, .75, 1.])
        result = metric(prediction, target)
        self.assertIsNone(result["pearson"])
        self.assertIsNone(result["spearman"])
        self.assertAlmostEqual(result["mae"], .375, places=6)

    def test_importance_diagnostics_respects_availability_and_groups(self):
        summarize = getattr(diagnostics, "importance_diagnostics")
        reliability = torch.tensor(
            [[.2, .8], [.6, .1], [.9, .4], [.3, .7]]
        )
        target = torch.tensor(
            [[.1, .9], [.5, float("nan")], [float("nan"), .3], [.4, .6]]
        )
        availability = torch.tensor(
            [[1., 1.], [1., 0.], [0., 1.], [1., 1.]]
        )
        missing_type = torch.tensor([0, 1, 2, 0])
        result = summarize(
            reliability, target, availability, missing_type,
            calibration_bins=4,
        )
        self.assertEqual(result["image"]["overall"]["count"], 3)
        self.assertEqual(result["text"]["overall"]["count"], 3)
        self.assertEqual(result["image"]["groups"]["complete"]["count"], 2)
        self.assertEqual(result["image"]["groups"]["image_only"]["count"], 1)
        self.assertEqual(result["text"]["groups"]["text_only"]["count"], 1)
        self.assertIn("constant_0_5", result["image"]["baselines"])
        self.assertIn("group_mean", result["text"]["baselines"])


class ImportanceAttributionTests(unittest.TestCase):
    def test_condition_prior_replaces_only_within_availability_group(self):
        condition_prior = getattr(diagnostics, "condition_prior_reliability")
        reliability = torch.tensor(
            [[.1, .2], [.3, .4], [.8, .1], [.6, .9], [.2, .7]]
        )
        availability = torch.tensor(
            [[1., 1.], [1., 1.], [1., 0.], [1., 0.], [0., 1.]]
        )
        result = condition_prior(reliability, availability)
        expected = torch.tensor(
            [[.2, .3], [.2, .3], [.7, .5], [.7, .5], [.2, .7]]
        )
        self.assertTrue(torch.allclose(result, expected))

    def test_condition_shuffle_preserves_each_group_multiset(self):
        shuffle = getattr(diagnostics, "condition_preserving_shuffle")
        reliability = torch.tensor(
            [
                [.1, .2], [.3, .4], [.5, .6],
                [.7, .1], [.8, .2], [.9, .3],
                [.2, .7], [.3, .8], [.4, .9],
            ]
        )
        availability = torch.tensor(
            [[1., 1.]] * 3 + [[1., 0.]] * 3 + [[0., 1.]] * 3
        )
        first = shuffle(reliability, availability, seed=17)
        second = shuffle(reliability, availability, seed=17)
        self.assertTrue(torch.equal(first, second))
        for group in ((1., 1.), (1., 0.), (0., 1.)):
            selected = availability.eq(torch.tensor(group)).all(dim=1)
            before = sorted(tuple(row.tolist()) for row in reliability[selected])
            after = sorted(tuple(row.tolist()) for row in first[selected])
            self.assertEqual(before, after)

    def test_relative_metrics_use_complete_samples_only(self):
        relative_metrics = getattr(diagnostics, "relative_importance_metrics")
        prediction = torch.tensor(
            [[.8, .2], [.3, .7], [.6, .4], [.9, .1]]
        )
        target = torch.tensor(
            [[.7, .1], [.2, .6], [.55, .35], [.1, float("nan")]]
        )
        availability = torch.tensor(
            [[1., 1.], [1., 1.], [1., 1.], [1., 0.]]
        )
        result = relative_metrics(prediction, target, availability)
        self.assertEqual(result["count"], 3)
        self.assertAlmostEqual(result["pearson"], 1.0, places=6)
        self.assertAlmostEqual(result["spearman"], 1.0, places=6)
        self.assertAlmostEqual(result["sign_accuracy"], 1.0, places=6)
        self.assertAlmostEqual(result["mae"], 0.0, places=6)
        self.assertEqual(result["baselines"]["zero_difference"]["count"], 3)

    def test_gate_temperature_sweep_exposes_endpoint_smoothing(self):
        sweep = getattr(diagnostics, "gate_temperature_sweep")
        losses = torch.tensor(
            [
                [0., .01, .02, .03, .04],
                [.04, .03, .02, .01, 0.],
                [.02, .02, .02, .02, .02],
            ]
        )
        result = sweep(
            losses,
            [0., .25, .5, .75, 1.],
            temperatures=[.005, .05],
        )
        cold = result["0.005"]
        warm = result["0.05"]
        self.assertLess(cold["endpoint_target_mae"], warm["endpoint_target_mae"])
        self.assertLess(
            cold["normalized_entropy"]["mean"],
            warm["normalized_entropy"]["mean"],
        )
        self.assertAlmostEqual(cold["exact_tie_target_mean"], .5, places=6)
        self.assertAlmostEqual(warm["exact_tie_target_mean"], .5, places=6)

    def test_binary_gate_score_diagnostics_recovers_perfect_ranking(self):
        summarize = getattr(diagnostics, "binary_gate_score_diagnostics")
        scores = torch.tensor([.1, .2, .8, .9])
        binary_target = torch.tensor([0., 0., 1., 1.])
        candidate_losses = torch.tensor(
            [
                [.1, .2, .2, .2, .4],
                [.1, .2, .2, .2, .3],
                [.3, .2, .2, .2, .1],
                [.5, .2, .2, .2, .1],
            ]
        )
        endpoint_regret = (
            candidate_losses[:, -1] - candidate_losses[:, 0]
        ).abs()
        labels = torch.ones(4, 1)
        g0_logits = torch.tensor([[4.], [4.], [-4.], [-4.]])
        g1_logits = -g0_logits
        result = summarize(
            scores,
            g0_logits,
            g1_logits,
            labels,
            candidate_losses,
            binary_target,
            endpoint_regret,
        )
        self.assertAlmostEqual(result["roc_auc"], 1.0, places=6)
        self.assertAlmostEqual(
            result["endpoint_advantage_spearman"], 1.0, places=6
        )
        best = result["threshold_sweep"]["best_regret_weighted_accuracy"]
        self.assertAlmostEqual(best["regret_weighted_accuracy"], 1.0, places=6)
        self.assertAlmostEqual(best["hard_on_ratio"], .5, places=6)

    def test_binary_gate_score_diagnostics_marks_constant_scores(self):
        summarize = getattr(diagnostics, "binary_gate_score_diagnostics")
        scores = torch.full((4,), .6)
        binary_target = torch.tensor([0., 0., 1., 1.])
        candidate_losses = torch.tensor(
            [
                [.1, .2, .2, .2, .4],
                [.1, .2, .2, .2, .3],
                [.3, .2, .2, .2, .1],
                [.5, .2, .2, .2, .1],
            ]
        )
        endpoint_regret = (
            candidate_losses[:, -1] - candidate_losses[:, 0]
        ).abs()
        labels = torch.ones(4, 1)
        g0_logits = torch.tensor([[4.], [4.], [-4.], [-4.]])
        g1_logits = -g0_logits
        result = summarize(
            scores,
            g0_logits,
            g1_logits,
            labels,
            candidate_losses,
            binary_target,
            endpoint_regret,
        )
        self.assertAlmostEqual(result["roc_auc"], .5, places=6)
        self.assertIsNone(result["endpoint_advantage_pearson"])
        self.assertEqual(result["threshold_sweep"]["candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()
