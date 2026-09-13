from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

import torch
import torch.nn as nn

import run as run_module
import clip.modules.clip_missing_aware_prompt_module as model_module
import clip.modules.vision_transformer_prompts as prompt_module


class _FakePromptLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_adapter = nn.Linear(2, 2)
        self.prompt_token = nn.Parameter(torch.zeros(1))


class _FakeCounterfactualProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.adversary_strength = 0.0


class _FakeCustomCLIP(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model, config):
        super().__init__()
        self.image_feature_dim = 2
        self.text_feature_dim = 2
        self.prompt_learner = _FakePromptLearner()
        self.reliability_predictor = nn.Linear(2, 2)
        self.utility_gate = nn.Linear(2, 1)
        self.counterfactual_proxy = _FakeCounterfactualProxy()
        self.backbone = nn.Linear(2, 2)


def _config(reliability=False, counterfactual=False, task="mmimdb"):
    loss_names = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "mppd": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "mmimdb": 0,
        "hatememes": 0,
        "food101": 0,
    }
    if task is not None:
        loss_names[task] = 1
    return {
        "vit": "ViT-B/16",
        "prompt_length": 36,
        "prompt_depth": 6,
        "clip_cache_root": "",
        "reliability_enabled": reliability,
        "counterfactual_enabled": counterfactual,
        "loss_names": loss_names,
        "mmimdb_class_num": 23,
        "hatememes_class_num": 2,
        "food101_class_num": 101,
        "test_only": False,
        "load_path": "",
        "finetune_first": False,
        "original_dcp_path": "",
        "adapter_train_epochs": 10,
        "learning_rate": 1e-4,
        "weight_decay": .01,
        "counterfactual_adversary_strength": .25,
        "counterfactual_adversary_warmup_steps": 500,
    }


class TrainingModeTests(unittest.TestCase):
    def _build(self, config):
        with mock.patch.object(model_module, "load_clip_to_cpu", return_value=object()), mock.patch.object(
            model_module, "CustomCLIP", _FakeCustomCLIP
        ), mock.patch.object(model_module.clip_utils, "set_metrics"), mock.patch.object(
            model_module.CLIPransformerSS, "_load_original_dcp_base"
        ) as load_original:
            model = model_module.CLIPransformerSS(config)
        return model, load_original

    def test_training_mode_resolution_and_reliability_task_boundary(self):
        resolver = getattr(model_module, "training_mode_from_config")
        self.assertEqual(resolver(_config()), "dcp")
        self.assertEqual(resolver(_config(counterfactual=True)), "counterfactual")
        self.assertEqual(resolver(_config(reliability=True)), "reliability")
        with self.assertRaises(ValueError):
            resolver(_config(reliability=True, counterfactual=True))
        with self.assertRaises(ValueError):
            resolver(_config(reliability=True, task="food101"))

    def test_original_dcp_uses_standard_optimization_without_original_checkpoint(self):
        model, load_original = self._build(_config())
        self.assertEqual(model.training_mode, "dcp")
        self.assertTrue(model.automatic_optimization)
        load_original.assert_not_called()
        self.assertTrue(model.mmimdb_classifier.weight.requires_grad)
        self.assertTrue(model.model.prompt_learner.prompt_token.requires_grad)
        self.assertFalse(model.model.backbone.weight.requires_grad)
        self.assertFalse(model.model.counterfactual_proxy.weight.requires_grad)
        with mock.patch.object(
            model_module.clip_utils, "set_schedule", return_value="standard-schedule"
        ) as set_schedule:
            self.assertEqual(model.configure_optimizers(), "standard-schedule")
        set_schedule.assert_called_once_with(model)

    def test_counterfactual_uses_standard_optimization_without_original_checkpoint(self):
        model, load_original = self._build(_config(counterfactual=True))
        self.assertEqual(model.training_mode, "counterfactual")
        self.assertTrue(model.automatic_optimization)
        load_original.assert_not_called()
        self.assertTrue(model.model.counterfactual_proxy.weight.requires_grad)

    def test_reliability_keeps_manual_three_optimizer_schedule(self):
        model, load_original = self._build(_config(reliability=True))
        self.assertEqual(model.training_mode, "reliability")
        self.assertFalse(model.automatic_optimization)
        load_original.assert_called_once_with()
        optimizers = model.configure_optimizers()
        self.assertEqual(len(optimizers), 3)

    def test_reliability_optimizers_use_dedicated_hyperparameters(self):
        config = _config(reliability=True)
        config.update(
            reliability_predictor_lr=1e-3,
            reliability_adapter_lr=2e-3,
            reliability_gate_lr=3e-3,
            reliability_weight_decay=1e-4,
        )
        model, _ = self._build(config)
        optimizers = model.configure_optimizers()
        self.assertEqual(
            [optimizer.param_groups[0]["lr"] for optimizer in optimizers],
            [1e-3, 2e-3, 3e-3],
        )
        self.assertEqual(
            [optimizer.param_groups[0]["weight_decay"] for optimizer in optimizers],
            [1e-4, 1e-4, 1e-4],
        )

    def test_reliability_phase_freezes_everything_except_active_components(self):
        model, _ = self._build(_config(reliability=True))
        adapter_trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(
            any(name.startswith("model.reliability_predictor.") for name in adapter_trainable)
        )
        self.assertTrue(
            any(name.startswith("model.prompt_learner.prompt_adapter.") for name in adapter_trainable)
        )
        self.assertFalse(
            any(name.startswith("model.utility_gate.") for name in adapter_trainable)
        )
        self.assertFalse(
            any(name.startswith("mmimdb_classifier.") for name in adapter_trainable)
        )

        model._set_training_phase("gate")
        gate_trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(gate_trainable)
        self.assertTrue(
            all(name.startswith("model.utility_gate.") for name in gate_trainable)
        )

    def test_direct_task_gate_uses_one_adapted_forward_and_only_gate_gradients(self):
        class TinyUtilityGate(nn.Module):
            def __init__(self):
                super().__init__()
                self.logit = nn.Parameter(torch.tensor(0.0))

            def logits(self, reliability, availability, base_context):
                return self.logit.expand(reliability.shape[0], 1)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.utility_gate = TinyUtilityGate()
                self.frozen_offset = nn.Parameter(
                    torch.tensor(1.0), requires_grad=False
                )
                self.encode_calls = 0

            def prepare_reliability_state(
                self, image, text, missing_type, return_adapter_diagnostics=False
            ):
                batch_size = image.shape[0]
                return {
                    "reliability": torch.full((batch_size, 2), 0.5),
                    "availability_mask": torch.ones(batch_size, 2),
                    "base_cls_feats": torch.zeros(batch_size, 1),
                    "adapter_diagnostics": [],
                }

            def encode_reliability_state(self, state, gate):
                self.encode_calls += 1
                return (
                    state["base_cls_feats"]
                    + gate.reshape(-1, 1) * self.frozen_offset
                )

        tiny_model = TinyModel()
        classifier = nn.Linear(1, 1, bias=False)
        classifier.weight.data.fill_(1.0)
        classifier.weight.requires_grad_(False)
        metrics = {}
        trainer_stub = SimpleNamespace(
            training=True,
            gate_supervision_mode="direct_task",
            model=tiny_model,
            mmimdb_classifier=classifier,
            _gate_epoch_predictions=[],
            _gate_epoch_targets=[],
            _gate_epoch_hard_targets=[],
            _phase_for_epoch=lambda: "gate",
            _labels=lambda batch, device: batch["labels"].to(device),
            _log_adapter_diagnostics=lambda diagnostics: None,
            _buffer_epoch_metric=lambda name, value: metrics.setdefault(
                name, []
            ).append(value),
            log=lambda *args, **kwargs: None,
        )
        batch = {
            "image": [torch.zeros(2, 1)],
            "text": (["a", "b"],),
            "missing_type": [0, 0],
            "labels": torch.ones(2, 1),
        }

        loss = model_module.CLIPransformerSS._compute_gate_loss(
            trainer_stub, batch
        )
        self.assertEqual(tiny_model.encode_calls, 1)
        self.assertEqual(trainer_stub._gate_epoch_targets, [])
        self.assertEqual(trainer_stub._gate_epoch_hard_targets, [])
        loss.backward()
        self.assertIsNotNone(tiny_model.utility_gate.logit.grad)
        self.assertNotEqual(tiny_model.utility_gate.logit.grad.item(), 0.0)
        self.assertIsNone(tiny_model.frozen_offset.grad)
        self.assertIsNone(classifier.weight.grad)
        self.assertIn("gate_loss", metrics)

    def test_standard_training_step_returns_task_loss(self):
        model, _ = self._build(_config())
        expected = torch.tensor(2.5, requires_grad=True)
        model.forward = mock.Mock(
            return_value={"mmimdb_loss": expected, "mmimdb_logits": torch.zeros(1)}
        )
        with mock.patch.object(model_module.clip_utils, "set_task"):
            actual = model.training_step({}, 0)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(actual.requires_grad)

    def test_checkpoint_policy_is_mode_specific(self):
        builder = getattr(run_module, "build_checkpoint_callback")
        with mock.patch.object(run_module.pl.callbacks, "ModelCheckpoint") as standard, mock.patch.object(
            run_module, "GatePhaseModelCheckpoint"
        ) as gate:
            standard.return_value = "standard"
            gate.return_value = "gate"
            self.assertEqual(builder(_config(), Path("checkpoints")), "standard")
            standard.assert_called_once()
            gate.assert_not_called()
        with mock.patch.object(run_module.pl.callbacks, "ModelCheckpoint") as standard, mock.patch.object(
            run_module, "GatePhaseModelCheckpoint"
        ) as gate:
            standard.return_value = "standard"
            gate.return_value = "gate"
            self.assertEqual(
                builder(_config(reliability=True), Path("checkpoints")), "gate"
            )
            gate.assert_called_once()
            standard.assert_not_called()


class PromptConfigurationTests(unittest.TestCase):
    def test_current_prompt_configuration_is_valid(self):
        validator = getattr(prompt_module, "validate_prompt_configuration")
        self.assertEqual(validator(36, 6, 12, 12), (36, 6))

    def test_invalid_prompt_configuration_fails_explicitly(self):
        validator = getattr(prompt_module, "validate_prompt_configuration")
        invalid = [
            (0, 6, 12, 12),
            (35, 6, 12, 12),
            (None, 6, 12, 12),
            (36.0, 6, 12, 12),
            (36, 0, 12, 12),
            (36, 13, 12, 12),
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validator(*arguments)


class ZeroGateEquivalenceTests(unittest.TestCase):
    @staticmethod
    def _state():
        return {
            "image_prompts": [torch.ones(2, 1, 2)],
            "text_prompts": [torch.ones(2, 1, 2)],
            "image_offsets": [torch.full((2, 1, 2), 3.0)],
            "text_offsets": [torch.full((2, 1, 2), 4.0)],
            "image": torch.zeros(2, 1),
            "tokenized_texts": torch.zeros(2, 1, dtype=torch.long),
            "missing_type": [0, 0],
            "availability_mask": torch.ones(2, 2),
            "base_cls_feats": torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
            ),
        }

    @staticmethod
    def _fake_model():
        encoded_image = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
        encoded_text = torch.tensor([[13.0, 14.0], [15.0, 16.0]])
        return SimpleNamespace(
            _encode_from_prompts=mock.Mock(
                return_value=(encoded_image, encoded_text)
            ),
            _mask_features=mock.Mock(
                side_effect=lambda image, text, availability: (image, text)
            ),
        )

    def test_all_zero_gate_returns_existing_base_without_reencoding(self):
        fake = self._fake_model()
        state = self._state()
        actual = model_module.CustomCLIP.encode_reliability_state(
            fake, state, torch.zeros(2, 1)
        )
        self.assertTrue(torch.equal(actual, state["base_cls_feats"]))
        fake._encode_from_prompts.assert_not_called()

    def test_mixed_gate_preserves_only_zero_gate_rows(self):
        fake = self._fake_model()
        state = self._state()
        actual = model_module.CustomCLIP.encode_reliability_state(
            fake, state, torch.tensor([[0.0], [1.0]])
        )
        expected = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 15.0, 16.0]]
        )
        self.assertTrue(torch.equal(actual, expected))
        fake._encode_from_prompts.assert_called_once()


if __name__ == "__main__":
    unittest.main()
