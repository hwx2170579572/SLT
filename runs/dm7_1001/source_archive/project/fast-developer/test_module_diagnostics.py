"""Telemetry must preserve the computation and distinguish absent/zero gradients."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algos.sb3_torch.module_diagnostics import collect_parameter_gradient_diagnostics
from algos.sb3_torch.sac import SceneRepresentationSAC
from behavior_diagnostics import BehaviorDiagnosticsRecorder


class ParameterGradientDiagnosticsTests(unittest.TestCase):
    def test_exact_norms_and_no_parameter_gradient_or_rng_mutation(self):
        module = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            module.weight.copy_(torch.tensor([[3.0, 4.0]]))
        module.weight.grad = torch.tensor([[6.0, 8.0]])
        before_parameter = module.weight.detach().clone()
        before_gradient = module.weight.grad.clone()
        before_rng = torch.random.get_rng_state().clone()
        result = collect_parameter_gradient_diagnostics(module, {"live": ("weight",)})
        self.assertEqual(result["encoder_grad/live/parameter_l2"], 5.0)
        self.assertEqual(result["encoder_grad/live/gradient_l2"], 10.0)
        self.assertEqual(result["encoder_grad/live/gradient_to_parameter_l2"], 2.0)
        self.assertEqual(result["encoder_grad/live/gradient_nonzero_elements"], 2)
        self.assertTrue(torch.equal(before_parameter, module.weight))
        self.assertTrue(torch.equal(before_gradient, module.weight.grad))
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))

    def test_missing_zero_nonfinite_and_unknown_are_distinct(self):
        module = nn.Linear(2, 1, bias=False)
        missing = collect_parameter_gradient_diagnostics(module, {"all": ("",), "unknown": ("absent",)})
        self.assertIsNone(missing["encoder_grad/all/gradient_l2"])
        self.assertEqual(missing["encoder_grad/all/missing_gradient_elements"], 2)
        self.assertEqual(missing["encoder_grad/unknown/parameter_elements"], 0)
        module.weight.grad = torch.zeros_like(module.weight)
        zero = collect_parameter_gradient_diagnostics(module, {"all": ("",)})
        self.assertEqual(zero["encoder_grad/all/gradient_l2"], 0.0)
        self.assertEqual(zero["encoder_grad/all/gradient_parameter_fraction"], 1.0)
        module.weight.grad = torch.tensor([[float("nan"), float("inf")]])
        invalid = collect_parameter_gradient_diagnostics(module, {"all": ("",)})
        self.assertEqual(invalid["encoder_grad/all/gradient_nonfinite_elements"], 2)
        self.assertIsNone(invalid["encoder_grad/all/gradient_l2"])

    def test_prefix_boundary_excludes_similarly_named_modules(self):
        module = nn.Module()
        module.slot = nn.Linear(2, 1, bias=False)
        module.slot_extra = nn.Linear(2, 1, bias=False)
        module.slot.weight.grad = torch.zeros_like(module.slot.weight)
        module.slot_extra.weight.grad = torch.full_like(module.slot_extra.weight, 100.0)
        result = collect_parameter_gradient_diagnostics(module, {"slot": ("slot",)})
        self.assertEqual(result["encoder_grad/slot/parameter_elements"], 2)
        self.assertEqual(result["encoder_grad/slot/gradient_l2"], 0.0)


class DiagnosticEventTests(unittest.TestCase):
    def test_events_survive_logger_dump_and_drain_once(self):
        model = object.__new__(SceneRepresentationSAC)
        model.num_timesteps = 11
        model._raw_steps_seen = 33
        metrics = {"norm": 1.0, "missing": None}
        model._queue_module_diagnostic_event("critic_td_gradient_pre_clip", 5, metrics)
        metrics["norm"] = 999.0
        events = model.pop_module_diagnostic_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["metrics"]["norm"], 1.0)
        self.assertEqual(events[0]["raw_steps"], 33)
        self.assertEqual(events[0]["updates"], 5)
        self.assertEqual(model.pop_module_diagnostic_events(), [])


class RepresentationRecorderTests(unittest.TestCase):
    def test_sources_missingness_and_episode_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = BehaviorDiagnosticsRecorder(tmp, "train", "test")
            recorder.episode_index = 7
            recorder.record_representation(30, 10, {"norm": 2.0, "missing": None,
                                                    "invalid": float("nan")},
                                           "critic_td_gradient_pre_clip", updates=5)
            recorder.record_representation(33, 11, {"norm": 4.0},
                                           "eval_policy_forward", episode=7)
            recorder.close()
            path = Path(tmp) / "diagnostics" / "train"
            rows = [json.loads(line) for line in (path / "representation.jsonl").read_text().splitlines()]
            self.assertIsNone(rows[0]["episode_index"])
            self.assertEqual(rows[0]["updates"], 5)
            self.assertIsNone(rows[0]["metrics"]["invalid"])
            self.assertEqual(rows[1]["episode_index"], 7)
            summary = json.loads((path / "summary.json").read_text())
            self.assertEqual(summary["representation_samples"], 2)
            stats = summary["representation_statistics_by_source"]
            self.assertEqual(stats["critic_td_gradient_pre_clip"]["norm"]["sum"], 2.0)
            self.assertEqual(stats["critic_td_gradient_pre_clip"]["missing"]["missing"], 1)
            self.assertEqual(stats["eval_policy_forward"]["norm"]["sum"], 4.0)

    def test_existing_representation_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics" / "eval"
            path.mkdir(parents=True)
            artifact = path / "representation.jsonl"
            artifact.write_text("preserve\n")
            with self.assertRaises(FileExistsError):
                BehaviorDiagnosticsRecorder(tmp, "eval", "test")
            self.assertEqual(artifact.read_text(), "preserve\n")


if __name__ == "__main__":
    unittest.main()
