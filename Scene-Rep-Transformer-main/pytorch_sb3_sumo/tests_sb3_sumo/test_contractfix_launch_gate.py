from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAST_DEVELOPER = PROJECT_ROOT / "fast-developer"
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

import launch_sorted_contractfix_20261003 as launcher


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


class ContractfixLaunchGateFixtureTests(unittest.TestCase):
    """Exercise only the launcher's file-based predecessor gate; never launch."""

    def _check_gate(self, mutate=None) -> dict:
        with tempfile.TemporaryDirectory(prefix="contractfix_gate_") as temporary:
            base = Path(temporary)
            roots = {
                method: base / "predecessors" / method
                for method in launcher.OLD_ARMS
            }
            run_dirs = {
                method: roots[method] / launcher.OLD_METHOD_DIRS[method]
                for method in roots
            }
            output_root = base / "new_contractfix_run"
            receipt = base / "global_pair_receipt.json"
            checkpoint_payload = b"fixture checkpoint bytes; not a model"
            checkpoint_sha = hashlib.sha256(checkpoint_payload).hexdigest()
            methods = tuple(roots)
            commands = [
                [
                    "python.exe", "trainer.py", "--method", method,
                    "--max-steps", "100000", "--checkpoint-frequency", "10000",
                ]
                for method in methods
            ]
            suite = {
                "python": str(Path(sys.executable).resolve()),
                "fresh": True,
                "resume": False,
                "training_seed": 0,
                "raw_step_budget_per_method": 100000,
                "smoke": False,
                "evaluation": {
                    "split": "validation",
                    "episodes": 100,
                    "seed_start": 10000,
                    "seed_end": 10099,
                    "checkpoint": "final_model.zip",
                },
                "commands": commands,
            }

            for method, run_dir in run_dirs.items():
                run_dir.mkdir(parents=True)
                checkpoint = run_dir / "final_model.zip"
                checkpoint.write_bytes(checkpoint_payload)
                _write_json(run_dir / "status.json", {"status": "trained"})
                _write_json(
                    run_dir / "training_complete.json",
                    {
                        "raw_steps": 100000,
                        "smoke": False,
                        "checkpoint_sha256": checkpoint_sha,
                    },
                )
                _write_json(
                    run_dir / "arguments.json",
                    {
                        "method": method,
                        "scenario": "intersection_sorted",
                        "raw_budget": 100000,
                        "seed": 0,
                        "smoke": False,
                        "resume": False,
                    },
                )
                _write_json(
                    run_dir / "evaluation_results.json",
                    {
                        "identity": {
                            "method": method,
                            "scenario": "intersection_sorted",
                            "depart_scale": 4.0,
                            "eval_traffic_split": "validation",
                            "episodes": 100,
                            "smoke": False,
                            "checkpoint_sha256": checkpoint_sha,
                            "checkpoint": str(checkpoint.resolve()),
                        },
                        "episode_records": [
                            {"seed": seed} for seed in range(10000, 10100)
                        ],
                        "summary": {"episodes": 100},
                    },
                )
                _write_json(roots[method] / "suite_manifest.json", suite)

            fixture = {
                "roots": roots,
                "run_dirs": run_dirs,
                "output_root": output_root,
                "receipt": receipt,
            }
            if mutate is not None:
                mutate(fixture)

            with (
                patch.object(launcher, "OLD_ARMS", roots),
                patch.object(launcher, "OLD_METHOD_DIRS", launcher.OLD_METHOD_DIRS),
                patch.object(launcher, "PAIR_RECEIPT", receipt),
            ):
                return launcher.check_preconditions(output_root)

    def test_complete_fresh_pair_with_matching_identity_is_ready(self):
        gate = self._check_gate()
        self.assertTrue(gate["ready"], gate["blockers"])
        self.assertTrue(all(arm["ready"] for arm in gate["valid_predecessor_arms"]))

    def test_missing_final_evaluation_blocks(self):
        def mutate(fixture):
            method = next(iter(fixture["run_dirs"]))
            (fixture["run_dirs"][method] / "evaluation_results.json").unlink()

        gate = self._check_gate(mutate)
        self.assertFalse(gate["ready"])
        self.assertIn("missing_evaluation_results", gate["valid_predecessor_arms"][0]["reasons"])

    def test_checkpoint_sha_mismatch_blocks(self):
        def mutate(fixture):
            method = next(iter(fixture["run_dirs"]))
            path = fixture["run_dirs"][method] / "evaluation_results.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            result["identity"]["checkpoint_sha256"] = "0" * 64
            _write_json(path, result)

        gate = self._check_gate(mutate)
        self.assertFalse(gate["ready"])
        self.assertIn(
            "evaluation_checkpoint_sha_mismatch",
            gate["valid_predecessor_arms"][0]["reasons"],
        )

    def test_wrong_evaluation_seed_blocks(self):
        def mutate(fixture):
            method = next(iter(fixture["run_dirs"]))
            path = fixture["run_dirs"][method] / "evaluation_results.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            result["episode_records"][0]["seed"] = 10001
            _write_json(path, result)

        gate = self._check_gate(mutate)
        self.assertFalse(gate["ready"])
        self.assertIn(
            "evaluation_seed_records_not_10000_10099",
            gate["valid_predecessor_arms"][0]["reasons"],
        )

    def test_resume_argument_blocks(self):
        def mutate(fixture):
            method = next(iter(fixture["run_dirs"]))
            path = fixture["run_dirs"][method] / "arguments.json"
            arguments = json.loads(path.read_text(encoding="utf-8"))
            arguments["resume"] = True
            _write_json(path, arguments)

        gate = self._check_gate(mutate)
        self.assertFalse(gate["ready"])
        self.assertTrue(
            any(
                reason.startswith("arguments_indicate_resume_or_checkpoint_input")
                for reason in gate["valid_predecessor_arms"][0]["reasons"]
            )
        )

    def test_missing_training_completion_blocks(self):
        def mutate(fixture):
            method = next(iter(fixture["run_dirs"]))
            (fixture["run_dirs"][method] / "training_complete.json").unlink()

        gate = self._check_gate(mutate)
        self.assertFalse(gate["ready"])
        self.assertIn("missing_training_complete", gate["valid_predecessor_arms"][0]["reasons"])

    def test_existing_output_root_blocks(self):
        def mutate(fixture):
            fixture["output_root"].mkdir()

        gate = self._check_gate(mutate)
        self.assertFalse(gate["ready"])
        self.assertIn("requested_new_run_root_already_exists", gate["blockers"])

    def test_existing_global_pair_receipt_blocks(self):
        def mutate(fixture):
            _write_json(fixture["receipt"], {"status": "reserved", "run_root": "fixture"})

        gate = self._check_gate(mutate)
        self.assertFalse(gate["ready"])
        self.assertIn("contractfix_pair_already_reserved_or_launched", gate["blockers"])

    def test_workers_use_distinct_short_run_and_traffic_roots(self):
        with tempfile.TemporaryDirectory(prefix="contractfix_worker_roots_") as temporary:
            pair_root = Path(temporary) / "pair"
            with patch.object(launcher, "_runner_python", return_value=Path(sys.executable)):
                specs = launcher.worker_specs(pair_root)
            self.assertEqual([spec["method"] for spec in specs], list(launcher.METHODS))
            run_roots = [Path(spec["run_root"]) for spec in specs]
            self.assertEqual(
                run_roots,
                [pair_root.resolve() / "st", pair_root.resolve() / "st_rt"],
            )
            self.assertEqual(len(set(run_roots)), 2)

            scaled_roots = [Path(spec["scaled_traffic_root"]) for spec in specs]
            overlay_roots = [Path(spec["traffic_overlay_root"]) for spec in specs]
            self.assertEqual(len(set(scaled_roots)), 2)
            self.assertEqual(len(set(overlay_roots)), 2)
            for spec, run_root in zip(specs, run_roots):
                argv = spec["argv"]
                command_root = Path(argv[argv.index("--run-root") + 1])
                self.assertEqual(command_root, run_root)
                self.assertEqual(Path(spec["authorization_marker_value"]), command_root)
                self.assertTrue(Path(spec["scaled_traffic_root"]).is_relative_to(run_root))
                self.assertTrue(Path(spec["traffic_overlay_root"]).is_relative_to(run_root))
                self.assertTrue(
                    Path(spec["train_traffic_overlay_root"]).is_relative_to(run_root)
                )
                self.assertTrue(
                    Path(spec["eval_traffic_overlay_root"]).is_relative_to(run_root)
                )


if __name__ == "__main__":
    unittest.main()
