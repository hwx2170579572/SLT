from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

from tools import run_topo_v4_5_1_experiments
from tools.checkpoint_selector_v4_4 import selection_key as selection_key_v4_4
from tools.checkpoint_selector_v4_5_1 import (
    select_checkpoint,
    selection_key,
    sha256,
    terminal_event_evidence,
)


def _summary(*, episodes: int, success: int, collision: int, off_route: int, timeout: int, mean_return: float = 0.0):
    return {
        "episodes": episodes,
        "mean_return": mean_return,
        "std_return": 0.0,
        "mean_decision_steps": 1.0,
        "mean_raw_steps": 3.0,
        "success_rate": success / episodes,
        "collision_rate": collision / episodes,
        "off_route_rate": off_route / episodes,
        "timeout_rate": timeout / episodes,
    }


def _record(episode: int, *, success: bool = False, collision: bool = False, off_route: bool = False, timeout: bool = False):
    return {
        "episode": episode,
        "seed": 100 + episode,
        "traffic_variant": f"traffic_{episode}.rou.xml",
        "success": success,
        "collision": collision,
        "off_route": off_route,
        "timeout": timeout,
    }


class SelectorV451Tests(unittest.TestCase):
    def _checkpoint(self, root: Path, name: str) -> Path:
        path = root / f"{name}.zip"
        with ZipFile(path, "w") as archive:
            archive.writestr("payload.txt", name)
        return path

    def _candidate(self, path: Path, kind: str, summary, records):
        return {
            "checkpoint_kind": kind,
            "checkpoint_path": str(path),
            "checkpoint_sha256": sha256(path),
            "traffic_partition": "train",
            "calibration_seed_start": 100,
            "summary": summary,
            "episode_records": records,
            "calibration_result_sha256": "a" * 64,
        }

    def test_success_and_timeout_overlap_is_preserved_and_selectable(self):
        records = [
            _record(0, success=True, timeout=True),
            _record(1, timeout=True),
        ]
        evidence = terminal_event_evidence(
            _summary(episodes=2, success=1, collision=0, off_route=0, timeout=2),
            records,
        )
        self.assertEqual(evidence["overlap_episode_count"], 1)
        self.assertEqual(evidence["overlap_episodes"][0]["active_flags"], ["success", "timeout"])
        self.assertEqual(evidence["event_flag_memberships"], 3)

    def test_selector_keeps_original_raw_flag_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = self._checkpoint(root, "best")
            final = self._checkpoint(root, "final")
            signature = [
                _record(0, success=True, timeout=True),
                _record(1, timeout=True),
            ]
            candidates = [
                self._candidate(
                    best,
                    "highest_training_success",
                    _summary(episodes=2, success=1, collision=0, off_route=0, timeout=2),
                    signature,
                ),
                self._candidate(
                    final,
                    "exact_final",
                    _summary(episodes=2, success=0, collision=0, off_route=0, timeout=2),
                    [
                        _record(0, timeout=True),
                        _record(1, timeout=True),
                    ],
                ),
            ]
            # Pairing covers episode/seed/traffic, not model-dependent outcomes.
            receipt = select_checkpoint(candidates)
            self.assertEqual(receipt["selected_checkpoint_kind"], "highest_training_success")
            self.assertEqual(receipt["terminal_event_semantics"], "source_flags_may_overlap")
            self.assertTrue(receipt["summary_and_episode_records_preserved"])
            self.assertFalse(receipt["event_precedence_added"])

    def test_summary_record_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "success summary count"):
            terminal_event_evidence(
                _summary(episodes=1, success=0, collision=0, off_route=0, timeout=1),
                [_record(0, success=True, timeout=True)],
            )

    def test_episode_without_terminal_flag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no terminal event flag"):
            terminal_event_evidence(
                _summary(episodes=1, success=0, collision=0, off_route=0, timeout=0),
                [_record(0)],
            )

    def test_exclusive_inputs_retain_v44_selection_key(self):
        summary = _summary(
            episodes=4, success=2, collision=1, off_route=0, timeout=1, mean_return=0.25
        )
        self.assertEqual(
            selection_key(summary, checkpoint_kind="exact_final"),
            selection_key_v4_4(summary, checkpoint_kind="exact_final"),
        )

    def test_orchestrator_wrapper_changes_only_trainer_path(self):
        with mock.patch.object(
            run_topo_v4_5_1_experiments,
            "_BASE_COMMAND_FOR",
            return_value=["python.exe", "old_trainer.py", "--scenario", "cross"],
        ):
            command = run_topo_v4_5_1_experiments.command_for(
                object(), {}, device="cuda"
            )
        self.assertEqual(command[0], "python.exe")
        self.assertEqual(Path(command[1]).name, "train_paper_sb3_sumo_v4_5_1.py")
        self.assertEqual(command[2:], ["--scenario", "cross"])

    def test_real_e1_calibration_boundary_selects_training_best(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "results_topo_v4_5_dev"
            / "development"
            / "runs"
            / "E1__cand__cross__s4__p3c1f7f8b"
        )
        if not root.is_dir():
            self.skipTest("real E1 calibration artifacts are unavailable")
        candidates = []
        for kind, short, filename in (
            ("highest_training_success", "best", "best_training_success_model.zip"),
            ("exact_final", "final", "final_model.zip"),
        ):
            detailed_path = root / "selector" / "cal" / short / "detailed.json"
            detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
            checkpoint = root / filename
            candidates.append(
                {
                    "checkpoint_kind": kind,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "traffic_partition": "train",
                    "calibration_seed_start": 68_000,
                    "summary": detailed["summary"],
                    "episode_records": detailed["episode_records"],
                    "calibration_result_sha256": sha256(detailed_path),
                }
            )
        receipt = select_checkpoint(candidates)
        self.assertEqual(receipt["selected_checkpoint_kind"], "highest_training_success")
        best = next(row for row in receipt["candidates"] if row["checkpoint_kind"] == "highest_training_success")
        overlap = best["terminal_event_evidence"]["overlap_episodes"]
        self.assertEqual(len(overlap), 1)
        self.assertEqual(overlap[0]["seed"], 68_002)


if __name__ == "__main__":
    unittest.main()
