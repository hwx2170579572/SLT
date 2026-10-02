import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from tools.recover_interrupted_paper_evaluation import validate_completed_training


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_model(path: Path, raw_steps: int) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "data",
            json.dumps({"_raw_steps_seen": raw_steps, "num_timesteps": raw_steps // 3}),
        )


class InterruptedEvaluationRecoveryTest(unittest.TestCase):
    def _completed_training(self, root: Path) -> Path:
        run = root / "paper__proposed__cross__seed2"
        checkpoints = run / "checkpoints"
        checkpoints.mkdir(parents=True)
        requested = {
            "scenario": "cross",
            "algo": "scene_rep",
            "max_steps": 40,
            "seed": 2,
            "device": "cpu",
            "discount": 0.99,
            "neighbors": 5,
            "history_steps": 10,
            "path_length": 10,
            "action_repeat": 3,
            "checkpoint_freq": 20,
            "eval_episodes": 2,
        }
        (run / "arguments.json").write_text(
            json.dumps({"requested_raw_steps": requested}), encoding="utf-8"
        )
        rows = []
        for step in (20, 40):
            model = checkpoints / f"scene_rep_raw_{step}_steps.zip"
            _write_model(model, step)
            rows.append(
                {
                    "raw_step": step,
                    "path": str(model.relative_to(run)),
                    "clock_field": "_raw_steps_seen",
                    "expected_clock": step,
                    "recorded_clock": step,
                    "zip_crc_ok": True,
                    "bytes": model.stat().st_size,
                    "sha256": _sha256(model),
                }
            )
        _write_model(run / "final_model.zip", 40)
        audit = {
            "algorithm": "scene_rep",
            "scenario": "cross",
            "requested_raw_steps": 40,
            "checkpoint_frequency_raw_steps": 20,
            "clock_field": "_raw_steps_seen",
            "expected_checkpoint_count": 2,
            "checkpoints": rows,
        }
        (run / "checkpoint_audit.json").write_text(json.dumps(audit), encoding="utf-8")
        return run

    def test_accepts_only_fully_audited_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = self._completed_training(Path(temporary))
            result = validate_completed_training(run)
            self.assertEqual(result["arguments"]["max_steps"], 40)

    def test_rejects_tampered_recorded_clock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = self._completed_training(Path(temporary))
            audit_path = run / "checkpoint_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["checkpoints"][1]["recorded_clock"] = 41
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact source clock"):
                validate_completed_training(run)


if __name__ == "__main__":
    unittest.main()
