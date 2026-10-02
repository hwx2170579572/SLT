"""Bounded Windows replace-retry behavior for training JSON artifacts."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


FAST_DEVELOPER = Path(__file__).resolve().parent
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

import train_intersection_yield_v2 as trainer


class AtomicJsonWriteTests(unittest.TestCase):
    def test_transient_permission_denials_retry_then_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            real_replace = trainer.os.replace
            attempts = []

            def fail_twice_then_replace(source, target):
                attempts.append((source, target))
                if len(attempts) <= 2:
                    raise PermissionError(13, "simulated transient lock")
                return real_replace(source, target)

            with mock.patch.object(trainer.os, "replace", side_effect=fail_twice_then_replace), \
                    mock.patch.object(trainer.time, "sleep") as sleep:
                self.assertTrue(trainer._write_json_atomic(path, {"status": "training"}))

            self.assertEqual(len(attempts), 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.05, 0.10])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "training"})
            self.assertFalse(attempts[-1][0].exists())

    def test_persistent_progress_denial_keeps_canonical_and_full_temp_then_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            temporary = path.with_name(f".{trainer.os.getpid()}.tmp")
            self.assertTrue(trainer._write_json_atomic(path, {"raw_steps": 100}))
            canonical_before = path.read_text(encoding="utf-8")
            payload = {"raw_steps": 200, "updates": 50}

            with mock.patch.object(trainer.os, "replace", side_effect=PermissionError(13, "persistent lock")) as replace, \
                    mock.patch.object(trainer.time, "sleep") as sleep, \
                    self.assertLogs(trainer.__name__, level="WARNING") as logs:
                self.assertFalse(trainer._write_json_atomic(path, payload))

            self.assertEqual(replace.call_count, 4)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.05, 0.10, 0.20])
            self.assertEqual(path.read_text(encoding="utf-8"), canonical_before)
            self.assertEqual(json.loads(temporary.read_text(encoding="utf-8")), payload)
            self.assertIn("4 attempts", " ".join(logs.output))

            self.assertTrue(trainer._write_json_atomic(path, payload))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
            self.assertFalse(temporary.exists())

    def test_persistent_status_denial_remains_fatal_and_preserves_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            temporary = path.with_name(f".{trainer.os.getpid()}.tmp")
            payload = {"status": "failed", "raw_steps": 250}

            with mock.patch.object(trainer.os, "replace", side_effect=PermissionError(13, "persistent lock")) as replace, \
                    mock.patch.object(trainer.time, "sleep") as sleep:
                with self.assertRaises(PermissionError):
                    trainer._write_json_atomic(path, payload)

            self.assertEqual(replace.call_count, 4)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.05, 0.10, 0.20])
            self.assertFalse(path.exists())
            self.assertEqual(json.loads(temporary.read_text(encoding="utf-8")), payload)

    def test_non_permission_oserror_is_not_retried_or_swallowed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            with mock.patch.object(trainer.os, "replace", side_effect=OSError(28, "disk full")) as replace, \
                    mock.patch.object(trainer.time, "sleep") as sleep:
                with self.assertRaises(OSError):
                    trainer._write_json_atomic(path, {"raw_steps": 1})

            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
