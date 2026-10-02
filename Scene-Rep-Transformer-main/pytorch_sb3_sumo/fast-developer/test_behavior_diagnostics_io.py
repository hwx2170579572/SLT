"""Windows-safe summary publication regressions; no SUMO or training required."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


FAST_DEVELOPER = Path(__file__).resolve().parent
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

import behavior_diagnostics as behavior
from behavior_diagnostics import BehaviorDiagnosticsRecorder


def _actor(identity="ego", position=(0.0, 0.0)):
    return {
        "id": identity,
        "kind": "vehicle",
        "position": list(position),
        "velocity": [0.0, 0.0],
        "heading": 0.0,
        "speed": 0.0,
        "length": 4.0,
        "width": 2.0,
    }


def _begin_terminal_collision(recorder):
    ego = _actor()
    snapshot = {
        "raw_step": 1,
        "sim_time": 0.1,
        "ego": ego,
        "vehicles": [ego],
        "ego_state_source": "current",
        "observed_neighbor_ids": [],
        "events": {"collision": True},
    }
    recorder.on_reset(snapshot, info={"traffic_variant": "fixture"}, seed=17)
    recorder.on_decision_start([0.0, 0.0], {"target_speed": 0.0})
    recorder.on_raw_step(snapshot)
    recorder.on_decision_end(-1.0, True, False, {"collision": True})
    recorder.on_policy_step([0.0, 0.0], -1.0, True, False, {"collision": True})


class SummaryPublicationTests(unittest.TestCase):
    def _summary_paths(self, recorder):
        directory = recorder.path
        return directory / "summary.json", directory / "summary.json.tmp"

    def _patch_replace(self, recorder, replacement):
        path_type = type(recorder.path / "summary.json")
        return mock.patch.object(path_type, "replace", new=replacement)

    def test_transient_replace_denials_retry_and_publish_canonical_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = BehaviorDiagnosticsRecorder(directory, "train", "fixture")
            summary, _ = self._summary_paths(recorder)
            path_type = type(summary)
            real_replace = path_type.replace
            attempts = []

            def fail_twice_then_replace(source, target):
                if target.name == "summary.json":
                    attempts.append(target)
                    if len(attempts) <= 2:
                        raise PermissionError(13, "simulated transient lock")
                return real_replace(source, target)

            try:
                with self._patch_replace(recorder, fail_twice_then_replace), \
                        mock.patch.object(behavior.time, "sleep") as sleep:
                    self.assertTrue(recorder._json_file("summary.json", {"episodes_finished": 0}))

                self.assertEqual(len(attempts), 3)
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.05, 0.10])
                payload = json.loads(summary.read_text(encoding="utf-8"))
                self.assertEqual(payload["episodes_finished"], 0)
                self.assertEqual(payload["summary_write"], {
                    "pending": False,
                    "permission_denials": 2,
                    "deferred_flushes": 0,
                    "recovered_flushes": 0,
                    "last_error": None,
                })
                self.assertFalse(recorder.summary_write_pending)
                self.assertEqual(recorder.summary_permission_denials, 2)
                self.assertEqual(recorder.summary_deferred_flushes, 0)
                self.assertEqual(recorder.summary_recovered_flushes, 0)
                self.assertIsNone(recorder.summary_last_error)
            finally:
                recorder.close()

    def test_terminal_step_survives_persistent_replace_denial_and_later_flush_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = BehaviorDiagnosticsRecorder(directory, "train", "fixture")
            summary, temporary = self._summary_paths(recorder)
            recorder.flush()
            previous_canonical = summary.read_text(encoding="utf-8")
            path_type = type(summary)
            attempts = []

            def deny_summary_replace(source, target):
                if target.name == "summary.json":
                    attempts.append(target)
                    raise PermissionError(13, "simulated persistent lock")
                return path_type.replace(source, target)

            try:
                with self._patch_replace(recorder, deny_summary_replace), \
                        mock.patch.object(behavior.time, "sleep"), \
                        mock.patch.object(behavior.time, "monotonic", return_value=recorder.last_flush):
                    # The optional summary failure must not escape the real
                    # terminal policy-step path or discard the JSONL episode.
                    _begin_terminal_collision(recorder)

                self.assertEqual(len(attempts), 4)
                self.assertEqual(summary.read_text(encoding="utf-8"), previous_canonical)
                pending = json.loads(temporary.read_text(encoding="utf-8"))
                self.assertTrue(pending["summary_write"]["pending"])
                self.assertEqual(pending["episodes_finished"], 1)
                self.assertEqual(pending["outcome_counts"], {"collision": 1})
                self.assertEqual(pending["summary_write"]["permission_denials"], 4)
                self.assertEqual(recorder.summary_permission_denials, 4)
                self.assertEqual(recorder.summary_deferred_flushes, 1)
                self.assertTrue(recorder.summary_write_pending)
                self.assertIsNotNone(recorder.summary_last_error)
                episodes = [json.loads(row) for row in
                            (recorder.path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(episodes), 1)
                self.assertEqual(episodes[0]["outcome"], "collision")

                # Once the transient lock is gone, the next ordinary flush
                # publishes the pending snapshot and removes its sidecar.
                recorder.flush()
                recovered = json.loads(summary.read_text(encoding="utf-8"))
                self.assertEqual(recovered["episodes_finished"], 1)
                self.assertFalse(recovered["summary_write"]["pending"])
                self.assertEqual(recovered["summary_write"]["permission_denials"], 4)
                self.assertEqual(recovered["summary_write"]["deferred_flushes"], 1)
                self.assertEqual(recovered["summary_write"]["recovered_flushes"], 1)
                self.assertIsNone(recovered["summary_write"]["last_error"])
                self.assertFalse(temporary.exists())
                self.assertFalse(recorder.summary_write_pending)
                self.assertEqual(recorder.summary_recovered_flushes, 1)
                self.assertIsNone(recorder.summary_last_error)
            finally:
                recorder.close()

    def test_close_preserves_pending_snapshot_when_summary_remains_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = BehaviorDiagnosticsRecorder(directory, "train", "fixture")
            summary, temporary = self._summary_paths(recorder)
            recorder.flush()
            previous_canonical = summary.read_text(encoding="utf-8")
            recorder.on_reset({"ego": _actor()}, seed=18)
            path_type = type(summary)

            def deny_summary_replace(source, target):
                if target.name == "summary.json":
                    raise PermissionError(13, "simulated persistent lock")
                return path_type.replace(source, target)

            with self._patch_replace(recorder, deny_summary_replace), \
                    mock.patch.object(behavior.time, "sleep"), \
                    mock.patch.object(behavior.time, "monotonic", return_value=recorder.last_flush):
                recorder.close()

            self.assertTrue(recorder.closed)
            self.assertTrue(recorder.summary_write_pending)
            self.assertEqual(summary.read_text(encoding="utf-8"), previous_canonical)
            pending = json.loads(temporary.read_text(encoding="utf-8"))
            self.assertTrue(pending["summary_write"]["pending"])
            self.assertEqual(pending["episodes_finished"], 1)
            self.assertEqual(pending["episode_summaries"][0]["finish_reason"], "closed_before_terminal")
            episodes = [json.loads(row) for row in
                        (recorder.path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0]["finish_reason"], "closed_before_terminal")

    def test_manifest_permission_error_remains_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = BehaviorDiagnosticsRecorder(directory, "train", "fixture")
            manifest = recorder.path / "manifest.json"
            path_type = type(manifest)
            real_replace = path_type.replace

            def deny_manifest_replace(source, target):
                if target.name == "manifest.json":
                    raise PermissionError(13, "simulated metadata lock")
                return real_replace(source, target)

            try:
                with self._patch_replace(recorder, deny_manifest_replace):
                    with self.assertRaises(PermissionError):
                        recorder.write_run_metadata({"provenance": "must publish"})
            finally:
                recorder.close()


if __name__ == "__main__":
    unittest.main()
