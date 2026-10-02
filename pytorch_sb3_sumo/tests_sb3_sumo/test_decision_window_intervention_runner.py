from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


FD = Path(__file__).resolve().parents[1] / "fast-developer"
if str(FD) not in sys.path:
    sys.path.insert(0, str(FD))

import diagnose_strt_decision_windows_20261003 as runner


class DecisionWindowRunnerTests(unittest.TestCase):
    def test_stable_hash_ignores_mapping_insertion_order(self):
        self.assertEqual(
            runner.stable_hash({"b": 2, "a": 1}),
            runner.stable_hash({"a": 1, "b": 2}),
        )

    def test_physical_snapshot_excludes_hidden_route_fields_and_sorts_ids(self):
        left = {
            "raw_step": 7,
            "sim_time": 0.7,
            "ego_state_source": "current",
            "ego": {"id": "ego", "key": "vehicle:ego", "position": [1.0, 2.0], "route_id": "secret"},
            "vehicles": [
                {"id": "b", "key": "vehicle:b", "position": [3.0, 4.0], "route_id": "hidden-b"},
                {"id": "a", "key": "vehicle:a", "position": [5.0, 6.0], "route_id": "hidden-a"},
            ],
        }
        right = {
            **left,
            "ego": {**left["ego"], "route_id": "different-secret"},
            "vehicles": [
                {**left["vehicles"][1], "route_id": "other-a"},
                {**left["vehicles"][0], "route_id": "other-b"},
            ],
        }
        a = runner.physical_snapshot_payload(left)
        b = runner.physical_snapshot_payload(right)
        self.assertEqual(a, b)
        self.assertEqual([row["id"] for row in a["vehicles"]], ["a", "b"])
        self.assertNotIn("route_id", a["ego"])

    def test_case_selection_uses_ascending_seed_with_separate_outcome_strata(self):
        rows = []
        for seed in range(10000, 10020):
            rows.append({
                "seed": seed,
                "collision": seed % 2 == 0,
                "success": seed % 2 == 1,
                "raw_steps": seed - 9990,
                "decision_steps": seed - 9995,
            })
        chosen = runner.choose_cases(rows)
        self.assertEqual([x["seed"] for x in chosen[:8]], list(range(10000, 10016, 2)))
        self.assertEqual([x["seed"] for x in chosen[8:]], list(range(10001, 10009, 2)))
        self.assertEqual([x["outcome"] for x in chosen], ["collision"] * 8 + ["success"] * 4)

    def test_anchor_plan_merges_overlapping_phases_and_marks_unavailable_internal(self):
        trace = [
            {"pre_ego_road_id": "-E1", "pre_obs_raw_step": 1},
            {"pre_ego_road_id": ":J1_1", "pre_obs_raw_step": 4},
            {"pre_ego_road_id": "-E0", "pre_obs_raw_step": 7},
        ]
        plan = runner.plan_anchors(trace)
        self.assertEqual(plan["phases"]["three_decisions_before_first_internal"], 0)
        self.assertEqual(plan["phases"]["first_internal_pre_action"], 1)
        self.assertEqual(plan["phases"]["three_decisions_before_terminal"], 0)
        self.assertEqual(plan["unique"][0]["decision_index"], 0)
        self.assertEqual(
            set(plan["unique"][0]["phases"]),
            {"three_decisions_before_first_internal", "three_decisions_before_terminal"},
        )

        no_internal = runner.plan_anchors([
            {"pre_ego_road_id": "-E1"},
            {"pre_ego_road_id": "-E0"},
        ])
        self.assertIsNone(no_internal["phases"]["first_internal_pre_action"])

    def test_speed_action_mapping_and_lateral_channel_preservation(self):
        import numpy as np

        self.assertEqual(runner.action_for_target_speed(0), -1.0)
        self.assertAlmostEqual(runner.action_for_target_speed(6), 0.2)
        self.assertEqual(runner.action_for_target_speed(10), 1.0)
        source = np.asarray([0.37, -0.62], dtype=np.float32)
        changed = runner.replace_speed_action(source, 6.0)
        self.assertAlmostEqual(float(changed[0]), 0.2, places=6)
        self.assertEqual(float(changed[1]), float(source[1]))
        self.assertAlmostEqual(float(source[0]), 0.37, places=6)
        with self.assertRaises(ValueError):
            runner.action_for_target_speed(10.01)

    def test_prefix_match_requires_observation_action_and_anchor_physical_identity(self):
        baseline = [{
            "observation_sha256": "obs",
            "policy_action_sha256": "act",
            "physical_state_sha256": "physical",
            "physical_state_complete": True,
        }]
        self.assertIsNone(runner.prefix_mismatch(baseline, [dict(baseline[0])], 0))
        self.assertEqual(
            runner.prefix_mismatch(baseline, [{**baseline[0], "observation_sha256": "other"}], 0)["reason"],
            "observation_sha256_mismatch",
        )
        self.assertEqual(
                runner.prefix_mismatch(baseline, [{**baseline[0], "physical_state_sha256": None}], 0)["reason"],
                "prefix_physical_state_missing",
        )
        self.assertEqual(
            runner.prefix_mismatch(baseline, [], 0)["reason"],
            "prefix_length_mismatch",
        )

    def test_prefix_gate_rejects_incomplete_full_physical_state_even_if_hashes_match(self):
        baseline = [{
            "observation_sha256": "same",
            "policy_action_sha256": "same-action",
            "physical_state_sha256": "same-hash",
            "physical_state_complete": False,
        }]
        result = runner.prefix_mismatch(baseline, [dict(baseline[0])], 0)
        self.assertEqual(result["reason"], "prefix_full_physical_state_unavailable")

    def test_prefix_gate_checks_physical_state_at_each_pre_anchor_decision(self):
        baseline = [
            {
                "observation_sha256": f"obs-{i}",
                "policy_action_sha256": f"act-{i}",
                "physical_state_sha256": f"physical-{i}",
                "physical_state_complete": True,
                "physical_state": {"ego": {"position": [float(i), 0.0]}, "vehicles": []},
            }
            for i in range(2)
        ]
        replay = [dict(row) for row in baseline]
        replay[0]["physical_state_sha256"] = "different-at-prefix"
        replay[0]["physical_state"] = {"ego": {"position": [99.0, 0.0]}, "vehicles": []}
        mismatch = runner.prefix_mismatch(baseline, replay, anchor_index=1)
        self.assertEqual(mismatch["reason"], "prefix_physical_state_mismatch")
        self.assertEqual(mismatch["decision_index"], 0)
        self.assertEqual(mismatch["physical_state_diff"]["changed_actors_preview"][0]["fields"], ["position"])

    def test_prefix_gate_does_not_accept_two_missing_physical_hashes(self):
        rows = [{
            "observation_sha256": "obs",
            "policy_action_sha256": "action",
            "physical_state_sha256": None,
            "physical_state_complete": True,
        }]
        mismatch = runner.prefix_mismatch(rows, [dict(rows[0])], anchor_index=0)
        self.assertEqual(mismatch["reason"], "prefix_physical_state_missing")

    def test_raw_snapshot_observer_captures_all_actors_before_diagnostic_cap(self):
        from types import SimpleNamespace

        full = {
            "raw_step": 9, "sim_time": 0.9,
            "ego_state_source": "current", "ego_state_time": 0.9,
            "ego": {
                "id": "ego", "key": "vehicle:ego", "kind": "vehicle",
                "position": [0.0, 0.0], "velocity": [1.0, 0.0],
                "heading": 0.0, "speed": 1.0, "length": 4.5, "width": 1.8,
                "road_id": "-E1", "lane_id": "-E1_0", "lane_position": 2.0,
                "state_source": "current", "state_time": 0.9,
            },
            "vehicles": [], "errors": [],
        }
        for index in range(40):
            full["vehicles"].append({
                "id": f"v{index:02d}", "key": f"vehicle:v{index:02d}", "kind": "vehicle",
                "position": [float(index + 1), 0.0], "velocity": [1.0, 0.0],
                "heading": 0.0, "speed": 1.0, "length": 4.5, "width": 1.8,
                "road_id": "-E1", "lane_id": "-E1_0", "lane_position": float(index),
                "state_source": "current", "state_time": 0.9,
            })

        class FakeRecorder:
            def __init__(self):
                self.capped_vehicle_count = None

            def on_raw_step(self, snapshot):
                self.capped_vehicle_count = min(32, len(snapshot["vehicles"]))

        recorder = FakeRecorder()
        holder, restore = runner._install_raw_snapshot_observer(recorder)
        predictor = SimpleNamespace(latest_full_physical_snapshot=None)
        holder["predictor"] = predictor
        recorder.on_raw_step(full)
        self.assertEqual(recorder.capped_vehicle_count, 32)
        self.assertEqual(len(predictor.latest_full_physical_snapshot["vehicles"]), 40)
        self.assertTrue(predictor.latest_full_physical_snapshot["physical_state_complete"])
        self.assertEqual(predictor.latest_full_physical_snapshot["snapshot_capture_source"], "raw_step_observer")
        self.assertEqual(predictor.latest_full_physical_snapshot["physical_actor_count"], 41)
        restore()

    def test_reset_snapshot_fallback_reads_cache_without_consuming_diagnostic_state(self):
        from types import SimpleNamespace

        def actor(actor_id, x, road_id="-E1", lane_id="-E1_0", sample_raw_step=0):
            return {
                "id": actor_id, "key": f"vehicle:{actor_id}", "kind": "vehicle",
                "position": [x, 0.0], "velocity": [1.0, 0.0], "heading": 0.0,
                "speed": 1.0, "length": 4.5, "width": 1.8, "road_id": road_id,
                "lane_id": lane_id, "lane_position": x, "state_source": "current",
                "state_time": 0.0, "context_sample_raw_step": sample_raw_step,
            }

        cache_snapshot = {
            "raw_step": 0, "sim_time": 0.0, "ego_state_source": "current",
            "ego_state_time": 0.0, "ego": actor("ego", 0.0),
            "vehicles": [actor("v1", 4.0)], "errors": [],
        }
        reset_snapshot = {
            **cache_snapshot,
            "vehicles": [actor("v1", 4.0, road_id=None, lane_id=None, sample_raw_step=None)],
        }
        live_identity = {
            "ego": ("-E1", "-E1_0"),
            "v1": ("-E1", "-E1_0"),
        }
        vehicle_api = SimpleNamespace(
            getIDList=lambda: ["ego", "v1"],
            getRoadID=lambda actor_id: live_identity[actor_id][0],
            getLaneID=lambda actor_id: live_identity[actor_id][1],
        )
        base = SimpleNamespace(
            _raw_steps=0,
            _behavior_pending_errors=[{"field": "existing"}],
            _behavior_error_keys={"existing"},
            _behavior_step_seconds=None,
            _last_events=(False, False, False, False),
            specification=SimpleNamespace(include_pedestrians=False),
            _connection=SimpleNamespace(
                vehicle=vehicle_api,
                person=SimpleNamespace(getIDList=lambda: []),
                simulation=SimpleNamespace(getTime=lambda: 0.0),
            ),
        )
        calls = []

        def behavior_snapshot(events):
            calls.append(events)
            base._behavior_pending_errors.clear()
            base._behavior_error_keys.add("fallback-error")
            base._behavior_step_seconds = 0.1
            return cache_snapshot

        base._behavior_snapshot = behavior_snapshot
        recorder = SimpleNamespace(episode={"initial_snapshot": reset_snapshot})
        env = SimpleNamespace(unwrapped=base)
        result = runner._verified_physical_snapshot(recorder, env)
        self.assertTrue(result["physical_state_complete"])
        self.assertEqual(result["snapshot_capture_source"], "environment_behavior_snapshot_cache")
        self.assertEqual(result["physical_membership"]["vehicle_id_set_matches"], True)
        self.assertEqual(result["physical_identity_query"]["completed"], True)
        self.assertIn("vehicle:v1:road_id_missing", result["initial_snapshot_incomplete_reasons"])
        self.assertEqual(result["raw_step"], 0)
        self.assertEqual(calls, [(False, False, False, False)])
        self.assertEqual(base._behavior_pending_errors, [{"field": "existing"}])
        self.assertEqual(base._behavior_error_keys, {"existing"})
        self.assertIsNone(base._behavior_step_seconds)

    def test_live_road_lane_query_uses_current_values_and_preserves_stale_cache(self):
        from types import SimpleNamespace

        def actor(actor_id, road, lane, sample):
            return {
                "id": actor_id, "key": f"vehicle:{actor_id}", "kind": "vehicle",
                "position": [0.0, 0.0], "velocity": [1.0, 0.0], "heading": 0.0,
                "speed": 1.0, "length": 4.5, "width": 1.8, "road_id": road,
                "lane_id": lane, "lane_position": 0.0, "state_source": "current",
                "state_time": 2.5, "context_sample_raw_step": sample,
            }

        raw_snapshot = {
            "raw_step": 5, "sim_time": 2.5, "ego_state_source": "current",
            "ego_state_time": 2.5, "ego": actor("ego", "-E1", "-E1_0", 5),
            "vehicles": [actor("v1", "-old", "-old_0", 2)], "errors": [],
        }
        live_identity = {
            "ego": ("-E1", "-E1_0"),
            "v1": (":J1_1", ":J1_1_0"),
        }

        def make_env():
            vehicle = SimpleNamespace(
                getIDList=lambda: ["ego", "v1"],
                getRoadID=lambda actor_id: live_identity[actor_id][0],
                getLaneID=lambda actor_id: live_identity[actor_id][1],
            )
            return SimpleNamespace(unwrapped=SimpleNamespace(
                _raw_steps=5,
                _connection=SimpleNamespace(
                    simulation=SimpleNamespace(getTime=lambda: 2.5), vehicle=vehicle,
                    person=SimpleNamespace(getIDList=lambda: []),
                ),
            ))

        snapshot = runner.physical_snapshot_payload(raw_snapshot)
        checked = runner.verify_physical_snapshot_membership(
            snapshot, current_vehicle_ids=["ego", "v1"], current_person_ids=[], current_raw_step=5
        )
        result = runner._supplement_live_road_lane_identity(
            checked, make_env(), current_vehicle_ids=["ego", "v1"], current_person_ids=[],
            current_raw_step=5,
        )
        self.assertTrue(result["physical_state_complete"])
        self.assertEqual(result["ego"]["road_id"], "-E1")
        self.assertEqual(result["vehicles"][0]["road_id"], ":J1_1")
        observation = result["vehicles"][0]["road_lane_cache_observation"]
        self.assertEqual(observation["cached_road_id"], "-old")
        self.assertEqual(observation["sample_raw_step"], 2)
        self.assertEqual(observation["relation_to_live_query"], "stale_sample")
        self.assertEqual(result["physical_identity_query"]["source"], "read_only_traci_current_road_lane")

        fresh_snapshot = runner.physical_snapshot_payload({
            **raw_snapshot,
            "vehicles": [actor("v1", ":J1_1", ":J1_1_0", 5)],
        })
        fresh_checked = runner.verify_physical_snapshot_membership(
            fresh_snapshot, current_vehicle_ids=["ego", "v1"], current_person_ids=[], current_raw_step=5
        )
        fresh = runner._supplement_live_road_lane_identity(
            fresh_checked, make_env(), current_vehicle_ids=["ego", "v1"], current_person_ids=[],
            current_raw_step=5,
        )
        self.assertTrue(fresh["physical_state_complete"])
        self.assertEqual(
            runner.stable_hash(runner.physical_fingerprint_payload(result)),
            runner.stable_hash(runner.physical_fingerprint_payload(fresh)),
        )

    def test_live_road_lane_query_rejects_same_step_conflict_and_query_failure(self):
        from types import SimpleNamespace

        def snapshot(sample=1):
            def actor(actor_id, road, lane):
                return {
                    "id": actor_id, "key": f"vehicle:{actor_id}", "kind": "vehicle",
                    "position": [0.0, 0.0], "velocity": [1.0, 0.0], "heading": 0.0,
                    "speed": 1.0, "length": 4.5, "width": 1.8, "road_id": road,
                    "lane_id": lane, "lane_position": 0.0, "state_source": "current",
                    "state_time": 1.0, "context_sample_raw_step": sample,
                }
            return runner.physical_snapshot_payload({
                "raw_step": 1, "sim_time": 1.0, "ego_state_source": "current",
                "ego_state_time": 1.0, "ego": actor("ego", "-E1", "-E1_0"),
                "vehicles": [actor("v1", "-old", "-old_0")], "errors": [],
            })

        def run(raw):
            stable_ids = ["ego", "v1"]
            vehicle = SimpleNamespace(
                getIDList=lambda: stable_ids,
                getRoadID=lambda actor_id: "-E1" if actor_id == "ego" else ":J1_1",
                getLaneID=lambda actor_id: (
                    "-E1_0" if actor_id == "ego" else (_ for _ in ()).throw(RuntimeError("getter failed"))
                ) if raw == "failure" else ("-E1_0" if actor_id == "ego" else ":J1_1_0"),
            )
            env = SimpleNamespace(unwrapped=SimpleNamespace(
                _raw_steps=1,
                _connection=SimpleNamespace(
                    simulation=SimpleNamespace(getTime=lambda: 1.0), vehicle=vehicle,
                    person=SimpleNamespace(getIDList=lambda: []),
                ),
            ))
            checked = runner.verify_physical_snapshot_membership(
                snapshot(), current_vehicle_ids=stable_ids, current_person_ids=[], current_raw_step=1
            )
            return runner._supplement_live_road_lane_identity(
                checked, env, current_vehicle_ids=stable_ids, current_person_ids=[], current_raw_step=1
            )

        conflict = run("conflict")
        self.assertFalse(conflict["physical_state_complete"])
        self.assertTrue(any(
            reason.startswith("id_query_failed:same_step_cached_road_id_mismatch")
            for reason in conflict["physical_state_incomplete_reasons"]
        ), conflict["physical_state_incomplete_reasons"])
        failed = run("failure")
        self.assertFalse(failed["physical_state_complete"])
        self.assertTrue(any("actor_query:vehicle:v1" in error for error in failed["physical_identity_query"]["errors"]))

    def test_live_road_lane_query_rejects_time_change_during_reads(self):
        from types import SimpleNamespace

        actor = {
            "id": "ego", "key": "vehicle:ego", "kind": "vehicle",
            "position": [0.0, 0.0], "velocity": [1.0, 0.0], "heading": 0.0,
            "speed": 1.0, "length": 4.5, "width": 1.8, "road_id": "-E1",
            "lane_id": "-E1_0", "lane_position": 0.0, "state_source": "current",
            "state_time": 1.0,
        }
        checked = runner.verify_physical_snapshot_membership(
            runner.physical_snapshot_payload({
                "raw_step": 1, "sim_time": 1.0, "ego_state_source": "current",
                "ego_state_time": 1.0, "ego": actor, "vehicles": [], "errors": [],
            }),
            current_vehicle_ids=["ego"], current_person_ids=[], current_raw_step=1,
        )
        times = iter([1.0, 1.1])
        vehicle = SimpleNamespace(
            getIDList=lambda: ["ego"], getRoadID=lambda _actor_id: "-E1",
            getLaneID=lambda _actor_id: "-E1_0",
        )
        env = SimpleNamespace(unwrapped=SimpleNamespace(
            _raw_steps=1,
            _connection=SimpleNamespace(
                simulation=SimpleNamespace(getTime=lambda: next(times)), vehicle=vehicle,
                person=SimpleNamespace(getIDList=lambda: []),
            ),
        ))
        result = runner._supplement_live_road_lane_identity(
            checked, env, current_vehicle_ids=["ego"], current_person_ids=[], current_raw_step=1
        )
        self.assertFalse(result["physical_state_complete"])
        self.assertIn("sim_time_changed_during_road_lane_query", result["physical_identity_query"]["errors"])

    def test_full_physical_snapshot_requires_live_idset_and_rawstep_match(self):
        snapshot = runner.physical_snapshot_payload({
            "raw_step": 9,
            "sim_time": 0.9,
            "ego_state_source": "current",
            "ego_state_time": 0.9,
            "ego": {
                "id": "ego", "key": "vehicle:ego", "kind": "vehicle",
                "position": [0.0, 0.0], "velocity": [1.0, 0.0], "heading": 0.0,
                "speed": 1.0, "length": 4.5, "width": 1.8, "road_id": "-E1",
                "lane_id": "-E1_0", "lane_position": 2.0,
                "state_source": "current", "state_time": 0.9,
            },
            "vehicles": [{
                "id": "v1", "key": "vehicle:v1", "kind": "vehicle",
                "position": [2.0, 0.0], "velocity": [0.0, 0.0], "heading": 0.0,
                "speed": 0.0, "length": 4.5, "width": 1.8, "road_id": "-E1",
                "lane_id": "-E1_0", "lane_position": 4.0,
                "state_source": "current", "state_time": 0.9,
            }],
            "errors": [],
        })
        valid = runner.verify_physical_snapshot_membership(
            snapshot,
            current_vehicle_ids=["ego", "v1"],
            current_person_ids=[],
            current_raw_step=9,
        )
        self.assertTrue(valid["physical_state_complete"])
        self.assertTrue(valid["physical_membership"]["vehicle_id_set_matches"])
        mismatch = runner.verify_physical_snapshot_membership(
            snapshot,
            current_vehicle_ids=["ego", "v1", "unseen"],
            current_person_ids=[],
            current_raw_step=9,
        )
        self.assertFalse(mismatch["physical_state_complete"])
        self.assertEqual(mismatch["physical_membership"]["missing_vehicle_ids"], ["unseen"])
        stale = runner.verify_physical_snapshot_membership(
            snapshot,
            current_vehicle_ids=["ego", "v1"],
            current_person_ids=[],
            current_raw_step=10,
        )
        self.assertIn("stale_raw_snapshot", stale["physical_state_incomplete_reasons"])

    def test_nonfinite_vehicle_physical_state_is_incomplete(self):
        payload = {
            "raw_step": 3,
            "sim_time": 0.3,
            "ego_state_source": "current",
            "ego_state_time": 0.3,
            "ego": {
                "id": "ego", "key": "vehicle:ego", "kind": "vehicle",
                "position": [float("nan"), 0.0], "velocity": [1.0, 0.0], "heading": 0.0,
                "speed": 1.0, "length": 4.5, "width": 1.8, "road_id": "-E1",
                "lane_id": "-E1_0", "lane_position": 2.0,
                "state_source": "current", "state_time": 0.3,
            },
            "vehicles": [], "errors": [],
        }
        snapshot = runner.physical_snapshot_payload(payload)
        self.assertFalse(snapshot["physical_state_complete"])
        self.assertIn("ego:position_nonfinite", snapshot["physical_state_incomplete_reasons"])

    def test_intervention_and_episode_budget_are_bounded(self):
        self.assertEqual(runner.MAX_CONTROL_EPISODES, 86)
        self.assertEqual(runner.MAX_CONTROL_RAW_STEPS, 51600)
        self.assertEqual(runner.INTERVENTION_DECISIONS, 3)
        self.assertEqual(runner.INTERVENTION_TARGET_SPEEDS_MPS, (0.0, 6.0))

    def test_recovery_budget_reserves_parent_consumption_and_caps_paired_windows(self):
        plan = runner._recovery_budget_plan(13, 3340)
        self.assertEqual(plan["intervention_pairs_max"], 29)
        self.assertEqual(plan["intervention_episodes_max"], 58)
        self.assertEqual(plan["recovery_episodes_max"], 72)
        self.assertEqual(plan["planned_cumulative_episodes_upper_bound"], 85)
        self.assertEqual(plan["planned_cumulative_raw_control_steps_upper_bound"], 46540)
        self.assertEqual(plan["unused_episode_slots_after_schedule"], 1)
        with self.assertRaises(ValueError):
            runner._recovery_budget_plan(-1, 0)

    def test_recovery_schedule_reserves_phase_first_seed_order_and_both_actions(self):
        cases = [{"seed": seed} for seed in range(10000, 10012)]
        baselines = {}
        for seed in range(10000, 10012):
            trace = []
            for index in range(10):
                trace.append({
                    "pre_ego_road_id": ":J1_1" if index == 4 else "-E1",
                    "pre_obs_raw_step": index * 3,
                    "physical_state_complete": True,
                    "physical_state_sha256": f"{seed}-{index}",
                    "physical_identity_query": {"completed": True},
                })
            baselines[seed] = {"trace": trace, "anchor_plan": runner.plan_anchors(trace)}
        schedule = runner._build_intervention_schedule(cases, baselines, pair_limit=29)
        self.assertEqual(schedule["reserved_slot_count"], 29)
        self.assertEqual(schedule["pairs_planned"], 29)
        self.assertEqual(
            [pair["phase"] for pair in schedule["pairs"][:12]],
            [runner.ANCHOR_PHASE_ORDER[0]] * 12,
        )
        self.assertEqual(
            [pair["seed"] for pair in schedule["pairs"][:12]],
            list(range(10000, 10012)),
        )
        self.assertEqual(
            [pair["phase"] for pair in schedule["pairs"][12:24]],
            [runner.ANCHOR_PHASE_ORDER[1]] * 12,
        )
        self.assertEqual(
            [pair["seed"] for pair in schedule["pairs"][24:]],
            list(range(10000, 10005)),
        )
        self.assertTrue(all(
            pair["target_speeds_mps"] == [0.0, 6.0] for pair in schedule["pairs"]
        ))
        self.assertEqual(schedule["episodes_planned"], 58)

    def test_fixed_schedule_does_not_refill_missing_or_merged_slots(self):
        cases = [{"seed": seed} for seed in range(10000, 10012)]
        baselines = {}
        for seed in range(10000, 10012):
            length = 3 if seed == 10000 else 10
            internal_index = 1 if seed == 10000 else 4
            trace = []
            for index in range(length):
                trace.append({
                    "pre_ego_road_id": ":J1_1" if index == internal_index else "-E1",
                    "pre_obs_raw_step": index * 3,
                    "physical_state_complete": not (seed == 10000 and index == 0),
                    "physical_state_sha256": f"{seed}-{index}",
                    "physical_identity_query": {"completed": True},
                })
            baselines[seed] = {"trace": trace, "anchor_plan": runner.plan_anchors(trace)}
        schedule = runner._build_intervention_schedule(cases, baselines, pair_limit=29)
        self.assertEqual(schedule["reserved_slot_count"], 29)
        self.assertEqual(schedule["pairs_planned"], 26)
        reasons = {(row["schedule_slot"], row["reason"]) for row in schedule["skipped"]}
        self.assertIn((0, "baseline_physical_state_incomplete"), reasons)
        self.assertIn((12, "baseline_physical_state_incomplete"), reasons)
        self.assertIn((24, "duplicate_anchor_merged"), reasons)
        self.assertTrue(any(
            row["schedule_slot"] >= 29 and row["reason"] == "outside_fixed_schedule_cap"
            for row in schedule["skipped"]
        ))

    def test_history_index_audit_matches_nonzero_mask_and_separates_ego_social(self):
        import numpy as np

        trajectory = np.zeros((3, 10, 5), dtype=np.float32)
        trajectory[0, :, 0] = 1.0
        trajectory[1, 5:, 0] = np.arange(1.0, 6.0)
        observation = {"trajectory": trajectory}
        before = trajectory.copy()
        audit = runner._trajectory_history_index_audit(observation, np=np)
        np.testing.assert_array_equal(trajectory, before)
        self.assertEqual(audit["status"], "complete")
        self.assertEqual(audit["mask_rule"], "trajectory[..., 0] != 0; exactly matches _nonzero_mask")
        ego, social_leftpad, empty_social = audit["actors"]
        self.assertEqual(ego["actor_role"], "ego")
        self.assertEqual(ego["valid_count"], 10)
        self.assertFalse(ego["count_minus_one_index_is_model_padding"])
        self.assertEqual(social_leftpad["actor_role"], "social")
        self.assertEqual(social_leftpad["valid_count"], 5)
        self.assertEqual(social_leftpad["count_minus_one_index"], 4)
        self.assertTrue(social_leftpad["count_minus_one_index_is_model_padding"])
        self.assertEqual(social_leftpad["true_last_valid_index"], 9)
        self.assertEqual(social_leftpad["index_delta_true_minus_count_minus_one"], 5)
        self.assertEqual(empty_social["valid_count"], 0)
        summary = runner._summarize_history_index_audit([{"trace": [{
            "observation_history_index_audit": audit,
        }]}])
        self.assertEqual(summary["actor_groups"]["ego"]["nonempty_actor_histories"], 1)
        self.assertEqual(summary["actor_groups"]["social"]["nonempty_actor_histories"], 1)
        self.assertEqual(summary["actor_groups"]["social"]["selected_model_padding_count"], 1)

    def test_intervention_anchor_must_be_reached_but_can_terminate_after_one_action(self):
        from types import SimpleNamespace

        not_reached = SimpleNamespace(
            target_speed_mps=6.0, anchor_index=5, prefix_error=None,
            trace=[{}] * 5, intervention_applied_decisions=0,
        )
        runner._validate_intervention_anchor(not_reached, outcome="collision")
        self.assertEqual(not_reached.prefix_error["reason"], "anchor_not_reached")
        self.assertEqual(not_reached.prefix_error["replay_outcome"], "collision")

        one_step_terminal = SimpleNamespace(
            target_speed_mps=0.0, anchor_index=5, prefix_error=None,
            trace=[{}] * 6, intervention_applied_decisions=1,
        )
        runner._validate_intervention_anchor(one_step_terminal, outcome="collision")
        self.assertIsNone(one_step_terminal.prefix_error)

    def test_runtime_snapshot_contains_runner_and_protocol_copies_with_hashes(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diag"
            root.mkdir()
            manifest = runner._archive_runtime_sources(root)
            by_path = {row["path"]: row for row in manifest["files"]}
            for relative in (
                "fast-developer/diagnose_strt_decision_windows_20261003.py",
                "fast-developer/analysis/decision_window_intervention_protocol_20261003.md",
            ):
                row = by_path[relative]
                copied = root / "runtime_source_snapshot" / Path(relative)
                self.assertTrue(copied.is_file())
                self.assertEqual(runner.sha256_file(copied), row["sha256"])
            saved = json.loads((root / "runtime_source_snapshot_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["file_count"], len(by_path))


if __name__ == "__main__":
    unittest.main()
