"""One real SUMO transition through the behavior wrapper; no learner is created."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAST_DEVELOPER = PROJECT_ROOT / "fast-developer"
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

from behavior_diagnostics import wrap_behavior_diagnostics  # noqa: E402
from envs.sumo.paper_env import PaperSumoSceneEnv  # noqa: E402


def test_darrl_real_env_step_merges_route_info_and_records_policy_input(tmp_path):
    raw_env = PaperSumoSceneEnv(
        scenario="intersection_random_darrl_medium_v1",
        traffic_partition="train",
        traffic_split="train",
        action_repeat=1,
    )
    env = wrap_behavior_diagnostics(
        raw_env,
        tmp_path,
        "eval_worker_00",
        "diagnostic_info_merge_smoke",
        metadata={"scenario": "intersection_random_darrl_medium_v1"},
    )
    try:
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        next_observation, reward, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert not (terminated and truncated)
        assert info["behavior_telemetry_protocol"] == "route_lane_behavior_v1"
        assert "route_lane_status_known" in info
        assert "route_lane_status_reason" in info
        assert "planned_next_edge" in info
        assert "current_lane_can_reach_next_edge" in info
        assert info["action_preclip"] == [0.0, 0.0]
        assert info["action_clipped"] == [0.0, 0.0]
        assert info["target_speed"] == 5.0
        assert info["lane_command_requested"] == 0
        assert info["lane_control_request_status"] == "hold"
        assert info["lane_control_request_reason"] == "hold_command"
        assert "expected_target_lane_id" in info
        assert "actual_speed_mps" in info
    finally:
        env.close()

    diagnostics = tmp_path / "diagnostics" / "eval_worker_00"
    with gzip.open(diagnostics / "decisions.jsonl.gz", "rt", encoding="utf-8") as stream:
        decisions = [json.loads(line) for line in stream]
    with gzip.open(diagnostics / "raw_steps.jsonl.gz", "rt", encoding="utf-8") as stream:
        raw_steps = [json.loads(line) for line in stream]
    with gzip.open(
        diagnostics / "policy_observations.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        policy_observations = [json.loads(line) for line in stream]

    assert len(decisions) == 1
    assert decisions[0]["control"]["behavior_telemetry_protocol"] == "route_lane_behavior_v1"
    assert decisions[0]["control"]["lane_control_request_status"] == "hold"
    assert len(raw_steps) == 1
    assert raw_steps[0]["route_reachability_protocol"] == "route_continuation_v1"
    assert len(policy_observations) >= 1
    assert policy_observations[0]["trigger"] == "eval_first_decision_input"
    assert policy_observations[0]["observation"] == {
        key: value.tolist() for key, value in observation.items()
    }
