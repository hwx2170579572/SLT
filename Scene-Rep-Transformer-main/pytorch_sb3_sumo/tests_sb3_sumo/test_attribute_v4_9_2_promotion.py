from __future__ import annotations

from tools import attribute_v4_9_2_promotion as attribution


def _row(
    decision: int,
    lane: int,
    *,
    collision: bool = False,
    success: bool = False,
    terminated: bool = False,
    actor_lane: int = 0,
) -> dict:
    probabilities = {
        -1: [1.0, 0.0, 0.0],
        0: [0.0, 1.0, 0.0],
        1: [0.0, 0.0, 1.0],
    }[actor_lane]
    return {
        "episode": 0,
        "decision": decision,
        "seed": 100,
        "lane_command": lane,
        "collision": collision,
        "is_success": success,
        "off_route": False,
        "timeout": False,
        "terminated": terminated,
        "truncated": False,
        "lane_change_applied": lane != 0,
        "pre_action_route_intent_valid": True,
        "pre_action_non_keep_feasible": True,
        "route_intent_match": lane == 0,
        "longitudinal_saturated": False,
        "target_speed_mps": 4.0 + decision,
        "actual_speed_mps": 3.0 + decision,
        "hybrid_policy": {"lane_probabilities": probabilities},
    }


def test_outcome_priority() -> None:
    assert attribution.outcome({"collision": True, "is_success": True}) == "collision"
    assert attribution.outcome({"collision": False, "is_success": True}) == "success"
    assert attribution.outcome({"collision": False, "timeout": True}) == "timeout"


def test_behavior_metrics_measure_switches_and_actor_mismatch() -> None:
    episodes = {
        0: [
            _row(0, 0, actor_lane=0),
            _row(1, 1, actor_lane=-1),
            _row(2, -1, collision=True, terminated=True, actor_lane=-1),
        ]
    }
    metrics = attribution.behavior_metrics(episodes)
    assert metrics["episodes"] == 1
    assert metrics["outcomes"] == {"collision": 1}
    assert metrics["non_keep_command_rate"] == 2 / 3
    assert metrics["lane_change_applied_rate"] == 2 / 3
    assert metrics["command_switch_rate"] == 1.0
    assert metrics["opposite_direction_flip_rate"] == 0.5
    assert metrics["actor_target_lane_match_rate"] == 2 / 3
    assert metrics["actor_non_keep_rate"] == 2 / 3
    assert metrics["terminal_windows"]["1"]["target_speed_mps"][
        "collision_episode_mean"
    ] == 6.0

