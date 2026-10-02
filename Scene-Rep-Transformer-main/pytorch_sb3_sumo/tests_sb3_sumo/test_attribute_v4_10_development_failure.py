from __future__ import annotations

import pytest

from tools import attribute_v4_10_development_failure as attribution


def _decoder() -> dict:
    return {
        "candidate_source": "learned_actor_component_means",
        "fixed_speed_grid_used": False,
        "action_rewritten": False,
        "semantic_tie_override_used": False,
        "valid_lane_actions": [False, True, True],
        "component_probabilities": [
            [0.5, 0.5],
            [0.8, 0.2],
            [0.4, 0.6],
        ],
        "lane_probabilities": [0.0, 0.7, 0.3],
        "learned_speed_proposals_normalized": [
            [-0.5, 0.5],
            [-0.4, 0.4],
            [-0.2, 0.2],
        ],
        "minimum_target_twin_q": [None, [0.5, 0.7], [0.4, 0.6]],
        "maximum_target_twin_collision_value": [None, [0.1, 0.3], [0.05, 0.2]],
        "reward_twin_disagreement": [None, [0.02, 0.03], [0.01, 0.04]],
        "collision_twin_disagreement": [None, [0.01, 0.02], [0.02, 0.01]],
        "learned_uncertainty": [None, [0.03, 0.05], [0.03, 0.05]],
        "supported_risk_adjusted_score": [None, [0.39, 0.42], [0.30, 0.35]],
        "selected_lane_index": 1,
        "selected_lane": 0,
        "selected_component_index": 1,
    }


def _row(*, episode: int, decision: int, collision: bool = False, success: bool = False) -> dict:
    return {
        "episode": episode,
        "decision": decision,
        "seed": 100 + episode,
        "traffic_variant": f"traffic_{episode}.rou.xml",
        "collision": collision,
        "is_success": success,
        "timeout": False,
        "off_route": False,
        "action_longitudinal": 0.4,
        "actual_speed_mps": 7.0,
        "lane_change_applied": False,
        "route_intent_match": True,
        "pre_action_non_keep_feasible": True,
        "target_critic_decoder": _decoder(),
    }


def test_decision_metrics_use_only_learned_feasible_proposals() -> None:
    value = attribution.decision_metrics(_row(episode=0, decision=0, success=True))
    assert value["selected_collision_value"] == pytest.approx(0.3)
    assert value["minimum_feasible_collision_value"] == pytest.approx(0.05)
    assert value["risk_regret_to_feasible_minimum"] == pytest.approx(0.25)
    assert value["same_lane_risk_regret"] == pytest.approx(0.2)
    assert value["score_margin_over_runner_up"] == pytest.approx(0.03)
    assert value["selected_speed_normalized"] == pytest.approx(0.4)
    assert value["selected_lane_proposal_speed_spread"] == pytest.approx(0.8)
    assert value["selected_component_is_probability_mode"] is False


def test_terminal_rows_are_episode_local_and_outcome_derived() -> None:
    rows = []
    for episode in (0, 1):
        for decision in range(4):
            rows.append(
                attribution.decision_metrics(
                    _row(
                        episode=episode,
                        decision=decision,
                        collision=episode == 0 and decision == 3,
                        success=episode == 1 and decision == 3,
                    )
                )
            )
    episodes = attribution.group_episodes(rows)
    selected = attribution.terminal_rows(episodes, outcome="collision", window=2)
    assert [(row["episode"], row["decision"]) for row in selected] == [(0, 2), (0, 3)]
    assert all(row["eventual_outcome"] == "success" for row in episodes[1])


def test_binary_auc_is_tie_aware_and_handles_single_class() -> None:
    assert attribution.binary_auc([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 1.0
    assert attribution.binary_auc([0.5, 0.5], [0, 1]) == 0.5
    assert attribution.binary_auc([0.1, 0.2], [0, 0]) is None


def test_imminent_collision_window_labels_only_collision_tail() -> None:
    rows = []
    for episode in (0, 1):
        for decision in range(3):
            row = _row(
                episode=episode,
                decision=decision,
                collision=episode == 0 and decision == 2,
                success=episode == 1 and decision == 2,
            )
            metric = attribution.decision_metrics(row)
            metric["selected_collision_value"] = float(decision) / 10.0
            rows.append(metric)
    episodes = attribution.group_episodes(rows)
    value = attribution.imminent_collision_ranking(episodes, horizon=2)
    assert value["sample_count"] == 6
    assert value["positive_count"] == 2
    assert value["positive_rate"] == pytest.approx(1 / 3)


def test_subset_summary_empty_is_explicit() -> None:
    assert attribution.subset_summary([]) == {"decision_count": 0}
