from __future__ import annotations

import copy

import pytest

from tools import attribute_v4_12_development_failure as subject


def _paper(seed_start: int, outcomes: tuple[str, str]) -> dict:
    records = []
    for episode, outcome in enumerate(outcomes):
        records.append(
            {
                "episode": episode,
                "seed": seed_start + episode,
                "traffic_variant": f"traffic_{episode}.rou.xml",
                "decision_steps": 3,
                "success": outcome == "success",
                "collision": outcome == "collision",
                "off_route": outcome == "off_route",
                "timeout": outcome == "timeout",
            }
        )
    return {"episode_records": records}


def _decoder() -> dict:
    return {
        "valid_lane_actions": [True, True, False],
        "component_probabilities": [[1.0], [1.0], [1.0]],
        "learned_speed_proposals_normalized": [[-0.1], [0.2], [0.3]],
        "minimum_target_twin_q": [[0.42], [0.40], None],
        "maximum_target_twin_collision_value": [[0.1], [0.1], None],
        "reward_twin_disagreement": [[0.01], [0.01], None],
        "collision_twin_disagreement": [[0.01], [0.01], None],
        "learned_uncertainty": [[0.02], [0.02], None],
        # Lane 0 wins only after the 0.05 * log(p_lane) support term.
        "supported_risk_adjusted_score": [[0.25], [0.24], None],
        "lane_probabilities": [0.9, 0.1, 0.0],
        "lane_prior_coef": 0.05,
        "selected_lane_index": 0,
        "selected_lane": -1,
        "selected_component_index": 0,
        "candidate_source": "learned_actor_component_means",
        "fixed_speed_grid_used": False,
        "action_rewritten": False,
        "semantic_tie_override_used": False,
    }


def test_episode_alignment_refuses_to_call_different_seeds_paired() -> None:
    result = subject.episode_alignment(
        _paper(100, ("success", "collision")),
        _paper(200, ("collision", "collision")),
    )
    assert result["exact_seed_paired"] is False
    assert result["causal_paired_outcome_claim_allowed"] is False
    assert result["outcome_transition_counts_descriptive_only"] == {
        "collision__to__collision": 1,
        "success__to__collision": 1,
    }


def test_remove_lane_prior_uses_continuous_score_not_rule() -> None:
    decoder = _decoder()
    row = {"target_critic_decoder": decoder}
    choice = subject.remove_lane_prior_choice(row)
    assert choice["lane_index"] == 1
    assert choice["lane_code"] == 0
    assert choice["changed_lane"] is True
    expected = 0.24 - 0.05 * __import__("math").log(0.1)
    assert choice["score_without_lane_prior"] == pytest.approx(expected)


def test_stored_lane_prior_coefficient_reproduces_stored_choice() -> None:
    choice = subject.lane_prior_choice(
        {"target_critic_decoder": _decoder()}, 0.05
    )
    assert choice["lane_index"] == 0
    assert choice["changed_lane"] is False


def test_remove_lane_prior_rejects_non_argmax_trace() -> None:
    decoder = copy.deepcopy(_decoder())
    decoder["selected_lane_index"] = 1
    decoder["selected_lane"] = 0
    with pytest.raises(ValueError, match="not score argmax"):
        subject.remove_lane_prior_choice({"target_critic_decoder": decoder})
