from __future__ import annotations

import math

from tools.attribute_v4_9_learned_model import _components, _rank_auc


def _row(selected_lane: int = 0) -> dict:
    return {
        "lane_command": selected_lane,
        "target_critic_decoder": {
            "valid_lane_actions": [True, True, False],
            "maximum_target_twin_collision_value": [0.1, 0.3, None],
            "minimum_target_twin_q": [0.4, 0.9, None],
            "risk_adjusted_target_score": [0.3, 0.6, None],
            "collision_risk_coef": 1.0,
        },
    }


def test_components_report_learned_collision_regret_without_override() -> None:
    result = _components(_row())
    assert result["selected_index"] == 1
    assert result["safest_index"] == 0
    assert math.isclose(result["collision_value_regret"], 0.2)
    assert result["selected_differs_from_minimum_collision_action"] is True


def test_components_reject_score_equation_drift() -> None:
    row = _row(-1)
    row["target_critic_decoder"]["risk_adjusted_target_score"][0] = 0.31
    try:
        _components(row)
    except ValueError as error:
        assert "equation drifted" in str(error)
    else:
        raise AssertionError("score equation drift must be rejected")


def test_rank_auc_is_tie_aware() -> None:
    assert _rank_auc([1.0, 2.0], [0.0, 2.0]) == 0.625
    assert _rank_auc([], [1.0]) is None
