from __future__ import annotations

from pathlib import Path

import pytest

from tools import probe_v4_9_2_speed_proposal_coverage as probe


def _row(
    *,
    same_gain: float,
    same_risk: float,
    same_speed: float,
    global_gain: float,
    global_risk: float,
    global_speed: float,
    source: str = "diagnostic_grid",
    lane_changed: bool = False,
) -> dict:
    return {
        "learned_speed_proposal_probe": {
            "same_lane_best": {"source": source},
            "same_lane_score_gain": same_gain,
            "same_lane_collision_value_reduction": same_risk,
            "same_lane_speed_delta_mps": same_speed,
            "global_score_gain": global_gain,
            "global_collision_value_reduction": global_risk,
            "global_speed_delta_mps": global_speed,
            "global_lane_changed": lane_changed,
        }
    }


def test_probe_summary_separates_score_gain_from_safer_gain() -> None:
    rows = [
        _row(
            same_gain=0.2,
            same_risk=0.1,
            same_speed=-2.0,
            global_gain=0.3,
            global_risk=0.2,
            global_speed=-2.5,
        ),
        _row(
            same_gain=0.1,
            same_risk=-0.2,
            same_speed=1.0,
            global_gain=0.1,
            global_risk=-0.2,
            global_speed=1.0,
            lane_changed=True,
        ),
        _row(
            same_gain=0.0,
            same_risk=0.0,
            same_speed=0.0,
            global_gain=0.0,
            global_risk=0.0,
            global_speed=0.0,
            source="actor",
        ),
    ]
    value = probe.summarize_probe_rows(rows)
    assert value["decision_count"] == 3
    assert value["same_lane_strict_score_improvement_rate"] == pytest.approx(2 / 3)
    assert value["same_lane_safer_and_higher_score_rate"] == pytest.approx(1 / 3)
    assert value["same_lane_lower_speed_when_improved_rate"] == pytest.approx(0.5)
    assert value["global_lane_changed_rate"] == pytest.approx(1 / 3)


def test_outcome_and_terminal_window_summaries_use_eventual_episode_result() -> None:
    rows = []
    for episode, gains in ((0, [0.0, 0.2, 0.4]), (1, [0.0, 0.0])):
        for gain in gains:
            row = _row(
                same_gain=gain,
                same_risk=gain,
                same_speed=-gain,
                global_gain=gain,
                global_risk=gain,
                global_speed=-gain,
            )
            row["episode"] = episode
            rows.append(row)
    report = {
        "episode_records": [
            {"collision": True, "success": False, "off_route": False},
            {"collision": False, "success": True, "off_route": False},
        ]
    }
    value = probe.summarize_probe(rows, report)
    assert value["by_episode_outcome"]["collision"]["decision_count"] == 3
    assert value["by_episode_outcome"]["success"]["decision_count"] == 2
    assert value["collision_terminal_windows"]["1"]["decision_count"] == 1
    assert value["collision_terminal_windows"]["5"]["decision_count"] == 3


def test_complete_attribution_is_required() -> None:
    complete = {
        "decision_allowed": True,
        "partial_matrix": False,
        "pair_count": 6,
        "pairs": [
            {"scenario": "cross", "seed": seed} for seed in range(6)
        ],
    }
    assert len(probe._candidate_pairs(complete)) == 6
    with pytest.raises(ValueError, match="partial"):
        probe._candidate_pairs({**complete, "partial_matrix": True})
    with pytest.raises(ValueError, match="six"):
        probe._candidate_pairs({**complete, "pair_count": 5})


def test_default_grid_is_strictly_inside_action_bounds() -> None:
    assert tuple(sorted(set(probe.DEFAULT_SPEED_GRID))) == probe.DEFAULT_SPEED_GRID
    assert min(probe.DEFAULT_SPEED_GRID) > -1.0
    assert max(probe.DEFAULT_SPEED_GRID) < 1.0


def test_evaluation_outcomes_reads_detailed_report_summary() -> None:
    report = {
        "summary": {
            "success_rate": 0.5,
            "collision_rate": 0.25,
            "off_route_rate": 0.0,
            "timeout_rate": 0.25,
        },
        "episode_records": [],
    }
    assert probe.evaluation_outcomes(report) == {
        "success_rate": 0.5,
        "collision_rate": 0.25,
        "off_route_rate": 0.0,
        "timeout_rate": 0.25,
    }
