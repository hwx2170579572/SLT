from __future__ import annotations

import pytest

from tools import attribute_v4_10_speed_proposal_gap as attribution


def _row(
    *,
    episode: int,
    decision: int,
    outcome: str,
    gain: float,
    risk: float,
    delta: float,
    best_normalized: float,
) -> dict:
    return {
        "episode": episode,
        "decision": decision,
        "eventual_episode_outcome": outcome,
        "learned_speed_proposal_probe": {
            "same_lane_score_gain": gain,
            "same_lane_collision_value_reduction": risk,
            "same_lane_speed_delta_mps": delta,
            "baseline_speed_mps": 5.0,
            "same_lane_best": {
                "source": "diagnostic_grid",
                "best_speed_normalized": best_normalized,
                "best_speed_mps": (best_normalized + 1.0) * 5.0,
            },
        },
    }


def test_terminal_rows_are_episode_local() -> None:
    rows = [
        _row(
            episode=episode,
            decision=decision,
            outcome="collision",
            gain=0.1,
            risk=0.1,
            delta=-1.0,
            best_normalized=-0.95,
        )
        for episode in (0, 1)
        for decision in range(4)
    ]
    selected = attribution.terminal_rows(rows, outcome="collision", window=2)
    assert [(row["episode"], row["decision"]) for row in selected] == [
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    ]


def test_summary_separates_edges_direction_and_safety() -> None:
    rows = [
        _row(
            episode=0,
            decision=0,
            outcome="collision",
            gain=0.2,
            risk=0.1,
            delta=-2.0,
            best_normalized=-0.95,
        ),
        _row(
            episode=0,
            decision=1,
            outcome="collision",
            gain=0.1,
            risk=0.2,
            delta=2.0,
            best_normalized=0.95,
        ),
        _row(
            episode=0,
            decision=2,
            outcome="collision",
            gain=0.1,
            risk=-0.2,
            delta=1.0,
            best_normalized=0.0,
        ),
    ]
    value = attribution.summarize_rows(rows, grid_min=-0.95, grid_max=0.95)
    assert value["strict_score_improvement_rate"] == 1.0
    assert value["safer_and_higher_score_rate"] == pytest.approx(2 / 3)
    assert value["slower_given_safer_higher_rate"] == 0.5
    assert value["faster_given_safer_higher_rate"] == 0.5
    assert value["either_grid_edge_rate_given_grid_selected"] == pytest.approx(2 / 3)


def test_empty_summary_is_explicit() -> None:
    assert attribution.summarize_rows([], grid_min=-0.95, grid_max=0.95) == {
        "decision_count": 0
    }
