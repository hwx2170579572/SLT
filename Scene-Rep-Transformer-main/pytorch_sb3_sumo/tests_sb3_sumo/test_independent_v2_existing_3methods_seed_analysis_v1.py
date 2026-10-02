from __future__ import annotations

import pytest

from tools import analyze_independent_v2_existing_3methods_100s100e_v1 as analysis


def test_directional_advantage_is_positive_when_v48_is_better() -> None:
    assert analysis._directional_advantage(0.9, 0.8, "higher") == pytest.approx(0.1)
    assert analysis._directional_advantage(0.1, 0.2, "lower") == pytest.approx(0.1)
    assert analysis._directional_advantage(None, 0.2, "lower") is None


def _seed_row(
    *,
    index: int,
    balanced: int,
    regressions: int,
    dominance: int,
    worst_margin: float,
    mean_margin: float,
    secondary_rate: float,
    return_advantage: float,
) -> dict:
    return {
        "logical_test_seed_index": index,
        "balanced_primary_dominance_count": balanced,
        "total_primary_regression_count": regressions,
        "total_primary_dominance_count": dominance,
        "worst_baseline_macro_primary_margin": worst_margin,
        "mean_primary_margin": mean_margin,
        "secondary_net_win_rate": secondary_rate,
        "worst_baseline_macro_return_advantage": return_advantage,
    }


def test_seed_ranking_prioritizes_balanced_primary_dominance() -> None:
    strongest_balanced = _seed_row(
        index=9,
        balanced=3,
        regressions=0,
        dominance=6,
        worst_margin=0.01,
        mean_margin=0.01,
        secondary_rate=0.0,
        return_advantage=0.0,
    )
    larger_margin_but_unbalanced = _seed_row(
        index=1,
        balanced=2,
        regressions=0,
        dominance=5,
        worst_margin=0.50,
        mean_margin=0.50,
        secondary_rate=1.0,
        return_advantage=1.0,
    )

    ranked = analysis._rank_seed_rows(
        [larger_margin_but_unbalanced, strongest_balanced]
    )

    assert ranked[0]["logical_test_seed_index"] == 9
    assert ranked[0]["selection_tier"] == "A"
    assert ranked[0]["selection_rank"] == 1


def test_seed_ranking_uses_seed_index_as_deterministic_final_tie_breaker() -> None:
    rows = [
        _seed_row(
            index=index,
            balanced=2,
            regressions=1,
            dominance=5,
            worst_margin=0.1,
            mean_margin=0.2,
            secondary_rate=0.5,
            return_advantage=0.3,
        )
        for index in (7, 2)
    ]

    ranked = analysis._rank_seed_rows(rows)

    assert [row["logical_test_seed_index"] for row in ranked] == [2, 7]
    assert all(row["selection_tier"] == "B" for row in ranked)
