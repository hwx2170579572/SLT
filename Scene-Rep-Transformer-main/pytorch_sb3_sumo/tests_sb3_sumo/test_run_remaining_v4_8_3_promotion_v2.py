from __future__ import annotations

import pytest

from tools import run_remaining_v4_8_3_promotion as v1
from tools import run_remaining_v4_8_3_promotion_v2 as v2


def test_v2_runs_one_exact_promotion_job_serially() -> None:
    job = "P__cand__ram__s1__p42f54135"
    command = v2.runner_command(
        "python",
        "run_promotion_job",
        device="cuda",
        job_name=job,
    )
    assert command[-9:] == [
        "run",
        "--stage",
        "promotion",
        "--job",
        job,
        "--device",
        "cuda",
        "--workers",
        "1",
    ]
    assert "all" not in command
    assert "formal" not in command


def test_v2_rejects_non_promotion_or_missing_job_selector() -> None:
    with pytest.raises(v1.PromotionAutomationError, match="exact frozen"):
        v2.runner_command(
            "python",
            "run_promotion_job",
            device="cuda",
            job_name=None,
        )
    with pytest.raises(v1.PromotionAutomationError, match="exact frozen"):
        v2.runner_command(
            "python",
            "run_promotion_job",
            device="cuda",
            job_name="F__cand__cross__s0__p42f54135",
        )


def test_v2_formal_guard_allows_incomplete_promotion_but_zero_formal() -> None:
    v2.assert_formal_untouched(
        {
            "stages": {
                "promotion": {"accepted": 7, "complete": False},
                "formal": {"accepted": 0, "complete": False},
            }
        }
    )
    with pytest.raises(v1.PromotionAutomationError, match="formal stage"):
        v2.assert_formal_untouched(
            {
                "stages": {
                    "promotion": {"accepted": 7, "complete": False},
                    "formal": {"accepted": 1, "complete": False},
                }
            }
        )


def test_v2_extracts_only_evidence_bound_accepted_metrics() -> None:
    row = {
        "run_name": "P__tg__cross__s0__p42f54135",
        "method": "temporal_graph",
        "algorithm": "temporal_graph_v1_control",
        "scenario": "intersection_cross",
        "seed": 0,
        "success_rate": 0.8,
        "collision_rate": 0.1,
        "off_route_rate": 0.0,
        "timeout_rate": 0.1,
        "mean_return": 0.7,
        "mean_decision_steps": 10.0,
        "mean_raw_steps": 30.0,
        "selected_checkpoint_kind": "exact_final",
        "selected_deployment_decoder": "parent_control",
        "training_wall_seconds": 1.0,
        "mean_inference_ms": 2.0,
        "episode_records": [{"must_not_be_copied": True}],
    }
    result = v2.accepted_metric_rows({"per_run": [row]})
    assert len(result) == 1
    assert result[0]["success_rate"] == 0.8
    assert result[0]["selected_checkpoint_kind"] == "exact_final"
    assert "episode_records" not in result[0]


def test_v2_has_distinct_runtime_evidence_and_schema() -> None:
    assert v2.SCHEMA_VERSION.endswith("/v2")
    assert v2.DEFAULT_STATE != v1.DEFAULT_STATE
    assert v2.DEFAULT_EVENTS != v1.DEFAULT_EVENTS
    assert v2.DEFAULT_CHILD_LOG != v1.DEFAULT_CHILD_LOG
    assert v2.DEFAULT_LOCK != v1.DEFAULT_LOCK
