from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import reconcile_v4_8_3_promotion_v3 as v3


def test_frozen_commands_never_expose_formal_or_run() -> None:
    status = v3.frozen_command("python", "status")
    summary = v3.frozen_command("python", "summarize_promotion")
    gate = v3.frozen_command("python", "gate_promotion")
    for command in (status, summary, gate):
        assert "formal" not in command
        assert "run" not in command[2:]
    assert summary[-2:] == ["--stage", "promotion"]
    assert gate[-2:] == ["--stage", "promotion"]
    with pytest.raises(v3.ReconciliationError, match="unsupported"):
        v3.frozen_command("python", "formal")


def test_formal_zero_guard() -> None:
    v3.assert_formal_zero(
        {"stages": {"formal": {"accepted": 0, "complete": False}}}
    )
    with pytest.raises(v3.ReconciliationError, match="formal stage was accessed"):
        v3.assert_formal_zero(
            {"stages": {"formal": {"accepted": 1, "complete": False}}}
        )


def test_context_bound_states_enters_full_patch_context(monkeypatch) -> None:
    entered = {"value": False}

    @contextmanager
    def patched():
        entered["value"] = True
        yield
        entered["value"] = False

    job = SimpleNamespace(name="P__cand__x", algorithm="candidate")
    monkeypatch.setattr(v3.runner, "_patched_v4_8_3_interfaces", patched)
    monkeypatch.setattr(
        v3.v48,
        "accepted_run_reason",
        lambda _job, _hashes: (entered["value"], "accepted"),
    )
    monkeypatch.setattr(v3.v48, "run_directory", lambda _job: Path("."))
    rows = v3.context_bound_states([job], {})
    assert rows[0]["accepted"] is True
    assert rows[0]["reason"] == "accepted"
    assert entered["value"] is False


def test_v2_false_negative_trigger_requires_all_candidate_jobs() -> None:
    jobs = [
        SimpleNamespace(name=f"P__cand__x__s{seed}", algorithm=v3.runner.V48_CANDIDATE)
        for seed in range(2)
    ]
    state = {
        "status": "all_jobs_attempted_with_failures",
        "accepted_count": 6,
        "expected_count": 12,
        "scientific_gate_computed": False,
        "formal_stage_launched": False,
    }
    summary = {
        "attempts_this_orchestration": [
            {
                "job": job.name,
                "status": "failed",
                "returncode": 0,
                "acceptance_reason": "v4.8 changed return estimator",
            }
            for job in jobs
        ],
        "all_jobs_attempted_or_preaccepted": True,
        "formal_accepted": 0,
    }
    v3.validate_v2_false_negative(state, summary, jobs)
    summary["attempts_this_orchestration"][0]["acceptance_reason"] = "other"
    with pytest.raises(v3.ReconciliationError, match="job set drifted"):
        v3.validate_v2_false_negative(state, summary, jobs)


def test_summary_requires_exact_job_identity_set() -> None:
    jobs = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    v3.validate_summary(
        {"complete": True, "per_run": [{"run_name": "a"}, {"run_name": "b"}]},
        jobs,
    )
    with pytest.raises(v3.ReconciliationError, match="identities drifted"):
        v3.validate_summary(
            {"complete": True, "per_run": [{"run_name": "a"}, {"run_name": "c"}]},
            jobs,
        )


def test_v3_uses_distinct_schema_and_artifacts() -> None:
    assert v3.SCHEMA_VERSION.endswith("/v3")
    assert "v3" in v3.DEFAULT_OUTPUT.name
    assert "v3" in v3.DEFAULT_STATE.name
    assert v3.DEFAULT_OUTPUT != v3.DEFAULT_V2_SUMMARY
