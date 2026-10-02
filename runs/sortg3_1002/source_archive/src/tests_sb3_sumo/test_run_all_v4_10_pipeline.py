from __future__ import annotations

from tools import run_all_pending_promotions as registry_subject
from tools import run_all_v4_10_pipeline as subject


def _install_protocol_stubs(monkeypatch):
    monkeypatch.setattr(subject.runner, "load_contract", lambda: {})
    monkeypatch.setattr(subject.runner, "validate_contract", lambda value: value)
    monkeypatch.setattr(subject.runner, "_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(
        subject.runner,
        "protocol_hashes",
        lambda: {
            "experiment_contract_sha256": "a" * 64,
            "stage0_results_sha256": "b" * 64,
            "attribution_sha256": "c" * 64,
            "implementation_freeze_sha256": "d" * 64,
        },
    )


def test_development_failure_is_aggregated_before_pipeline_stops(
    monkeypatch, tmp_path
):
    _install_protocol_stubs(monkeypatch)
    entered = []

    def fake_attempt(**kwargs):
        entered.append(kwargs["stage"])
        return {
            "stage": kwargs["stage"],
            "matrix_complete": True,
            "decision": "fail",
        }

    monkeypatch.setattr(subject, "_attempt_stage", fake_attempt)
    code, value = subject.run_all(
        device="cuda", report_path=tmp_path / "report.json"
    )
    assert code == 2
    assert entered == ["development"]
    assert value["status"] == "development_gate_failed_after_all_jobs"
    assert value["formal_stage_launched"] is False
    assert value["failure_does_not_cancel_remaining_jobs_in_entered_stage"] is True


def test_complete_pipeline_runs_promotion_after_development_and_ablation(
    monkeypatch, tmp_path
):
    _install_protocol_stubs(monkeypatch)
    entered = []
    decisions = {"development": "pass", "ablation": "complete", "promotion": "fail"}

    def fake_attempt(**kwargs):
        stage = kwargs["stage"]
        entered.append(stage)
        return {"stage": stage, "matrix_complete": True, "decision": decisions[stage]}

    monkeypatch.setattr(subject, "_attempt_stage", fake_attempt)
    code, value = subject.run_all(
        device="cuda", report_path=tmp_path / "report.json"
    )
    assert code == 2
    assert entered == ["development", "ablation", "promotion"]
    assert value["status"] == "promotion_gate_failed_after_all_jobs"
    assert value["formal_test_accessed"] is False


def test_global_registry_continues_after_version_exception(tmp_path):
    calls = []

    def broken(**kwargs):
        calls.append("broken")
        raise RuntimeError("synthetic")

    def later(**kwargs):
        calls.append("later")
        return 0, {
            "status": "promotion_gate_passed",
            "accepted_jobs": 12,
            "expected_jobs": 12,
            "promotion_gate_decision": "pass",
        }

    code, value = registry_subject.run_registered(
        device="cuda",
        report_path=tmp_path / "registry.json",
        registry=(("broken", broken), ("later", later)),
    )
    assert code == 1
    assert calls == ["broken", "later"]
    assert value["failure_does_not_cancel_remaining_versions_or_jobs"] is True
    assert [row["version"] for row in value["versions"]] == ["broken", "later"]
