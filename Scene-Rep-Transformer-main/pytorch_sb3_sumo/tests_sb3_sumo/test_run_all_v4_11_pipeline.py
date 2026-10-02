from __future__ import annotations

import json

from tools import run_all_pending_promotions_v4_11 as registry_subject
from tools import run_all_v4_11_pipeline as subject


def _install_protocol_stubs(monkeypatch) -> None:
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
) -> None:
    _install_protocol_stubs(monkeypatch)
    entered: list[str] = []

    def fake_attempt(**kwargs):
        entered.append(kwargs["stage"])
        return {
            "stage": kwargs["stage"],
            "accepted_jobs": 4,
            "expected_jobs": 4,
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
    assert value["accepted_jobs"] == value["expected_jobs"] == 4
    assert value["formal_stage_launched"] is False
    assert value["failure_does_not_cancel_remaining_jobs_in_entered_stage"] is True


def test_complete_pipeline_enters_promotion_only_after_prior_gates(
    monkeypatch, tmp_path
) -> None:
    _install_protocol_stubs(monkeypatch)
    entered: list[str] = []
    decisions = {
        "development": "pass",
        "ablation": "complete",
        "promotion": "fail",
    }

    def fake_attempt(**kwargs):
        stage = kwargs["stage"]
        entered.append(stage)
        expected = {"development": 4, "ablation": 2, "promotion": 12}[stage]
        return {
            "stage": stage,
            "accepted_jobs": expected,
            "expected_jobs": expected,
            "matrix_complete": True,
            "decision": decisions[stage],
        }

    monkeypatch.setattr(subject, "_attempt_stage", fake_attempt)
    code, value = subject.run_all(
        device="cuda", report_path=tmp_path / "report.json"
    )
    assert code == 2
    assert entered == ["development", "ablation", "promotion"]
    assert value["status"] == "promotion_gate_failed_after_all_jobs"
    assert value["accepted_jobs"] == value["expected_jobs"] == 18
    assert value["promotion_gate_decision"] == "fail"
    assert value["formal_test_accessed"] is False


def test_discovery_finds_versioned_current_and_future_convention() -> None:
    discovered = registry_subject.discover_pipelines()
    versions = [version for version, _, _ in discovered]
    assert "v4.10" in versions
    assert "v4.11" in versions
    assert versions == sorted(
        versions, key=lambda value: tuple(int(part) for part in value[1:].split("."))
    )


def test_global_registry_continues_after_version_exception(tmp_path) -> None:
    calls: list[str] = []

    def broken(**kwargs):
        calls.append("broken")
        raise RuntimeError("synthetic")

    def later(**kwargs):
        calls.append("later")
        report = {
            "status": "promotion_gate_passed",
            "accepted_jobs": 18,
            "expected_jobs": 18,
            "promotion_gate_decision": "pass",
        }
        kwargs["report_path"].write_text(json.dumps(report), encoding="utf-8")
        return 0, report

    code, value = registry_subject.run_discovered(
        device="cuda",
        report_path=tmp_path / "registry.json",
        pipelines=(
            ("v9.1", broken, tmp_path / "broken.json"),
            ("v9.2", later, tmp_path / "later.json"),
        ),
    )
    assert code == 1
    assert calls == ["broken", "later"]
    assert value["failure_does_not_cancel_remaining_versions_or_jobs"] is True
    assert [row["version"] for row in value["versions"]] == ["v9.1", "v9.2"]
    assert value["aggregate"]["versions_with_errors_or_failed_gates"] == 1


def test_completed_version_is_summarized_without_rerun(tmp_path) -> None:
    calls: list[str] = []
    version_report = tmp_path / "completed.json"
    version_report.write_text(
        json.dumps(
            {
                "status": "promotion_gate_failed_after_all_jobs",
                "accepted_jobs": 12,
                "expected_jobs": 12,
                "promotion_gate_decision": "fail",
            }
        ),
        encoding="utf-8",
    )

    def must_not_run(**kwargs):
        calls.append("unexpected")
        return 0, {}

    code, value = registry_subject.run_discovered(
        device="cuda",
        report_path=tmp_path / "registry.json",
        pipelines=(("v8.0", must_not_run, version_report),),
    )
    assert code == 1
    assert calls == []
    assert value["versions"][0]["execution"] == (
        "skipped_completed_immutable_version"
    )
    assert value["aggregate"]["accepted_jobs_in_entered_stages"] == 12
