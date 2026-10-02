from __future__ import annotations

from pathlib import Path

from tools import run_all_pending_promotions_v4_11 as discovery
from tools import run_all_v4_12_pipeline as pipeline


def test_pipeline_is_discoverable_as_v4_12() -> None:
    versions = [version for version, _, _ in discovery.discover_pipelines()]
    assert "v4.12" in versions


def test_attempt_stage_summarizes_and_gates_after_execution_exception(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pipeline.runner,
        "execute_jobs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("job failure")),
    )
    monkeypatch.setattr(
        pipeline.runner,
        "summarize_stage",
        lambda *args, **kwargs: {
            "accepted_runs": 3,
            "expected_runs": 4,
            "complete": False,
            "aggregate": [{"scenario": "cross"}],
        },
    )
    value = pipeline._attempt_stage(
        contract={},
        digest="digest",
        hashes={},
        stage="development",
        device="cpu",
        gate=lambda *args: {"decision": "fail"},
    )
    assert value["execution_error"] == "RuntimeError: job failure"
    assert value["accepted_jobs"] == 3
    assert value["decision"] == "fail"


def test_run_all_stops_only_at_scientific_gate_after_stage_summary(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pipeline.runner, "load_contract", lambda: {})
    monkeypatch.setattr(pipeline.runner, "validate_contract", lambda value: value)
    monkeypatch.setattr(pipeline.runner, "protocol_hashes", lambda: {"h": "x"})
    monkeypatch.setattr(pipeline.runner, "_sha256", lambda path: "digest")
    calls: list[str] = []

    def attempt(**kwargs):
        calls.append(kwargs["stage"])
        return {
            "stage": kwargs["stage"],
            "accepted_jobs": 4,
            "expected_jobs": 4,
            "matrix_complete": True,
            "decision": "fail",
            "aggregate": [],
        }

    monkeypatch.setattr(pipeline, "_attempt_stage", attempt)
    code, report = pipeline.run_all(
        device="cpu", report_path=tmp_path / "report.json"
    )
    assert code == 2
    assert calls == ["development"]
    assert report["status"] == "development_gate_failed_after_all_jobs"
    assert report["failure_does_not_cancel_remaining_jobs_in_entered_stage"] is True
