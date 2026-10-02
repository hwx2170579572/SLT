from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import wait_then_finalize_v4_9_2_4 as finalizer


def test_analysis_commands_use_current_interpreter_and_full_attribution() -> None:
    commands = finalizer.analysis_commands()
    assert [name for name, _ in commands] == [
        "post_promotion_kinematic_model_audit",
        "complete_six_pair_model_attribution",
    ]
    assert all(command[0] == finalizer.sys.executable for _, command in commands)
    attribution = commands[1][1]
    assert "--allow-partial" not in attribution
    assert attribution[1].endswith("attribute_v4_9_2_promotion.py")


def test_all_analyses_run_after_an_earlier_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes = iter((True, False))
    sleeps: list[float] = []
    calls: list[list[str]] = []
    returns = iter((1, 0))
    monkeypatch.setattr(finalizer.time, "sleep", lambda value: sleeps.append(value))

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=next(returns), stdout="out", stderr="err"
        )

    code, report = finalizer.wait_then_finalize(
        wait_pid=123,
        wait_create_time=456.0,
        poll_seconds=0.5,
        max_wait_hours=1.0,
        report_path=tmp_path / "report.json",
        process_probe=lambda *_: next(probes),
        command_runner=run,
        commands=(("first", ["first"]), ("second", ["second"])),
        snapshot_builder=lambda: {
            "ready_for_scientific_decision": True,
            "formal_test_accessed": False,
        },
    )
    assert sleeps == [0.5]
    assert calls == [["first"], ["second"]]
    assert [row["returncode"] for row in report["analyses"]] == [1, 0]
    assert report["failure_does_not_cancel_later_analyses"] is True
    assert code == 1


def test_evidence_snapshot_requires_twelve_jobs_and_six_pairs(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "engine.json"
    attribution = tmp_path / "attribution.json"
    formal = tmp_path / "formal"
    engine.write_text(
        json.dumps(
            {
                "promotion_matrix_complete": True,
                "accepted_jobs": 12,
                "expected_jobs": 12,
                "promotion_gate_decision": "fail",
            }
        ),
        encoding="utf-8",
    )
    attribution.write_text(
        json.dumps(
            {
                "decision_allowed": True,
                "partial_matrix": False,
                "pair_count": 6,
                "expected_pair_count": 6,
            }
        ),
        encoding="utf-8",
    )
    snapshot = finalizer.evidence_snapshot(
        engine_report_path=engine,
        attribution_path=attribution,
        formal_root=formal,
    )
    assert snapshot["promotion_matrix_complete"] is True
    assert snapshot["attribution_complete"] is True
    assert snapshot["ready_for_scientific_decision"] is True
    assert snapshot["formal_test_accessed"] is False
    assert snapshot["next_stage"] == (
        "ccf_idea_optimizer_model_only_v4_10_iteration"
    )


def test_snapshot_failure_is_aggregated_after_all_analyses(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    def broken_snapshot():
        raise json.JSONDecodeError("bad evidence", "{", 1)

    code, report = finalizer.wait_then_finalize(
        wait_pid=123,
        wait_create_time=456.0,
        poll_seconds=0.5,
        max_wait_hours=1.0,
        report_path=tmp_path / "report.json",
        process_probe=lambda *_: False,
        command_runner=run,
        commands=(("first", ["first"]), ("second", ["second"])),
        snapshot_builder=broken_snapshot,
    )
    assert calls == [["first"], ["second"]]
    assert report["analyses_passed"] is True
    assert report["evidence"]["ready_for_scientific_decision"] is False
    assert report["evidence_snapshot_error"].startswith("JSONDecodeError:")
    assert report["finalization_passed"] is False
    assert code == 1


def test_malformed_evidence_is_reported_as_not_ready(tmp_path: Path) -> None:
    engine = tmp_path / "engine.json"
    attribution = tmp_path / "attribution.json"
    engine.write_text("{", encoding="utf-8")
    attribution.write_text("[]", encoding="utf-8")

    snapshot = finalizer.evidence_snapshot(
        engine_report_path=engine,
        attribution_path=attribution,
        formal_root=tmp_path / "formal",
    )
    assert snapshot["ready_for_scientific_decision"] is False
    assert snapshot["engine_report_error"].startswith("JSONDecodeError:")
    assert snapshot["attribution_error"].startswith("TypeError:")
