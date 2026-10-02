from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_all_v4_11_2_pipeline as short


def _job(name: str = "P__tg__cross__s21__pc71e98e7") -> SimpleNamespace:
    return SimpleNamespace(name=name, stage="promotion")


def test_short_name_is_deterministic_compact_and_versioned() -> None:
    first = short.short_recovery_name(_job(), 1)
    second = short.short_recovery_name(_job(), 1)
    assert first == second
    assert first.startswith("r1_")
    assert len(first) == 13
    with pytest.raises(short.engine.V4111RecoveryError):
        short.short_recovery_name(_job(), 0)


def test_tensorboard_event_path_stays_below_legacy_windows_budget() -> None:
    event_path = (
        short.short_recovery_directory(_job(), 1)
        / "tensorboard"
        / "SAC_1"
        / "events.out.tfevents.1787833589.DESKTOP-A06RVVE.45296.0"
    )
    assert len(str(event_path.resolve())) < 240


def test_configure_engine_changes_only_engineering_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in short._ENGINE_CONFIGURATION_FIELDS:
        monkeypatch.setattr(short.engine, name, getattr(short.engine, name))
    run_root = tmp_path / "r"
    log_root = tmp_path / "l"
    result_root = tmp_path / "evidence"
    monkeypatch.setattr(short, "SHORT_RUN_ROOT", run_root)
    monkeypatch.setattr(short, "SHORT_LOG_ROOT", log_root)
    monkeypatch.setattr(short, "RESULT_ROOT", result_root)
    short.configure_engine()
    assert short.engine.RECOVERY_RUN_ROOT == run_root
    assert short.engine.RECOVERY_LOG_ROOT == log_root
    assert short.engine.RECOVERY_KIND == short.KIND
    assert short.engine.RECOVERY_VERSION == short.VERSION
    assert short.engine.recovery_directory(_job(), 1).parent == run_root


def test_configured_engine_restores_shared_recovery_engine() -> None:
    original = {
        name: getattr(short.engine, name)
        for name in short._ENGINE_CONFIGURATION_FIELDS
    }
    with short.configured_engine():
        assert short.engine.RECOVERY_VERSION == short.VERSION
        assert short.engine.recovery_name(_job(), 1).startswith("r1_")
    assert all(
        getattr(short.engine, name) is value
        if callable(value)
        else getattr(short.engine, name) == value
        for name, value in original.items()
    )


def test_wrapper_preserves_scientific_contract_and_legacy_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = tmp_path / "legacy"
    failed = legacy / "failed-r1"
    failed.mkdir(parents=True)
    result_root = tmp_path / "result"
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(short, "LEGACY_FAILED_RECOVERY_ROOT", legacy)
    monkeypatch.setattr(short, "RESULT_ROOT", result_root)
    monkeypatch.setattr(short, "SHORT_RUN_ROOT", tmp_path / "short-runs")
    monkeypatch.setattr(short, "SHORT_LOG_ROOT", tmp_path / "short-logs")

    def fake_run_all(**_kwargs):
        return 1, {
            "status": "promotion_incomplete_after_all_attempts",
            "accepted_jobs": 1,
            "expected_jobs": 12,
            "promotion_matrix_complete": False,
            "formal_test_unlocked": False,
        }

    monkeypatch.setattr(short.engine, "run_all", fake_run_all)
    code, report = short.run_all(
        device="cpu", report_path=report_path, dry_run=True
    )
    assert code == 1
    assert report["scientific_model_changed"] is False
    assert report["training_protocol_changed"] is False
    assert report["experiment_contract_changed"] is False
    assert report["failure_does_not_cancel_remaining_jobs"] is True
    assert report["accepted_jobs"] == 1
    assert report["expected_jobs"] == 12
    assert report["legacy_failed_recovery_attempts_preserved"] == [
        str(failed.resolve())
    ]


def test_pipeline_is_discoverable_as_v4_11_2() -> None:
    from tools import run_all_pending_promotions_v4_11 as registry

    versions = [version for version, _, _ in registry.discover_pipelines()]
    assert "v4.11.2" in versions
