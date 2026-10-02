from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_all_pending_promotions_v4_9_2_3 as registry
from tools import run_all_v4_9_2_3_short_recovery as short


def _job(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def test_short_name_is_deterministic_and_compact() -> None:
    job = _job("P__cand__cross__s21__p9e99845d")
    first = short.short_recovery_name(job, 1)
    second = short.short_recovery_name(job, 1)
    assert first == second
    assert first.startswith("r1_")
    assert len(first) == 15


def test_tensorboard_prefix_stays_below_legacy_path_budget() -> None:
    job = _job("P__cand__cross__s21__p9e99845d")
    event_prefix = (
        short.short_recovery_directory(job, 1)
        / "tensorboard"
        / "SAC_1"
        / "events.out.tfevents.1787833589.DESKTOP-A06RVVE.45296.0"
    )
    assert len(str(event_prefix.resolve())) < 240


def test_configure_engine_uses_short_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(short, "SHORT_RUN_ROOT", Path("short-runs"))
    monkeypatch.setattr(short, "SHORT_LOG_ROOT", Path("short-logs"))
    short.configure_engine()
    assert short.engine.RECOVERY_RUN_ROOT == Path("short-runs")
    assert short.engine.RECOVERY_LOG_ROOT == Path("short-logs")
    assert short.engine.recovery_directory(_job("x"), 1).parent == Path(
        "short-runs"
    )


def test_registry_attempts_later_version_after_failure(tmp_path: Path) -> None:
    calls: list[str] = []

    def first(*, device: str):
        calls.append(f"first:{device}")
        return 2, {"status": "promotion_gate_failed"}

    def second(*, device: str):
        calls.append(f"second:{device}")
        return 0, {
            "status": "promotion_gate_passed",
            "accepted_jobs": 12,
            "expected_jobs": 12,
            "promotion_gate_decision": "pass",
        }

    code, report = registry.run_registered(
        device="cpu",
        report_path=tmp_path / "registry.json",
        registry=(("first", first), ("second", second)),
    )
    assert calls == ["first:cpu", "second:cpu"]
    assert code == 1
    assert len(report["versions"]) == 2

