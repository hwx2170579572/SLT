from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_all_v4_11_1_pipeline as recovery


def _job(name: str = "job", stage: str = "promotion") -> SimpleNamespace:
    return SimpleNamespace(name=name, stage=stage)


def test_recovery_name_is_versioned_and_bounded() -> None:
    assert recovery.recovery_name(_job("logical"), 1) == (
        "logical__recovery_r1"
    )
    with pytest.raises(recovery.V4111RecoveryError):
        recovery.recovery_name(_job(), 0)


def test_next_target_preserves_existing_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    monkeypatch.setattr(
        recovery.frozen, "run_directory", lambda _job: canonical
    )
    monkeypatch.setattr(
        recovery,
        "recovery_directory",
        lambda _job, attempt: tmp_path / f"recovery_{attempt}",
    )
    kind, root, attempt = recovery._next_target(_job())
    assert kind == "recovery_r1"
    assert root == tmp_path / "recovery_1"
    assert attempt == 1
    assert canonical.is_dir()


def test_command_changes_only_output_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = [
        "python",
        "train.py",
        "--scenario",
        "cross",
        "--output-dir",
        "old-root",
        "--model-name",
        "old-name",
        "--seed",
        "21",
    ]
    monkeypatch.setattr(
        recovery.frozen,
        "command_for",
        lambda *_args, **_kwargs: list(original),
    )
    target = tmp_path / "new-root" / "new-name"
    command = recovery.command_for_target(
        _job(), {}, device="cuda", target_root=target
    )
    assert command[command.index("--output-dir") + 1] == str(
        target.parent.resolve()
    )
    assert command[command.index("--model-name") + 1] == target.name
    for flag in ("--scenario", "--seed"):
        assert command[command.index(flag) + 1] == original[
            original.index(flag) + 1
        ]


def test_source_view_changes_only_selected_promotion_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recovery.frozen,
        "run_directory",
        lambda job: tmp_path / f"canonical_{job.name}",
    )
    selected = tmp_path / "selected"
    promotion = _job("promotion", "promotion")
    development = _job("development", "development")
    with recovery._source_view({promotion.name: selected}):
        assert recovery.frozen.run_directory(promotion) == selected
        assert recovery.frozen.run_directory(development) == (
            tmp_path / "canonical_development"
        )
    assert recovery.frozen.run_directory(promotion) == (
        tmp_path / "canonical_promotion"
    )


def test_execute_all_continues_after_child_failure_and_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = [_job("first"), _job("second")]
    calls: list[str] = []
    monkeypatch.setattr(
        recovery.frozen, "jobs_for_stage", lambda *_args: jobs
    )
    monkeypatch.setattr(
        recovery, "selected_source", lambda *_args: (None, None, [])
    )
    monkeypatch.setattr(
        recovery,
        "_next_target",
        lambda job: ("canonical", tmp_path / job.name, None),
    )
    monkeypatch.setattr(
        recovery,
        "command_for_target",
        lambda job, *_args, **_kwargs: ["fake", job.name],
    )
    monkeypatch.setattr(recovery, "RECOVERY_LOG_ROOT", tmp_path / "logs")

    def run(command, **_kwargs):
        calls.append(command[-1])
        if command[-1] == "first":
            raise RuntimeError("synthetic launcher failure")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(recovery.subprocess, "run", run)
    monkeypatch.setattr(
        recovery,
        "accepted_root_reason",
        lambda *_args: (False, "failed"),
    )
    attempts = recovery.execute_all({}, "digest", {}, device="cpu")
    assert calls == ["first", "second"]
    assert [row["status"] for row in attempts] == [
        "launcher_exception",
        "failed",
    ]


def test_recovery_pipeline_is_discoverable_by_global_registry() -> None:
    from tools import run_all_pending_promotions_v4_11 as registry

    versions = [version for version, _, _ in registry.discover_pipelines()]
    assert "v4.11.1" in versions
