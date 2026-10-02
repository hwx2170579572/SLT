from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_all_pending_promotions_v4_9_2_2 as registry
from tools import run_all_v4_9_2_2_recovery as recovery


def _job(name: str = "job") -> SimpleNamespace:
    return SimpleNamespace(name=name)


def test_recovery_name_is_versioned_and_bounded() -> None:
    assert recovery.recovery_name(_job("logical"), 1) == "logical__recovery_r1"
    with pytest.raises(recovery.V4922RecoveryError):
        recovery.recovery_name(_job(), 0)


def test_next_target_preserves_existing_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    monkeypatch.setattr(recovery.frozen, "run_directory", lambda _job: canonical)
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
        assert command[command.index(flag) + 1] == original[original.index(flag) + 1]


def test_execute_all_continues_after_child_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = [_job("first"), _job("second")]
    calls: list[str] = []
    monkeypatch.setattr(
        recovery.frozen,
        "jobs_for_stage",
        lambda *_args: jobs,
    )
    monkeypatch.setattr(
        recovery,
        "selected_source",
        lambda *_args: (None, None, []),
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
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(recovery.subprocess, "run", run)
    monkeypatch.setattr(
        recovery,
        "accepted_root_reason",
        lambda *_args: (False, "failed"),
    )
    attempts = recovery.execute_all({}, "digest", {}, device="cpu")
    assert calls == ["first", "second"]
    assert [row["status"] for row in attempts] == ["failed", "failed"]


def test_registry_continues_after_version_exception(tmp_path: Path) -> None:
    calls: list[str] = []

    def first(*, device: str):
        calls.append(f"first:{device}")
        raise RuntimeError("boom")

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

