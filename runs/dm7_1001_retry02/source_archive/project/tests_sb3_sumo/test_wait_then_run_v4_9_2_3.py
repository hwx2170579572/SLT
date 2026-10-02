from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import wait_then_run_v4_9_2_3 as handoff


def test_recovery_command_uses_current_interpreter() -> None:
    command = handoff.recovery_command(device="cuda")
    assert command[0] == handoff.sys.executable
    assert command[-2:] == ["--device", "cuda"]
    assert command[1].endswith("run_all_pending_promotions_v4_9_2_3.py")


def test_handoff_waits_then_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes = iter((True, True, False))
    sleeps: list[float] = []
    commands: list[list[str]] = []
    monkeypatch.setattr(handoff.time, "sleep", lambda value: sleeps.append(value))

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(handoff.subprocess, "run", run)
    code, report = handoff.wait_then_run(
        wait_pid=123,
        wait_create_time=456.0,
        device="cpu",
        poll_seconds=0.5,
        max_wait_hours=1.0,
        report_path=tmp_path / "report.json",
        process_probe=lambda *_: next(probes),
    )
    assert sleeps == [0.5, 0.5]
    assert len(commands) == 1
    assert commands[0][-1] == "cpu"
    assert code == 2
    assert report["short_path_recovery_launched"] is True

