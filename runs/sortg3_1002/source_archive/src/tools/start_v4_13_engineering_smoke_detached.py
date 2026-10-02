"""Launch the v4.13 real-SUMO engineering smoke independently of Codex."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_v4_13_engineering_smoke_supervised.py"
STATE = ROOT / "results_topo_v4_13_dev" / "engineering" / "smoke_supervisor_breakaway_state.json"
STDOUT = ROOT / "results_topo_v4_13_dev" / "engineering" / "smoke_supervisor_breakaway.stdout.log"
STDERR = ROOT / "results_topo_v4_13_dev" / "engineering" / "smoke_supervisor_breakaway.stderr.log"


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    if os.name != "nt":
        raise RuntimeError("the detached smoke launcher is Windows-only")
    command = [sys.executable, str(RUNNER)]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NO_WINDOW
        | subprocess.CREATE_BREAKAWAY_FROM_JOB
    )
    with STDOUT.open("wb") as stdout, STDERR.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=flags,
        )
    value: dict[str, object] = {
        "schema_version": "topo-scene-v4.13.detached-engineering-smoke/v1",
        "started_at_local": dt.datetime.now().astimezone().isoformat(),
        "pid": int(process.pid),
        "command": command,
        "supervised_attempt_directories": [
            str(ROOT / "r413s" / attempt / "m")
            for attempt in ("e7", "e8", "e9", "e10")
        ],
        "stdout_log": str(STDOUT),
        "stderr_log": str(STDERR),
        "detached_process": True,
        "hidden_window": True,
        "survives_codex_session_interrupt": True,
        "breakaway_from_codex_job": True,
        "formal_test_accessed": False,
    }
    _write_json(STATE, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
