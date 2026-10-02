"""Detach the safe newest-family promotion dispatcher from the session."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_all_pending_promotions_v4_12.py"
LOG_ROOT = ROOT / "results_promotion_automation" / "detached_logs_v4_12"
STATE_PATH = ROOT / "results_promotion_automation" / "latest_family_detached_state.json"


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args(argv)
    if os.name != "nt":
        raise RuntimeError("this detached launcher is intentionally Windows-only")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    stdout_path = LOG_ROOT / f"latest_{timestamp}.stdout.log"
    stderr_path = LOG_ROOT / f"latest_{timestamp}.stderr.log"
    command = [sys.executable, str(RUNNER), "--device", args.device]
    if args.rerun_completed:
        command.append("--rerun-completed")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NO_WINDOW
    )
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            close_fds=True,
            creationflags=flags,
        )
    state: dict[str, object] = {
        "schema_version": "topo-scene.latest-family-detached-launcher/v1",
        "started_at_local": dt.datetime.now().astimezone().isoformat(),
        "pid": int(process.pid),
        "command": command,
        "working_directory": str(ROOT),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "hidden_window": True,
        "detached_process": True,
        "newest_patch_per_version_family": True,
        "failure_does_not_cancel_remaining_versions_or_jobs": True,
    }
    _atomic_json(STATE_PATH, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

