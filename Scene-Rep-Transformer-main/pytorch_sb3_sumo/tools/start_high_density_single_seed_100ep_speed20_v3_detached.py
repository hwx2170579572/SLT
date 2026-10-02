"""Start the speed20-v3 matrix in a detached, hidden Windows process."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    PROJECT_ROOT
    / "tools"
    / "run_high_density_single_seed_100ep_speed20_v3.py"
)
RESULT_ROOT = PROJECT_ROOT / "results_hd_ss100_s20_v3"
LAUNCH_ROOT = RESULT_ROOT / "launcher_logs_speed20_v3"
STATE_PATH = RESULT_ROOT / "detached_launcher_state_speed20_v3.json"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if os.name != "nt":
        raise RuntimeError("this detached launcher is intentionally Windows-only")

    LAUNCH_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    stdout_path = LAUNCH_ROOT / f"matrix_{timestamp}.stdout.log"
    stderr_path = LAUNCH_ROOT / f"matrix_{timestamp}.stderr.log"
    command = [
        sys.executable,
        str(RUNNER),
        "run",
        "--profile",
        "comparison",
        "--device",
        args.device,
        "--workers",
        str(args.workers),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    creation_flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NO_WINDOW
    )
    with stdout_path.open("wb") as stdout_handle, stderr_path.open(
        "wb"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            close_fds=True,
            creationflags=creation_flags,
        )
    state: dict[str, object] = {
        "schema_version": "speed20-v3-detached-launcher/v1",
        "started_at_local": dt.datetime.now().astimezone().isoformat(),
        "pid": int(process.pid),
        "command": command,
        "working_directory": str(PROJECT_ROOT),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "result_root": str(RESULT_ROOT),
        "hidden_window": True,
        "detached_process": True,
    }
    _atomic_json(STATE_PATH, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
