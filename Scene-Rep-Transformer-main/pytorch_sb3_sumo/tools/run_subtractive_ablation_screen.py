"""Serial screen runner for one subtractive-ablation arm (idempotent, resumable).

Run two of these in parallel (one per arm) to keep the two arms on separate workers
while each arm serially completes its scenario list. A cell whose status.json is
already "completed" is skipped, so an interrupted run resumes from the first missing
cell instead of redoing finished work.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    "minus_horizon": ["cross", "carla"],
    "minus_factorized": ["cross", "carla"],
}

SCREEN_ROOT = ROOT / "results_subtractive_ablation" / "screen"


def cell_done(arm: str, scenario: str) -> bool:
    status = SCREEN_ROOT / f"{arm}__{scenario}__lr_half__seed0" / "status.json"
    if not status.exists():
        return False
    try:
        return json.loads(status.read_text(encoding="utf-8")).get("status") == "completed"
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scenarios", nargs="*", default=None)
    args = parser.parse_args()

    scenarios = args.scenarios or ARMS[args.arm]
    for sc in scenarios:
        if cell_done(args.arm, sc):
            print(f"[screen] SKIP {args.arm} x {sc} (already completed)", flush=True)
            continue
        argv = [
            sys.executable,
            str(ROOT / "tools" / "run_subtractive_ablation_v4_8.py"),
            "--arm", args.arm,
            "--scenario", sc,
            "--device", args.device,
        ]
        print(f"[screen] RUN {args.arm} x {sc}", flush=True)
        rc = subprocess.call(argv, cwd=ROOT)
        if rc != 0:
            print(f"[screen] FAIL {args.arm} x {sc} rc={rc}", flush=True)
            return rc
    print(f"[screen] {args.arm} ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
