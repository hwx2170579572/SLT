"""Guarded, independent two-worker M0/M1 development launcher."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scene_event.acceptance import validate_receipt
from scene_event.provenance import save_json, sha256, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--acceptance-receipt", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    receipt = args.acceptance_receipt.resolve()
    methods = ("sac_scene_dualgraph_v1", "sac_scene_eventgraph_cv_v1")
    for method in methods:
        validate_receipt(receipt, method)
    if root.exists():
        raise FileExistsError("Do not reuse or relaunch an existing pair root: " + str(root))
    cwd = Path(__file__).resolve().parent
    commands = {method: [sys.executable, str(cwd / "train_scene_event_v1.py"), "--mode", "train",
                        "--method", method, "--run-root", str(root / method),
                        "--seed", "0", "--device", "cuda", "--acceptance-receipt", str(receipt)]
                for method in methods}
    if not args.launch:
        print(json.dumps({"check_only": True, "launch_authorized_by_this_command": False,
                          "commands": commands}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=False)
    # Exclusive creation is the durable duplicate-launch guard, not a PID that
    # could later be recycled. Failed launch records remain intact.
    with (root / "launch.lock").open("x", encoding="utf-8") as lock:
        lock.write(f"pid={os.getpid()} utc={utc_now()}\n")
    receipt_data = {"launched_utc": utc_now(), "interpreter": sys.executable,
                    "acceptance_receipt": str(receipt), "acceptance_sha256": sha256(receipt),
                    "workers": [], "state": "launching"}
    save_json(root / "pair_launch_receipt.json", receipt_data)
    for method in methods:
        stdout_path, stderr_path = root / f"{method}.stdout.log", root / f"{method}.stderr.log"
        with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(commands[method], cwd=cwd, stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), close_fds=True)
        receipt_data["workers"].append({"method": method, "pid": process.pid,
            "root": str(root / method), "command": commands[method],
            "stdout": str(stdout_path), "stderr": str(stderr_path)})
        save_json(root / "pair_launch_receipt.json", receipt_data)
    receipt_data["state"] = "dispatched_not_yet_verified"
    save_json(root / "pair_launch_receipt.json", receipt_data)
    print(json.dumps(receipt_data, indent=2))


if __name__ == "__main__":
    main()
