"""Prepare and launch the three-scene hold35k vs MST+SLT training screen.

Training is GPU-bound and memory-heavy (six 50k-step cells).  The controller
runs two worker subprocesses at a time (configurable), each training one
(method, scenario) cell with a checkpoint every 2000 raw steps.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import threading
import time

from .common import (
    METHODS,
    RESULT_ROOT,
    SCENES,
    lock,
    read,
    write,
)


def controller(workers=2, smoke=False):
    import psutil
    import torch

    with lock(RESULT_ROOT / ("smoke/controller.lock" if smoke else "controller.lock")):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; this plan has no CPU fallback")
        torch.empty(1, device="cuda")
        torch.cuda.empty_cache()
        print(
            f"{workers} GPU worker processes on {torch.cuda.get_device_name(0)}; "
            f"free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB",
            flush=True,
        )
        jobs = [
            (method, scene) for method in METHODS for scene in SCENES
        ]
        start = time.time()
        output = RESULT_ROOT / ("smoke/train" if smoke else "train")
        log_dir = output / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        running, done = {}, {}
        stop_requested = threading.Event()

        def execute(job):
            if stop_requested.is_set():
                return dict(exit_code=-1, log=None, cancelled=True)
            method, scene = job
            name = f"{method}__{scene}"
            command = [
                sys.executable,
                "-m",
                "tools.three_scene_hold35k_vs_mst_v1.cli",
                "cell",
                "--method",
                method,
                "--scene",
                scene,
            ]
            if smoke:
                command.append("--smoke")
            stamp = time.strftime("%Y%m%d_%H%M%S")
            log = log_dir / f"{name}_{stamp}.log"
            with log.open("w", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    command,
                    cwd=RESULT_ROOT.parent,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                running[name] = process
                returncode = process.wait()
                running.pop(name, None)
            return dict(exit_code=returncode, log=str(log), completed_at=time.time())

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures = {pool.submit(execute, job): job for job in jobs}
        try:
            for future in concurrent.futures.as_completed(futures):
                method, scene = futures[future]
                name = f"{method}__{scene}"
                done[name] = future.result()
                print(name, done[name]["exit_code"], flush=True)
                write(
                    output / "controller_progress.json",
                    dict(done=done, total=len(jobs), updated_at=time.time()),
                )
        except BaseException:
            stop_requested.set()
            for future in futures:
                future.cancel()
            for process in list(running.values()):
                try:
                    parent = psutil.Process(process.pid)
                    for child in parent.children(recursive=True):
                        child.terminate()
                    parent.terminate()
                except psutil.NoSuchProcess:
                    pass
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        receipt = dict(
            passed=all(v["exit_code"] == 0 for v in done.values()),
            jobs=done,
            workers=workers,
            smoke=smoke,
            wall_seconds=time.time() - start,
        )
        write(output / "controller.json", receipt)
        if not receipt["passed"]:
            raise RuntimeError(
                "Some jobs failed; inspect logs. Partial training was not restarted."
            )
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("cell", "run", "smoke", "status"),
    )
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.command == "cell":
        if not args.method or not args.scene:
            parser.error("cell requires --method and --scene")
        from .jobs import run_cell

        print(run_cell(args.method, args.scene, smoke=args.smoke))
    elif args.command == "run":
        controller(workers=args.workers, smoke=False)
    elif args.command == "smoke":
        controller(workers=args.workers, smoke=True)
    elif args.command == "status":
        for root in (RESULT_ROOT / "train", RESULT_ROOT / "smoke" / "train"):
            for name in ("controller_progress.json", "controller.json"):
                path = root / name
                print(path, read(path) if path.exists() else "not started")


if __name__ == "__main__":
    main()
