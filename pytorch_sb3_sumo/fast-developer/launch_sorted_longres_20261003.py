"""Launch the authorized fresh ST-RT/longitudinal-residual pair.

Default is a read-only command preview. --launch reserves a NEW run root,
archives the current source, and starts a hidden supervisor. Existing run
directories are never reused or deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
import zipfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORKSPACE = PROJECT.parent.parent
METHODS = ("sac_mlp_d1_st_rt", "sac_mlp_d1_st_rt_longres_v1")
DEFAULT_ROOT = WORKSPACE / "runs" / "sortlr_1003"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def command(method, root):
    return [sys.executable, str(HERE / "train_intersection_yield_v2_d1.py"),
            "train-method", "--method", method,
            "--scenario", "intersection_sorted", "--depart-scale", "4.0",
            "--eval-traffic-split", "validation", "--max-steps", "100000",
            "--run-root", str(root), "--checkpoint-frequency", "10000",
            "--behavior-diagnostics", "--policy-shadow-probes",
            "--policy-shadow-train-raw-interval", "5000",
            "--policy-shadow-eval-max-per-episode", "4",
            "--policy-shadow-critical-eval-sample"]


def archive_source(root):
    # Keep the dirty working tree as it is. No checkout/reset/global git changes.
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT.as_posix()}", "ls-files", "-z",
         "--cached", "--others", "--exclude-standard"],
        cwd=PROJECT, check=True, capture_output=True)
    extensions = {".py", ".xml", ".sumocfg", ".yaml", ".yml", ".json", ".toml", ".md", ".ini"}
    files = set()
    for entry in result.stdout.decode("utf-8").split("\0"):
        if not entry:
            continue
        path = (PROJECT / entry).resolve()
        if path.is_file() and path.suffix.lower() in extensions and path.is_relative_to(PROJECT):
            if not any(p in {".git", "__pycache__", ".pytest_cache"} for p in path.parts):
                files.add(path)
    # These are required even if a local ignore rule hides newly added code.
    required = [HERE / "train_intersection_yield_v2_d1.py", Path(__file__),
                PROJECT / "algos/sb3_torch/longitudinal_residual_policy.py",
                PROJECT / "algos/sb3_torch/policy_shadow_probes.py",
                PROJECT / "algos/sb3_torch/replay_buffer.py",
                PROJECT / "algos/sb3_torch/sac.py", HERE / "reward_shaping_v2.py",
                PROJECT / "envs/sumo/sumo_env.py",
                HERE / "analysis/longitudinal_residual_protocol_20261003.md",
                HERE / "analysis/bootstrap_protocol_audit_20261003.md"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required source missing: {path}")
        files.add(path.resolve())
    entries = []
    archive = root / "source_archive.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(files):
            relative = path.relative_to(PROJECT).as_posix()
            payload = path.read_bytes()
            zipped.writestr(relative, payload)
            entries.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest(),
                            "bytes": len(payload)})
    provenance = {"created_at": now(), "project": str(PROJECT), "file_count": len(entries),
                  "archive": str(archive), "archive_sha256": digest(archive), "files": entries}
    write_json(root / "source_archive_manifest.json", provenance)
    return {key: value for key, value in provenance.items() if key != "files"}


def supervise(root):
    manifest_path = root / "suite_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Supervisor requires the launch-created suite manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    methods = manifest.get("methods")
    commands = manifest.get("commands")
    if (manifest.get("status") not in {"launching", "launched"}
            or Path(manifest.get("run_root", "")).resolve() != root
            or manifest.get("raw_step_budget_per_method") != 100000
            or not (root / "source_archive_manifest.json").is_file()
            or not isinstance(methods, list) or not methods
            or len(methods) != len(set(methods))
            or any(method not in METHODS for method in methods)
            or not isinstance(commands, list) or len(commands) != len(methods)):
        raise RuntimeError("Invalid supervisor launch contract")
    for method, argv in zip(methods, commands):
        if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
            raise RuntimeError("Invalid worker command in suite manifest")
        try:
            method_index = argv.index("--method")
            root_index = argv.index("--run-root")
        except ValueError as error:
            raise RuntimeError("Worker command is missing its method or run root") from error
        if (method_index + 1 >= len(argv) or root_index + 1 >= len(argv)
                or argv[method_index + 1] != method
                or Path(argv[root_index + 1]).resolve() != root):
            raise RuntimeError("Worker command does not match the suite manifest")
    with (root / "supervisor.lock").open("x", encoding="utf-8") as lock:
        json.dump({"pid": os.getpid(), "started_at": now()}, lock)
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    status = {"started_at": now(), "status": "running", "supervisor_pid": os.getpid(),
              "max_cuda_workers": len(methods), "workers": []}
    handles = []
    workers = []
    try:
        for method, argv in zip(methods, commands):
            stdout = (root / f"{method}.stdout.log").open("xb")
            stderr = (root / f"{method}.stderr.log").open("xb")
            handles.extend([stdout, stderr])
            process = subprocess.Popen(argv, cwd=PROJECT, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            record = {"method": method, "pid": process.pid, "command": argv,
                      "started_at": now(), "exit_code": None,
                      "stdout": str(root / f"{method}.stdout.log"),
                      "stderr": str(root / f"{method}.stderr.log")}
            workers.append((process, record))
            status["workers"].append(record)
            write_json(root / "launcher_status.json", status)
        while True:
            live = False
            for process, record in workers:
                code = process.poll()
                if code is None:
                    live = True
                elif record["exit_code"] is None:
                    record.update(exit_code=code, finished_at=now())
                    write_json(root / "launcher_status.json", status)
            if not live:
                break
            time.sleep(10)
        status.update(status="completed" if all(r["exit_code"] == 0 for _, r in workers) else "failed",
                      finished_at=now())
        write_json(root / "launcher_status.json", status)
    except BaseException as error:
        status.update(status="supervisor_error", error=repr(error), finished_at=now())
        write_json(root / "launcher_status.json", status)
        # Do not silently kill an already-started formal worker on supervisor error.
        raise
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--method", action="append", choices=METHODS,
                        help="select one method; repeat to select both")
    parser.add_argument("--methods",
                        help="comma-separated method list; defaults to both methods")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.supervise:
        supervise(root)
        return
    if args.method is not None and args.methods is not None:
        parser.error("use either --method or --methods, not both")
    if args.methods is None:
        methods = tuple(args.method) if args.method is not None else METHODS
    else:
        methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
        unknown = [method for method in methods if method not in METHODS]
        if unknown:
            parser.error(f"unsupported method(s): {', '.join(unknown)}")
    if not methods:
        parser.error("select at least one method")
    if len(methods) != len(set(methods)):
        parser.error("method selection contains duplicates")
    preview = {"mode": "preview", "run_root": str(root), "methods": list(methods),
               "commands": [command(method, root) for method in methods]}
    if not args.launch:
        print(json.dumps(preview, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to silently launch on CPU")
    root.mkdir(parents=True, exist_ok=False)
    try:
        archive = archive_source(root)
        manifest = {"created_at": now(), "status": "launching", "run_root": str(root),
                    "methods": list(methods), "scene": "intersection_sorted_depart4p0",
                    "scenario": "intersection_sorted", "depart_scale": 4.0,
                    "training_seed": 0, "fresh": True, "resume": False,
                    "raw_step_budget_per_method": 100000, "max_cuda_workers": len(methods),
                    "cuda_visible_devices": "0", "gpu": torch.cuda.get_device_name(0),
                    "python": sys.executable, "torch": torch.__version__, "smoke": False,
                    "evaluation": {"split": "validation", "episodes": 100,
                                   "seed_start": 10000, "seed_end": 10099,
                                   "checkpoint": "final_model.zip"},
                    "reward_schema": "environment_step_reward_v2",
                    "bootstrap_protocol": "unchanged released-source-compatible single_gamma n-step; timeout bootstrap",
                    "checkpoint_frequency_raw_steps": 10000,
                    "policy_shadow": {"train_raw_interval": 5000, "eval_max_states_per_episode": 4,
                                      "extra_environment_steps": 0},
                    "commands": preview["commands"], "source_archive": archive}
        write_json(root / "suite_manifest.json", manifest)
        log_out = (root / "supervisor.stdout.log").open("xb")
        log_err = (root / "supervisor.stderr.log").open("xb")
        try:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--supervise",
                                        "--run-root", str(root)], cwd=PROJECT,
                                       stdin=subprocess.DEVNULL, stdout=log_out, stderr=log_err,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                                       | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        finally:
            log_out.close()
            log_err.close()
        manifest.update(status="launched", supervisor_pid=process.pid, launched_at=now())
        write_json(root / "suite_manifest.json", manifest)
        print(json.dumps({"status": "launched", "run_root": str(root), "supervisor_pid": process.pid}))
    except BaseException as error:
        write_json(root / "launch_error.json", {"time": now(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
