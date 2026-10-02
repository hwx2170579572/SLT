"""Archive and launch the requested seven-method DARRL medium experiment queue."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT = Path(__file__).resolve().parent.parent
WORKSPACE = PROJECT.parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
SCENARIO = "intersection_random_darrl_medium_v1"
METHODS = (
    "mst_slt",
    "sac_mlp_d1_st",
    "sac_mlp_d1_st_rt",
    "sac_mlp_d1_st_rt_topo",
    "sac_mlp_d1_st_rt_topo_routeaware_v1",
    "sac_mlp_d1_st_rt_3slot",
    "sac_mlp_d1_st_rt_topo_3slot",
)
SKIP_DIRECTORIES = {
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules",
    "runs", "results", "outputs", "output", "logs", "checkpoints", "tensorboard",
    "analysis", "original_scenarios_v1",
}
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml"}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    import time

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.1)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_source(run_root):
    archive = run_root / "source_archive"
    archive.mkdir()
    sources = []
    for directory, children, filenames in os.walk(PROJECT):
        current = Path(directory)
        children[:] = sorted(
            name for name in children
            if name not in SKIP_DIRECTORIES and "__intersection" not in name
        )
        if (current / "training_complete.json").is_file() or (current / "evaluation_results.json").is_file():
            children[:] = []
            continue
        for filename in sorted(filenames):
            source = current / filename
            if source.suffix.lower() in SOURCE_SUFFIXES:
                sources.append((source, Path("project") / source.relative_to(PROJECT)))
    scenario = PROJECT / "envs" / "sumo" / "original_scenarios_v1" / SCENARIO
    for name in ("ego.rou.xml", "map.net.xml", "scenario_manifest.json", "traffic/traffic_00000.rou.xml"):
        source = scenario / name
        if not source.is_file():
            raise FileNotFoundError("Scenario asset missing before launch: " + str(source))
        sources.append((source, Path("project") / source.relative_to(PROJECT)))
    # Random traffic draws vehicle-type distributions from these original
    # templates; keep the input files as well as their hashes.
    sorted_source = scenario.parent / "intersection_sorted"
    templates = sorted((sorted_source / "traffic").glob("*.rou.xml"))
    if not templates:
        raise FileNotFoundError("Original traffic templates are missing: " + str(sorted_source))
    for source in templates:
        sources.append((source, Path("project") / source.relative_to(PROJECT)))
    for source in (
        WORKSPACE / "AGENTS.md",
        PROJECT / "fast-developer" / "RESEARCH_CONTEXT.md",
        PROJECT / "fast-developer" / "analysis" / "full_mst_slt_implementation.md",
        PROJECT / "fast-developer" / "analysis" / "experiment_history.md",
    ):
        if source.is_file():
            sources.append((source, Path("research_context") / source.name))
    records = []
    for source, relative in sources:
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(source)
        shutil.copy2(source, destination)
        copied = sha256(destination)
        if copied != before or sha256(source) != before:
            raise RuntimeError("Source changed while archiving: " + str(source))
        records.append({"source": str(source), "copy": str(relative), "sha256": copied,
                        "bytes": destination.stat().st_size})
    manifest = {"created_at_utc": now(), "source_project": str(PROJECT),
                "file_count": len(records), "files": records}
    write_json(archive / "manifest.json", manifest)
    return manifest


def package_versions():
    result = {}
    for package in ("torch", "stable-baselines3", "gymnasium", "numpy", "sumolib", "traci"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def validate_scenario():
    from envs.sumo.random_intersection import get_random_intersection_config
    configuration = get_random_intersection_config(SCENARIO)
    scenario = PROJECT / "envs" / "sumo" / "original_scenarios_v1" / SCENARIO
    stored = json.loads((scenario / "scenario_manifest.json").read_text(encoding="utf-8"))
    for label, config in (("runtime", configuration), ("scene manifest", stored["configuration"])):
        flows = config.get("flows", [])
        if len(flows) != 3 or any(abs(float(flow["probability_per_step"]) - 0.03) > 1e-12 for flow in flows):
            raise ValueError(label + ": expected three p=0.03 Bernoulli inlets")
        if any(str(flow["depart_lane"]) != "1" or str(flow["arrival_lane"]) != "1" for flow in flows):
            raise ValueError(label + ": background lane configuration changed")
        for key, expected in (
            ("protocol", "intersection_bernoulli_step_schedule_v1"),
            ("configuration_revision", "darrl_r2_20260930"),
            ("terminal_outcome_protocol", "exclusive_terminal_v2"),
            ("collision_action", "remove"),
            ("step_length_seconds", 0.1),
        ):
            if config.get(key) != expected:
                raise ValueError(label + ": unexpected " + key)
    return configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    configuration = validate_scenario()
    command = [
        sys.executable, "-u", str(PROJECT / "fast-developer" / "train_intersection_yield_v2_d1.py"),
        "--scenario", SCENARIO, "--depart-scale", "1.0",
        "--eval-traffic-split", "validation", "--methods", ",".join(METHODS),
        "--max-steps", "100000", "--run-root", str(run_root),
        "--checkpoint-frequency", "10000", "--behavior-diagnostics",
    ]
    if not args.start:
        print(json.dumps({"command": command, "methods": METHODS, "workers": 2,
                          "note": "Preview only; no files or training were created."}, indent=2))
        return
    # Refuse accidental continuation or overwrite before creating any worker.
    run_root.mkdir(parents=True, exist_ok=False)
    archive = archive_source(run_root)
    manifest = {
        "created_at_utc": now(),
        "scenario": SCENARIO,
        "scenario_revision": "darrl_r2_20260930",
        "scenario_configuration": configuration,
        "per_background_inlet_probability_per_0_1s": 0.03,
        "depart_scale": 1.0,
        "methods": list(METHODS),
        "training_seed": 0,
        "raw_steps_per_method": 100000,
        "continuation": False,
        "max_cuda_workers": 2,
        "cuda_visible_devices": "0",
        "evaluation_episodes": 100,
        "evaluation_checkpoint": "final_model",
        "training_traffic_split": "train",
        "evaluation_traffic_split": "validation",
        "terminal_outcome_protocol": "exclusive_terminal_v2",
        "route_reachability_protocol": "route_continuation_v1",
        "command": command,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "package_versions": package_versions(),
        "source_archive_file_count": archive["file_count"],
        "source_archive_manifest": str(run_root / "source_archive" / "manifest.json"),
        "mst_slt_protocol_note": (
            "MST+SLT retains its original algorithm/SLT. DARRL uses train/validation seed domains "
            "through the environment factory. A legacy frozen_80_20 provenance argument is not "
            "an active XML holdout for this random scenario."
        ),
    }
    write_json(run_root / "suite_manifest.json", manifest)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    status_path = run_root / "suite_supervisor_status.json"
    with (run_root / "suite.stdout.log").open("w", encoding="utf-8") as stdout, (
            run_root / "suite.stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=str(PROJECT), env=environment,
                                   stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        status = {"state": "running", "started_at_utc": now(), "supervisor_pid": process.pid,
                  "launcher_pid": os.getpid(), "methods": list(METHODS),
                  "max_cuda_workers": 2, "run_root": str(run_root)}
        try:
            write_json(status_path, status)
        except OSError as error:
            # Keep supervising the already-started child even if Windows locks the status file.
            print(json.dumps({"status_write_error": str(error), "status": status}), flush=True)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        exit_code = process.wait()
        status.update(state="complete" if exit_code == 0 else "failed",
                      finished_at_utc=now(), exit_code=exit_code)
        try:
            write_json(status_path, status)
        except OSError as error:
            print(json.dumps({"status_write_error": str(error), "status": status}), flush=True)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
