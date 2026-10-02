"""Real SUMO path smoke for the short DARRL overlay root; no learner is created."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = WORKSPACE / "Scene-Rep-Transformer-main" / "pytorch_sb3_sumo"
FAST = PROJECT / "fast-developer"
SMOKE_ROOT = WORKSPACE / "runs" / "dm7_pathchk"
SCENARIO = "intersection_random_darrl_medium_v1"
METHOD = "sac_mlp_d1_st_rt_topo_routeaware_v1"

if str(FAST) not in sys.path:
    sys.path.insert(0, str(FAST))

import train_intersection_yield_v2 as base  # noqa: E402
import train_intersection_yield_v2_d1 as d1  # noqa: E402


def _candidate_manifest_path(overlay_root: Path) -> Path:
    # Dynamic episode route filenames include a fixed-width 16-character hash.
    # Check the longest route-aware DARRL basename before starting SUMO.
    basename = "episode.rou__s" + ("0" * 16) + "__hdx1_i5m6s100_v1.rou.xml"
    return overlay_root / SCENARIO / f"{basename}.manifest.json"


def _make_env(overlay_root: Path, *, evaluation: bool):
    env = base.make_env_factory("base", overlay_root)(
        base._environment_namespace(), evaluation=evaluation
    )
    return d1._wrap_route_reachability_env(env, METHOD)


def _reset_and_step(env, phase: str, seed: int):
    observation, _ = env.reset(seed=seed)
    if not env.observation_space.contains(observation):
        raise AssertionError(f"{phase} seed={seed}: reset observation outside space")
    next_observation, reward, terminated, truncated, info = env.step(
        np.zeros(2, dtype=np.float32)
    )
    if not env.observation_space.contains(next_observation):
        raise AssertionError(f"{phase} seed={seed}: step observation outside space")

    raw_env = env.unwrapped
    manifest = getattr(raw_env, "_selected_high_density_manifest", None)
    overlay = getattr(raw_env, "_selected_high_density_overlay", None)
    if not manifest or overlay is None:
        raise AssertionError(f"{phase} seed={seed}: selected overlay was not recorded")
    overlay = Path(overlay)
    manifest_path = overlay.with_name(overlay.name + ".manifest.json")
    if not overlay.is_file() or not manifest_path.is_file():
        raise AssertionError(f"{phase} seed={seed}: overlay or manifest is missing")
    if len(str(manifest_path)) >= 240:
        raise AssertionError(
            f"{phase} seed={seed}: manifest path reached 240 characters: "
            f"{len(str(manifest_path))}: {manifest_path}"
        )
    overlay_sha256 = hashlib.sha256(overlay.read_bytes()).hexdigest()
    if overlay_sha256 != manifest.get("overlay_sha256"):
        raise AssertionError(f"{phase} seed={seed}: overlay does not match its manifest")
    if info.get("high_density_overlay_sha256") != overlay_sha256:
        raise AssertionError(f"{phase} seed={seed}: info reports another overlay hash")
    return {
        "phase": phase,
        "seed": seed,
        "overlay_path": str(overlay),
        "manifest_path": str(manifest_path),
        "manifest_path_chars": len(str(manifest_path)),
        "source_sha256": manifest.get("source_sha256"),
        "overlay_sha256": overlay_sha256,
        "base_explicit_vehicles": manifest.get("base_explicit_vehicles"),
        "additional_explicit_vehicles": manifest.get("additional_explicit_vehicles"),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "route_lane_status_known": info.get("route_lane_status_known"),
    }, dict(manifest)


def main() -> None:
    # Match D1's normal factory path construction while redirecting only the
    # output/run root to the requested isolated smoke directory.
    base.RESULT_ROOT = SMOKE_ROOT
    d1.RESULT_ROOT = SMOKE_ROOT
    d1._apply_patch(1.0, scenario=SCENARIO, eval_traffic_split="validation")
    run_dir = d1._run_dir(METHOD, 1.0)
    if run_dir.parent.resolve() != SMOKE_ROOT.resolve():
        raise AssertionError(f"unexpected run directory parent: {run_dir.parent}")

    train_root = run_dir.parent / "_hd" / run_dir.name.split("__", 1)[0] / "ns_tr"
    eval_root = run_dir.parent / "_hd" / run_dir.name.split("__", 1)[0] / "ns_eval"
    reference_root = run_dir.parent / "_ref" / METHOD / "ns_tr"
    roots = {"train": train_root, "eval": eval_root, "reference": reference_root}
    for phase, root in roots.items():
        candidate = _candidate_manifest_path(root)
        if len(str(candidate)) >= 240:
            raise AssertionError(
                f"{phase} candidate manifest path must be <240 chars before reset; "
                f"got {len(str(candidate))}: {candidate}"
            )

    records = []
    train_seed0_manifest = None
    for phase, root, evaluation in (
        ("train", train_root, False),
        ("eval", eval_root, True),
    ):
        env = _make_env(root, evaluation=evaluation)
        try:
            for seed in (0, 1):
                record, manifest = _reset_and_step(env, phase, seed)
                records.append(record)
                if phase == "train" and seed == 0:
                    train_seed0_manifest = manifest
        finally:
            env.close()

    # Confirm storage relocation leaves same-seed source traffic and overlay
    # bytes/counts unchanged.
    reference_env = _make_env(reference_root, evaluation=False)
    try:
        reference_record, reference_manifest = _reset_and_step(
            reference_env, "same_seed_reference", 0
        )
        records.append(reference_record)
    finally:
        reference_env.close()

    matched_fields = (
        "source_sha256",
        "overlay_sha256",
        "base_explicit_vehicles",
        "additional_explicit_vehicles",
    )
    differences = {
        key: (train_seed0_manifest.get(key), reference_manifest.get(key))
        for key in matched_fields
        if train_seed0_manifest.get(key) != reference_manifest.get(key)
    }
    if differences:
        raise AssertionError(f"same-seed storage invariant failed: {differences}")

    result = {
        "protocol": "routeaware_overlay_shortpath_multi_reset_v1",
        "scenario": SCENARIO,
        "method": METHOD,
        "run_dir": str(run_dir),
        "train_overlay_root": str(train_root),
        "eval_overlay_root": str(eval_root),
        "reference_overlay_root": str(reference_root),
        "real_sumo_resets": len(records),
        "real_sumo_steps": len(records),
        "same_seed_storage_invariant": {
            "seed": 0,
            "matched_fields": list(matched_fields),
            "passed": True,
        },
        "records": records,
    }
    output_path = SMOKE_ROOT / "overlay_path_smoke_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "result_path": str(output_path),
        "train_overlay_root": str(train_root),
        "eval_overlay_root": str(eval_root),
        "real_sumo_resets": len(records),
        "real_sumo_steps": len(records),
        "manifest_path_chars": [
            {"phase": item["phase"], "chars": item["manifest_path_chars"]}
            for item in records
        ],
        "same_seed_storage_invariant": result["same_seed_storage_invariant"],
    }, indent=2))


if __name__ == "__main__":
    main()
