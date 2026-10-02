"""Re-evaluate raw-step checkpoints on an explicit, identical seed sequence.

SB3's legacy periodic ``EvalCallback`` did not reset the SUMO seed stream at
each evaluation point.  This tool makes learning curves paper-checkable by
evaluating every persisted checkpoint with ``seed_start + episode`` and the
run's frozen evaluation traffic partition.  Each checkpoint result is written
atomically and reused only when its SHA256 and protocol fields still match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch import SceneRepresentationSAC, evaluate_model_detailed
from tools.paper_evaluation_contract import (
    make_paper_evaluation_env,
    validate_model_environment_spaces,
)


CONTRACT = "topo-scene.fixed-checkpoint-curve/v1"
CHECKPOINT_PATTERN = re.compile(r"_raw_(\d+)_steps\.zip$")
PAIRED_PROTOCOL_FIELDS = (
    "scenario",
    "episodes_per_checkpoint",
    "evaluation_seed_start",
    "evaluation_seeds",
    "traffic_protocol",
    "episode_limit_profile",
    "deterministic_policy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Checkpoint does not expose a raw-step suffix: {path}")
    return int(match.group(1))


def _paired_protocol(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("At least one checkpoint-curve run is required")
    reference = {field: runs[0][field] for field in PAIRED_PROTOCOL_FIELDS}
    mismatched = {
        str(run["algorithm"]): {
            field: run[field]
            for field in PAIRED_PROTOCOL_FIELDS
            if run[field] != reference[field]
        }
        for run in runs[1:]
    }
    mismatched = {algorithm: fields for algorithm, fields in mismatched.items() if fields}
    if mismatched:
        raise ValueError(f"Checkpoint curve protocols are not paired: {mismatched}")
    return reference


def _valid_existing(
    path: Path,
    *,
    checkpoint_sha256: str,
    raw_steps: int,
    episodes: int,
    seed_start: int,
    traffic_protocol: str,
    algorithm: str,
    scenario: str,
    episode_limit_profile: str,
    deterministic_policy: bool,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "contract": CONTRACT,
        "checkpoint_sha256": checkpoint_sha256,
        "raw_steps": raw_steps,
        "episodes": episodes,
        "evaluation_seed_start": seed_start,
        "traffic_protocol": traffic_protocol,
        "algorithm": algorithm,
        "scenario": scenario,
        "episode_limit_profile": episode_limit_profile,
        "deterministic_policy": deterministic_policy,
    }
    return payload if all(payload.get(key) == value for key, value in expected.items()) else None


def _evaluate_checkpoint(
    checkpoint: Path,
    requested: dict[str, Any],
    *,
    raw_steps: int,
    episodes: int,
    seed_start: int,
    device: str,
) -> dict[str, Any]:
    traffic_protocol = str(requested["traffic_protocol"])
    checkpoint_hash = _sha256(checkpoint)
    output_path = checkpoint.parent.parent / "fixed_checkpoint_evaluations" / f"raw_{raw_steps}.json"
    existing = _valid_existing(
        output_path,
        checkpoint_sha256=checkpoint_hash,
        raw_steps=raw_steps,
        episodes=episodes,
        seed_start=seed_start,
        traffic_protocol=traffic_protocol,
        algorithm=str(requested["algo"]),
        scenario=str(requested["scenario"]),
        episode_limit_profile=str(requested["episode_limit_profile"]),
        deterministic_policy=True,
    )
    if existing is not None:
        return existing

    env = make_paper_evaluation_env(requested, traffic_protocol=traffic_protocol)
    try:
        model = SceneRepresentationSAC.load(checkpoint, env=env, device=device)
        trained_raw_steps = getattr(model, "_raw_steps_seen", None)
        if trained_raw_steps is None or int(trained_raw_steps) != raw_steps:
            raise ValueError(
                f"Checkpoint raw-step clock mismatch for {checkpoint}: "
                f"model={trained_raw_steps}, filename={raw_steps}"
            )
        space_contract = validate_model_environment_spaces(model, env)
        report = evaluate_model_detailed(
            model,
            env,
            episodes=episodes,
            seed=seed_start,
            deterministic=True,
        )
    finally:
        env.close()
    payload = {
        "contract": CONTRACT,
        "computed_from_real_rollout": True,
        "fabricated_values": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "algorithm": str(requested["algo"]),
        "scenario": str(requested["scenario"]),
        "raw_steps": raw_steps,
        "episodes": episodes,
        "evaluation_seed_start": seed_start,
        "evaluation_seeds": list(range(seed_start, seed_start + episodes)),
        "traffic_protocol": traffic_protocol,
        "episode_limit_profile": str(requested["episode_limit_profile"]),
        "deterministic_policy": True,
        "space_contract": space_contract,
        **report.to_dict(),
    }
    _atomic_json(output_path, payload)
    return payload


def _evaluate_run(
    run_dir: Path,
    *,
    episodes: int | None,
    seed_start: int | None,
    device: str,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    arguments_payload = json.loads((run_dir / "arguments.json").read_text(encoding="utf-8"))
    requested = arguments_payload["requested_raw_steps"]
    algorithm = str(requested["algo"])
    if algorithm not in {
        "scene_rep",
        "mst",
        "topo_scene",
        "topo_scene_balanced",
        "temporal_graph",
        "sac",
    }:
        raise ValueError(f"Unsupported off-policy algorithm: {algorithm}")
    resolved_episodes = int(requested["eval_episodes"]) if episodes is None else int(episodes)
    resolved_seed = int(requested["seed"]) + 10_000 if seed_start is None else int(seed_start)
    if resolved_episodes <= 0 or resolved_seed < 0:
        raise ValueError("episodes must be positive and seed-start non-negative")
    checkpoints = sorted(
        (path for path in (run_dir / "checkpoints").glob("*.zip") if CHECKPOINT_PATTERN.search(path.name)),
        key=_checkpoint_step,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No raw-step checkpoints found under {run_dir}")
    results = [
        _evaluate_checkpoint(
            checkpoint,
            requested,
            raw_steps=_checkpoint_step(checkpoint),
            episodes=resolved_episodes,
            seed_start=resolved_seed,
            device=device,
        )
        for checkpoint in checkpoints
    ]
    return {
        "run_directory": str(run_dir),
        "algorithm": algorithm,
        "scenario": str(requested["scenario"]),
        "implementation_id": arguments_payload["implementation_fidelity"]["implementation_id"],
        "episodes_per_checkpoint": resolved_episodes,
        "evaluation_seed_start": resolved_seed,
        "evaluation_seeds": list(range(resolved_seed, resolved_seed + resolved_episodes)),
        "traffic_protocol": str(requested["traffic_protocol"]),
        "episode_limit_profile": str(requested["episode_limit_profile"]),
        "deterministic_policy": True,
        "points": [
            {
                "raw_steps": row["raw_steps"],
                **row["summary"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
            for row in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [
        _evaluate_run(
            run_dir,
            episodes=args.episodes,
            seed_start=args.seed_start,
            device=args.device,
        )
        for run_dir in args.run_dir
    ]
    reference_protocol = _paired_protocol(runs)
    payload = {
        "contract": CONTRACT,
        "computed_from_real_rollouts": True,
        "fabricated_values": False,
        "historical_periodic_curves_are_exploratory": True,
        "fixed_checkpoint_curves_are_primary": True,
        "paired_protocol": reference_protocol,
        "runs": runs,
    }
    output_path = args.output.resolve()
    _atomic_json(output_path, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
