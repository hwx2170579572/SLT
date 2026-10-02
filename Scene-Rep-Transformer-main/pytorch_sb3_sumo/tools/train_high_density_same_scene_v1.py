"""Train one frozen method job in the versioned high-density environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sumo.high_density_env_v1 import (  # noqa: E402
    HighDensityPaperSumoSceneEnvV1,
    HighDensityPaperSumoSceneEnvV4V1,
)
from tools import train_sb3, train_sb3_v4, train_sb3_v4_8  # noqa: E402
from tools.high_density_same_scene_v1_common import (  # noqa: E402
    DEFAULT_PROTOCOL_PATH,
    METHOD_ADAPTERS,
    TARGET_SCENARIOS,
    density_setting,
    load_protocol,
    method_setting,
    profile_setting,
    sha256,
)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid repeating a potentially long artifact basename in temporary paths
    # on Windows (the repository itself may already be close to MAX_PATH).
    temporary = path.with_name(f".j{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_ADAPTERS), required=True)
    parser.add_argument("--scenario", choices=TARGET_SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", default="comparison")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--check-only", action="store_true")
    return parser


def _make_environment_factory(
    *,
    adapter: str,
    density: dict[str, Any],
    overlay_root: Path,
    baseline_environment_class: type[Any] = HighDensityPaperSumoSceneEnvV1,
    v4_environment_class: type[Any] = HighDensityPaperSumoSceneEnvV4V1,
) -> Callable[..., Any]:
    environment_class = (
        baseline_environment_class
        if adapter == "base"
        else v4_environment_class
    )

    def make_environment(args: argparse.Namespace, *, evaluation: bool = False):
        return environment_class(
            scenario=args.scenario,
            history_steps=args.history_steps,
            neighbors=args.neighbors,
            path_length=args.path_length,
            action_repeat=args.action_repeat,
            reward_discount=args.discount,
            ego_control_profile=args.ego_control_profile,
            include_state_lstm=False,
            state_lstm_only=False,
            episode_limit_profile=args.episode_limit_profile,
            render_mode="human" if args.gui and not evaluation else None,
            high_density_vehicle_scale=float(density["vehicle_scale"]),
            high_density_pedestrian_scale=float(density["pedestrian_scale"]),
            high_density_clone_jitter_seconds=tuple(
                float(value) for value in density["clone_depart_jitter_seconds"]
            ),
            high_density_overlay_root=overlay_root,
            high_density_partition="evaluation" if evaluation else "train",
        )

    return make_environment


def _trainer_argv(
    args: argparse.Namespace,
    *,
    algorithm: str,
    adapter: str,
    profile: dict[str, Any],
    protocol_sha256: str,
) -> list[str]:
    values = [
        "--scenario",
        args.scenario,
        "--algo",
        algorithm,
        "--max-steps",
        str(int(profile["raw_training_steps"])),
        "--learning-starts",
        str(int(profile["learning_starts_raw_steps"])),
        "--checkpoint-freq",
        str(int(profile["checkpoint_frequency_raw_steps"])),
        "--eval-freq",
        "0",
        "--eval-episodes",
        str(int(profile["evaluation_episodes"])),
        "--seed",
        str(int(args.seed)),
        "--device",
        args.device,
        "--output-dir",
        str(args.output_dir.resolve()),
        "--model-name",
        args.model_name,
        "--ego-control-profile",
        "direct",
        "--episode-limit-profile",
        "source",
    ]
    if adapter == "base":
        values.extend(("--traffic-protocol", "frozen_80_20"))
    else:
        # The versioned trainer accepts this legacy token.  The injected v1
        # environment records and enforces one matched 80/20 split for every
        # adapter, independently of the legacy parser label.
        values.extend(
            (
                "--traffic-protocol",
                "frozen_60_20_20",
                "--evaluation-split",
                "validation",
                "--evaluation-seed-start",
                str(int(args.seed) + 10_000),
                "--experiment-contract-sha256",
                protocol_sha256,
            )
        )
    if adapter == "v4_8":
        values.extend(
            (
                "--calibration-episodes",
                str(int(profile["calibration_episodes"])),
                "--calibration-seed-start",
                str(80_000 + int(args.seed) * 1_000),
            )
        )
    if args.check_only:
        values.append("--check-only")
    return values


def _run_trainer(
    adapter: str,
    argv: list[str],
    *,
    env_factory: Callable[..., Any],
    output_dir: Path,
    tensorboard_log_root: Path | None,
) -> int:
    if adapter == "base":
        return train_sb3.main(
            argv,
            env_factory=env_factory,
            default_output_dir=output_dir,
            require_paper_evaluation_contract=True,
            tensorboard_log_root=tensorboard_log_root,
        )
    if adapter == "v4_1":
        return train_sb3_v4.main(
            argv,
            env_factory=env_factory,
            default_output_dir=output_dir,
            require_paper_evaluation_contract=True,
        )
    if adapter == "v4_8":
        return train_sb3_v4_8.main(
            argv,
            env_factory=env_factory,
            default_output_dir=output_dir,
            require_paper_evaluation_contract=True,
        )
    raise ValueError(f"unsupported trainer adapter {adapter!r}")


def _collect_overlay_manifests(overlay_root: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted(overlay_root.rglob("*.manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "source_path": payload["source_path"],
                "source_sha256": payload["source_sha256"],
                "overlay_path": payload["overlay_path"],
                "overlay_sha256": payload["overlay_sha256"],
                "additional_explicit_vehicles": payload[
                    "additional_explicit_vehicles"
                ],
                "additional_vehicle_flows": payload["additional_vehicle_flows"],
                "additional_explicit_persons": payload[
                    "additional_explicit_persons"
                ],
                "additional_person_flows": payload["additional_person_flows"],
            }
        )
    return manifests


def main(
    argv: list[str] | None = None,
    *,
    baseline_environment_class: type[Any] = HighDensityPaperSumoSceneEnvV1,
    v4_environment_class: type[Any] = HighDensityPaperSumoSceneEnvV4V1,
    overlay_directory_name: str = "high_density_overlays_v1",
    receipt_schema_version: str = "same-scene-high-density-job-receipt/v1",
    comparison_experiment_id: str = "high-density-same-scene-v1",
    environment_factory_builder: Callable[..., Callable[..., Any]] | None = None,
    tensorboard_log_root: Path | None = None,
) -> int:
    args = argument_parser().parse_args(argv)
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    protocol = load_protocol(args.protocol)
    method = method_setting(protocol, args.method)
    density = density_setting(protocol, args.scenario)
    profile = profile_setting(protocol, args.profile)
    adapter, expected_algorithm = METHOD_ADAPTERS[args.method]
    if method["trainer_adapter"] != adapter:
        raise ValueError("method trainer adapter conflicts with the frozen registry")
    if method["cli_algorithm"] != expected_algorithm:
        raise ValueError("method CLI algorithm conflicts with the frozen registry")

    output_dir = args.output_dir.resolve()
    run_dir = output_dir / args.model_name
    overlay_directory = Path(overlay_directory_name)
    if (
        not overlay_directory_name
        or overlay_directory_name in {".", ".."}
        or overlay_directory.is_absolute()
        or len(overlay_directory.parts) != 1
    ):
        raise ValueError("overlay_directory_name must be one directory basename")
    # Keep generated route paths below Windows' legacy MAX_PATH threshold.
    # The namespace remains job-specific (method + seed), while the environment
    # adds the scenario and source-route basename underneath it.
    overlay_root = (
        output_dir.parent
        / overlay_directory_name
        / args.method
        / f"seed_{args.seed}"
    )
    factory_builder = environment_factory_builder or _make_environment_factory
    environment_factory = factory_builder(
        adapter=adapter,
        density=density,
        overlay_root=overlay_root,
        baseline_environment_class=baseline_environment_class,
        v4_environment_class=v4_environment_class,
    )
    trainer_argv = _trainer_argv(
        args,
        algorithm=expected_algorithm,
        adapter=adapter,
        profile=profile,
        protocol_sha256=protocol["_sha256"],
    )
    exit_code = _run_trainer(
        adapter,
        trainer_argv,
        env_factory=environment_factory,
        output_dir=output_dir,
        tensorboard_log_root=tensorboard_log_root,
    )
    if exit_code != 0:
        return int(exit_code)

    required = (
        ("arguments.json", "check_result.json")
        if args.check_only
        else (
            "arguments.json",
            "final_model.zip",
            "paper_evaluation_detailed.json",
        )
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"trainer returned success but required artifacts are missing: {missing}"
        )
    receipt = {
        "schema_version": receipt_schema_version,
        "comparison_experiment_id": comparison_experiment_id,
        "status": "preflight_passed" if args.check_only else "completed",
        "computed_from_real_run": True,
        "fabricated_values": False,
        "method": args.method,
        "display_label": method["display_label"],
        "trainer_adapter": adapter,
        "algorithm": expected_algorithm,
        "implementation_id": method["implementation_id"],
        "scenario": args.scenario,
        "scenario_id": density["scenario_id"],
        "seed": int(args.seed),
        "profile": args.profile,
        "protocol_path": protocol["_path"],
        "protocol_sha256": protocol["_sha256"],
        "vehicle_scale": float(density["vehicle_scale"]),
        "pedestrian_scale": float(density["pedestrian_scale"]),
        "clone_depart_jitter_seconds": density[
            "clone_depart_jitter_seconds"
        ],
        "matched_traffic_partition": "frozen_80_20",
        "map_changed": False,
        "ego_route_changed": False,
        "observation_changed": False,
        "action_space_changed": False,
        "reward_changed": False,
        "run_directory": str(run_dir.resolve()),
        "overlay_manifests": _collect_overlay_manifests(
            overlay_root / args.scenario
        ),
        "artifact_sha256": {
            name: sha256(run_dir / name) for name in required
        },
    }
    _write_json_atomic(run_dir / "high_density_job_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
