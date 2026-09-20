"""Train exactly one fresh cell in the independent-v2 5x6 extension."""

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

from envs.sumo.independent_v2_five_methods_six_scenarios_100ep_v1 import (  # noqa: E402
    EXPERIMENT_ID,
    IndependentV2FiveBySixEnvV1,
    IndependentV2FiveBySixEnvV4V1,
)
from tools import train_sb3, train_sb3_v4_8, train_sb3_v4_13  # noqa: E402
from tools.independent_v2_5m6s100e_v1_common import (  # noqa: E402
    DEFAULT_PROTOCOL_PATH,
    METHOD_ADAPTERS,
    RECEIPT_NAME,
    SCENARIOS,
    assert_bound_file,
    density_setting,
    is_adopted_cell,
    load_protocol,
    method_setting,
    profile_setting,
    sha256,
)


OVERLAY_DIRECTORY_NAME = "overlays_i5m6s100_v1"
RECEIPT_SCHEMA_VERSION = "independent-v2-5m6s100e-job-receipt/v1"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".j{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_ADAPTERS), required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", choices=("smoke", "comparison"), required=True)
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
    baseline_environment_class: type[Any] = IndependentV2FiveBySixEnvV1,
    v4_environment_class: type[Any] = IndependentV2FiveBySixEnvV4V1,
) -> Callable[..., Any]:
    environment_class = (
        baseline_environment_class if adapter == "base" else v4_environment_class
    )

    def make_environment(args: argparse.Namespace, *, evaluation: bool = False):
        if not evaluation:
            physical_partition = "train"
            contract_partition = "train"
        elif adapter == "base":
            physical_partition = "evaluation"
            contract_partition = "evaluation"
        else:
            contract_partition = str(getattr(args, "evaluation_split", "validation"))
            if contract_partition == "train":
                physical_partition = "train"
            elif contract_partition == "validation":
                physical_partition = "evaluation"
            else:
                raise ValueError(
                    "formal/test access is forbidden in this independent extension"
                )
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
            high_density_partition=physical_partition,
            high_density_contract_partition=contract_partition,
        )

    return make_environment


def _trainer_argv(
    args: argparse.Namespace,
    *,
    adapter: str,
    algorithm: str,
    profile: dict[str, Any],
    protocol: dict[str, Any],
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
        "--batch-size",
        "32",
        "--learning-rate",
        "0.0001",
        "--discount",
        "0.99",
        "--buffer-size",
        "20000",
        "--action-repeat",
        "3",
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
        values.extend(
            (
                "--traffic-protocol",
                "frozen_60_20_20",
                "--tensorboard-log-root",
                str(args.output_dir.resolve().parents[1] / "tb_v4"),
                "--checkpoint-prefix",
                "ckpt",
                "--evaluation-split",
                "validation",
                "--evaluation-seed-start",
                str(int(profile["evaluation_seed_start"])),
            )
        )
        if adapter == "v4_13":
            lineage = protocol["v4_13_lineage"]
            values.extend(
                (
                    "--experiment-contract-sha256",
                    str(lineage["architecture_contract_sha256"]),
                    "--stage0-results-sha256",
                    str(lineage["stage0_results_sha256"]),
                    "--implementation-freeze-sha256",
                    str(lineage["implementation_freeze_sha256"]),
                )
            )
        else:
            values.extend(("--experiment-contract-sha256", protocol["_sha256"]))
        values.extend(
            (
                "--calibration-episodes",
                str(int(profile["calibration_episodes"])),
                "--calibration-seed-start",
                str(int(profile["calibration_seed_start"])),
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
) -> int:
    if adapter == "base":
        return train_sb3.main(
            argv,
            env_factory=env_factory,
            default_output_dir=output_dir,
            require_paper_evaluation_contract=True,
            tensorboard_log_root=output_dir.parent / "tensorboard_i5m6s100_v1",
        )
    if adapter == "v4_8":
        return train_sb3_v4_8.main(
            argv,
            env_factory=env_factory,
            default_output_dir=output_dir,
            require_paper_evaluation_contract=True,
        )
    if adapter == "v4_13":
        return train_sb3_v4_13.main(
            argv,
            env_factory=env_factory,
            default_output_dir=output_dir,
            require_paper_evaluation_contract=True,
        )
    raise ValueError(f"unsupported trainer adapter {adapter!r}")


def _collect_overlay_manifests(overlay_root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(overlay_root.rglob("*.manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        output.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "source_path": payload["source_path"],
                "source_sha256": payload["source_sha256"],
                "overlay_path": payload["overlay_path"],
                "overlay_sha256": payload["overlay_sha256"],
                "vehicle_scale": payload["vehicle_scale"],
                "pedestrian_scale": payload["pedestrian_scale"],
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
    return output


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.seed != 0:
        raise ValueError("this single-seed protocol permits seed 0 only")
    protocol = load_protocol(args.protocol)
    if is_adopted_cell(args.method, args.scenario):
        raise ValueError(
            "this cell already exists in independent v2; retraining and "
            "reevaluation are forbidden—use the adoption audit"
        )
    method = method_setting(protocol, args.method)
    density = density_setting(protocol, args.scenario)
    profile = profile_setting(protocol, args.profile)
    adapter, expected_algorithm = METHOD_ADAPTERS[args.method]
    if method["trainer_adapter"] != adapter:
        raise ValueError("method adapter conflicts with frozen registry")
    if method["cli_algorithm"] != expected_algorithm:
        raise ValueError("method algorithm conflicts with frozen registry")

    if adapter == "v4_13":
        assert_bound_file(
            protocol,
            "architecture_contract_path",
            "architecture_contract_sha256",
            "v4_13_lineage",
        )
        assert_bound_file(
            protocol,
            "implementation_freeze_path",
            "implementation_freeze_sha256",
            "v4_13_lineage",
        )

    output_dir = args.output_dir.resolve()
    run_dir = output_dir / args.model_name
    if run_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing run directory: {run_dir}"
        )
    overlay_root = (
        output_dir.parent
        / OVERLAY_DIRECTORY_NAME
        / args.method
        / f"seed_{args.seed}"
    )
    environment_factory = _make_environment_factory(
        adapter=adapter,
        density=density,
        overlay_root=overlay_root,
    )
    trainer_argv = _trainer_argv(
        args,
        adapter=adapter,
        algorithm=expected_algorithm,
        profile=profile,
        protocol=protocol,
    )
    exit_code = _run_trainer(
        adapter,
        trainer_argv,
        env_factory=environment_factory,
        output_dir=output_dir,
    )
    if exit_code != 0:
        return int(exit_code)

    required = (
        ("arguments.json", "check_result.json")
        if args.check_only
        else ("arguments.json", "final_model.zip", "paper_evaluation_detailed.json")
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"trainer returned success but artifacts are missing: {missing}"
        )
    if not args.check_only:
        detailed = json.loads(
            (run_dir / "paper_evaluation_detailed.json").read_text(encoding="utf-8")
        )
        summary = detailed.get("summary", {})
        if int(summary.get("episodes", -1)) != int(profile["evaluation_episodes"]):
            raise ValueError("evaluation episode count does not match the protocol")
        if detailed.get("evaluation_split") == "test" or detailed.get(
            "formal_test_accessed"
        ) is True:
            raise ValueError("formal/test evidence was unexpectedly accessed")

    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "comparison_experiment_id": EXPERIMENT_ID,
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
        "clone_depart_jitter_seconds": density["clone_depart_jitter_seconds"],
        "physical_traffic_partition": "frozen_80_20",
        "contract_evaluation_partition": (
            "evaluation" if adapter == "base" else "validation"
        ),
        "formal_test_accessed": False,
        "map_changed": False,
        "ego_route_changed": False,
        "observation_changed_within_method": False,
        "action_space_changed": False,
        "reward_changed": False,
        "original_traffic_assets_modified": False,
        "run_directory": str(run_dir.resolve()),
        "overlay_manifests": _collect_overlay_manifests(
            overlay_root / args.scenario
        ),
        "v4_13_lineage": (
            dict(protocol["v4_13_lineage"]) if adapter == "v4_13" else None
        ),
        "artifact_sha256": {name: sha256(run_dir / name) for name in required},
    }
    _write_json_atomic(run_dir / RECEIPT_NAME, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
