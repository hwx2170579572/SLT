"""Re-evaluate a fully trained paper run without resuming or changing training."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.paper_evaluation_contract import (
    build_evaluation_provenance,
    make_paper_evaluation_env,
    saved_evaluation_contract_errors,
    validate_model_environment_spaces,
)
from tools.recover_interrupted_paper_evaluation import (
    _sha256,
    validate_completed_training,
)


PROTOCOLS = ("frozen_80_20", "source_all")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _protocol_values(value: str) -> list[str]:
    protocols = [part.strip() for part in value.split(",") if part.strip()]
    invalid = sorted(set(protocols) - set(PROTOCOLS))
    if invalid or not protocols:
        raise ValueError(f"Choose one or more protocols from {PROTOCOLS}; got {invalid}")
    return list(dict.fromkeys(protocols))


def _model_class(algorithm: str) -> type[Any]:
    from algos.sb3_torch import SourcePPO, SceneRepresentationSAC

    return SourcePPO if algorithm == "ppo" else SceneRepresentationSAC


def _training_clock_payload(
    *, algorithm: str, arguments: dict[str, Any], model: Any
) -> dict[str, Any]:
    trained_raw_steps = getattr(model, "_raw_steps_seen", None)
    if trained_raw_steps is None and algorithm == "ppo":
        trained_raw_steps = model.num_timesteps
    if int(trained_raw_steps) != int(arguments["max_steps"]):
        raise ValueError(
            f"Loaded final model has {trained_raw_steps} raw steps; expected "
            f"{arguments['max_steps']}"
        )
    selected_timesteps = int(model.num_timesteps)
    collected_raw_steps = (
        int(math.ceil(int(arguments["max_steps"]) / 512) * 512)
        if algorithm == "ppo"
        else int(arguments["max_steps"])
    )
    return {
        "trained_raw_steps": int(trained_raw_steps),
        "collected_training_raw_steps": collected_raw_steps,
        "post_training_learner_timesteps": (
            collected_raw_steps if algorithm == "ppo" else selected_timesteps
        ),
        "selected_model_learner_timesteps": selected_timesteps,
        "source_test_checkpoint_step": (
            selected_timesteps if algorithm == "ppo" else None
        ),
    }


def _existing_protocol_evaluation(
    path: Path,
    *,
    algorithm: str,
    scenario: str,
    traffic_protocol: str,
    episode_limit_profile: str,
    episodes: int,
    evaluation_seed_start: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    detailed = json.loads(path.read_text(encoding="utf-8"))
    errors = saved_evaluation_contract_errors(
        detailed,
        algorithm=algorithm,
        scenario=scenario,
        traffic_protocol=traffic_protocol,
        episode_limit_profile=episode_limit_profile,
        episodes=episodes,
        evaluation_seed_start=evaluation_seed_start,
    )
    return detailed if not errors else None


def _promote_primary(run_dir: Path, protocol_dir: Path) -> dict[str, Any]:
    source_summary = protocol_dir / "final_evaluation.json"
    source_detailed = protocol_dir / "paper_evaluation_detailed.json"
    targets = {
        "final_evaluation.json": source_summary,
        "paper_evaluation_detailed.json": source_detailed,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = run_dir / "superseded_evaluations" / timestamp
    backed_up: list[str] = []
    for filename, source in targets.items():
        target = run_dir / filename
        if target.is_file() and target.read_bytes() != source.read_bytes():
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_dir / filename)
            backed_up.append(filename)
        shutil.copy2(source, target)
    return {
        "canonical_outputs_promoted": sorted(targets),
        "superseded_outputs_backed_up": backed_up,
        "backup_directory": str(backup_dir) if backed_up else None,
    }


def _adopt_valid_primary_evaluation(
    run_dir: Path,
    protocol_dir: Path,
    *,
    algorithm: str,
    scenario: str,
    traffic_protocol: str,
    episode_limit_profile: str,
    episodes: int,
    evaluation_seed_start: int,
    model_sha256: str,
    checkpoint_audit_sha256: str,
) -> dict[str, Any] | None:
    """Copy a valid canonical primary evaluation into the protocol ledger.

    Fresh runs already execute the strict final evaluation inside the training
    process.  Re-running those same frozen episodes would add no evidence, so
    accept the canonical artifact only after applying the exact same saved
    provenance/space contract used for protocol-specific evaluations.
    """

    canonical_path = run_dir / "paper_evaluation_detailed.json"
    detailed = _existing_protocol_evaluation(
        canonical_path,
        algorithm=algorithm,
        scenario=scenario,
        traffic_protocol=traffic_protocol,
        episode_limit_profile=episode_limit_profile,
        episodes=episodes,
        evaluation_seed_start=evaluation_seed_start,
    )
    if detailed is None:
        return None

    protocol_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = protocol_dir / "paper_evaluation_detailed.json"
    shutil.copy2(canonical_path, detailed_path)
    _atomic_json(protocol_dir / "final_evaluation.json", detailed["summary"])
    receipt = {
        "kind": "adopted_valid_primary_evaluation_without_rerun",
        "training_was_not_resumed_or_modified": True,
        "evaluation_was_not_rerun": True,
        "run_directory": str(run_dir),
        "canonical_detailed_sha256": _sha256(canonical_path),
        "adopted_detailed_sha256": _sha256(detailed_path),
        "model_sha256": model_sha256,
        "checkpoint_audit_sha256": checkpoint_audit_sha256,
        "traffic_protocol": traffic_protocol,
        "evaluation_seed_start": evaluation_seed_start,
        "episodes": episodes,
        "provenance_contract_version": detailed["evaluation_provenance"][
            "contract_version"
        ],
    }
    _atomic_json(protocol_dir / "evaluation_receipt.json", receipt)
    return detailed


def reevaluate_run(
    run_dir: Path,
    *,
    protocols: list[str],
    device: str,
    promote_primary: bool,
    force: bool,
) -> dict[str, Any]:
    from algos.sb3_torch import evaluate_model_detailed

    run_dir = run_dir.resolve()
    validated = validate_completed_training(
        run_dir, allow_existing_evaluation=True
    )
    arguments = validated["arguments"]
    assert isinstance(arguments, dict)
    algorithm = str(arguments["algo"])
    scenario = str(arguments["scenario"])
    episodes = int(arguments["eval_episodes"])
    evaluation_seed_start = int(arguments["seed"]) + 10_000
    episode_limit_profile = str(arguments.get("episode_limit_profile", "source"))
    primary_protocol = str(arguments.get("traffic_protocol", "source_all"))
    if promote_primary and primary_protocol not in protocols:
        raise ValueError(
            f"--promote-primary requires {primary_protocol!r} in --protocols"
        )

    model_path = validated["final_model"]
    assert isinstance(model_path, Path)
    evaluations: dict[str, Any] = {}
    for traffic_protocol in protocols:
        protocol_dir = run_dir / "protocol_evaluations" / traffic_protocol
        detailed_path = protocol_dir / "paper_evaluation_detailed.json"
        existing = None if force else _existing_protocol_evaluation(
            detailed_path,
            algorithm=algorithm,
            scenario=scenario,
            traffic_protocol=traffic_protocol,
            episode_limit_profile=episode_limit_profile,
            episodes=episodes,
            evaluation_seed_start=evaluation_seed_start,
        )
        if existing is not None:
            evaluations[traffic_protocol] = {
                "status": "reused_valid",
                "summary": existing["summary"],
                "detailed_path": str(detailed_path),
            }
            continue

        adopted = None
        if not force and traffic_protocol == primary_protocol:
            adopted = _adopt_valid_primary_evaluation(
                run_dir,
                protocol_dir,
                algorithm=algorithm,
                scenario=scenario,
                traffic_protocol=traffic_protocol,
                episode_limit_profile=episode_limit_profile,
                episodes=episodes,
                evaluation_seed_start=evaluation_seed_start,
                model_sha256=str(validated["final_model_sha256"]),
                checkpoint_audit_sha256=str(validated["audit_sha256"]),
            )
        if adopted is not None:
            evaluations[traffic_protocol] = {
                "status": "adopted_valid_primary",
                "summary": adopted["summary"],
                "detailed_path": str(detailed_path),
            }
            continue

        env = make_paper_evaluation_env(
            arguments, traffic_protocol=traffic_protocol
        )
        try:
            model = _model_class(algorithm).load(
                model_path, env=env, device=device
            )
            space_contract = validate_model_environment_spaces(model, env)
            clock_payload = _training_clock_payload(
                algorithm=algorithm, arguments=arguments, model=model
            )
            report = evaluate_model_detailed(
                model,
                env,
                episodes=episodes,
                seed=evaluation_seed_start,
                sumo_step_seconds=0.1,
                policy_action_hold=(
                    3 if algorithm == "ppo" and scenario == "carla" else 1
                ),
            )
            provenance = build_evaluation_provenance(
                model=model,
                env=env,
                report=report,
                algorithm=algorithm,
                scenario=scenario,
                traffic_protocol=traffic_protocol,
                episode_limit_profile=episode_limit_profile,
                evaluation_seed_start=evaluation_seed_start,
                environment_factory="tools.reevaluate_paper_run",
                space_contract=space_contract,
            )
        finally:
            env.close()

        best_metadata_path = run_dir / "best_training_success.json"
        detailed = {
            "model": str(model_path),
            "algorithm": algorithm,
            "scenario": scenario,
            "evaluation_seed_start": evaluation_seed_start,
            **clock_payload,
            "best_training_success_checkpoint": (
                json.loads(best_metadata_path.read_text(encoding="utf-8"))
                if best_metadata_path.is_file()
                else None
            ),
            "evaluation_provenance": provenance,
            "completion_time_population_std_ddof": 0,
            **report.to_dict(),
        }
        contract_errors = saved_evaluation_contract_errors(
            detailed,
            algorithm=algorithm,
            scenario=scenario,
            traffic_protocol=traffic_protocol,
            episode_limit_profile=episode_limit_profile,
            episodes=episodes,
            evaluation_seed_start=evaluation_seed_start,
        )
        if contract_errors:
            raise ValueError(
                "Generated evaluation failed its own contract: "
                + "; ".join(contract_errors)
            )
        _atomic_json(detailed_path, detailed)
        _atomic_json(protocol_dir / "final_evaluation.json", detailed["summary"])
        receipt = {
            "kind": "evaluation_only_without_retraining",
            "training_was_not_resumed_or_modified": True,
            "run_directory": str(run_dir),
            "model_sha256": validated["final_model_sha256"],
            "checkpoint_audit_sha256": validated["audit_sha256"],
            "traffic_protocol": traffic_protocol,
            "evaluation_seed_start": evaluation_seed_start,
            "episodes": episodes,
            "provenance_contract_version": provenance["contract_version"],
        }
        _atomic_json(protocol_dir / "evaluation_receipt.json", receipt)
        evaluations[traffic_protocol] = {
            "status": "evaluated",
            "summary": detailed["summary"],
            "detailed_path": str(detailed_path),
        }

    promotion = None
    if promote_primary:
        primary_dir = run_dir / "protocol_evaluations" / primary_protocol
        promotion = _promote_primary(run_dir, primary_dir)

    manifest_path = run_dir / "reevaluation_manifest.json"
    manifest = {
        "kind": "dual_protocol_evaluation_without_retraining",
        "training_was_not_resumed_or_modified": True,
        "run_directory": str(run_dir),
        "algorithm": algorithm,
        "scenario": scenario,
        "training_seed": int(arguments["seed"]),
        "model_sha256": validated["final_model_sha256"],
        "checkpoint_audit_sha256": validated["audit_sha256"],
        "primary_protocol": primary_protocol,
        "evaluations": evaluations,
        "promotion": promotion,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--protocols", default="frozen_80_20,source_all"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--promote-primary", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    result = reevaluate_run(
        args.run_dir,
        protocols=_protocol_values(args.protocols),
        device=args.device,
        promote_primary=args.promote_primary,
        force=args.force,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
