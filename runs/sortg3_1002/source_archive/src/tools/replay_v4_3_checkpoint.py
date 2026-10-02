"""Replay a frozen v4.2 checkpoint with the v4.3 deterministic decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch.hybrid_policy_v4_3 import (
    TargetCriticDecisionAlignedSACPolicyV43,
)
from algos.sb3_torch.sac_v4_3 import TargetCriticLaneDecoderSACV43
from configs.sb3_configs_v4_3 import V43_IMPLEMENTATION_IDS
from envs.sumo.paper_env_v4 import PaperSumoSceneEnvV4
from tools.action_diagnostics_v4_3 import evaluate_with_action_diagnostics_v4_3
from tools.paper_evaluation_contract import validate_model_environment_spaces
from tools.paper_evaluation_contract_v2 import build_evaluation_provenance_v2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _verify_zip(path: Path) -> None:
    with ZipFile(path, "r") as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise ValueError(f"source checkpoint ZIP CRC failed at {bad_member}")


def _make_env(scenario: str, *, action_repeat: int, discount: float):
    return PaperSumoSceneEnvV4(
        scenario=scenario,
        action_repeat=action_repeat,
        reward_discount=discount,
        traffic_partition="validation",
        episode_limit_profile="source",
        include_state_lstm=False,
        state_lstm_only=False,
    )


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--source-model-sha256", required=True)
    parser.add_argument("--source-training-seed", type=int, required=True)
    parser.add_argument("--scenario", choices=("carla",), required=True)
    parser.add_argument("--evaluation-seed-start", type=int, required=True)
    parser.add_argument("--evaluation-episodes", type=int, default=12)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-contract-sha256", required=True)
    parser.add_argument("--attribution-sha256", required=True)
    parser.add_argument("--implementation-freeze-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.evaluation_episodes <= 0 or args.evaluation_seed_start < 0:
        raise ValueError("evaluation episodes must be positive and seeds non-negative")
    if args.action_repeat != 3 or abs(args.discount - 0.99) > 1e-12:
        raise ValueError("v4.3 replay freezes action_repeat=3 and discount=0.99")

    source_run = args.source_run_dir.resolve()
    source_model = source_run / "final_model.zip"
    source_arguments_path = source_run / "arguments.json"
    source_detailed_path = source_run / "paper_evaluation_detailed.json"
    for path in (source_model, source_arguments_path, source_detailed_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    _verify_zip(source_model)
    source_model_sha = _sha256(source_model)
    if source_model_sha != args.source_model_sha256.lower():
        raise ValueError("source checkpoint hash does not match preregistration")
    source_arguments = _load_json(source_arguments_path)
    source_detailed = _load_json(source_detailed_path)
    requested = source_arguments.get("requested_raw_steps", {})
    if requested.get("seed") != args.source_training_seed:
        raise ValueError("source training seed does not match preregistration")
    if source_detailed.get("scenario") != args.scenario:
        raise ValueError("source checkpoint scenario does not match replay")
    if source_detailed.get("evaluation_split") != "validation":
        raise ValueError("source checkpoint did not originate from validation protocol")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    replay_arguments = {
        "schema_version": "topo-scene-v4.3.checkpoint-replay-arguments/v1",
        "implementation_id": V43_IMPLEMENTATION_IDS[
            "topo_v4_3_target_critic_decoder"
        ],
        "single_change": "deterministic_target_critic_lane_decoder",
        "source_run_directory": str(source_run),
        "source_training_seed": args.source_training_seed,
        "scenario": args.scenario,
        "evaluation_split": "validation",
        "evaluation_seed_start": args.evaluation_seed_start,
        "evaluation_episodes": args.evaluation_episodes,
        "traffic_partition": "validation",
        "traffic_protocol": "frozen_60_20_20",
        "episode_limit_profile": "source",
        "action_repeat": args.action_repeat,
        "discount": args.discount,
        "device": args.device,
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "attribution_sha256": args.attribution_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
    }
    _write_json(output / "replay_arguments.json", replay_arguments)
    source_receipt = {
        "schema_version": "topo-scene-v4.3.source-checkpoint-receipt/v1",
        "source_model": str(source_model),
        "source_model_sha256": source_model_sha,
        "source_arguments": str(source_arguments_path),
        "source_arguments_sha256": _sha256(source_arguments_path),
        "source_paper_evaluation": str(source_detailed_path),
        "source_paper_evaluation_sha256": _sha256(source_detailed_path),
    }
    _write_json(output / "source_checkpoint_receipt.json", source_receipt)

    env = _make_env(
        args.scenario, action_repeat=args.action_repeat, discount=args.discount
    )
    try:
        model = TargetCriticLaneDecoderSACV43.load(
            source_model,
            env=env,
            device=args.device,
            custom_objects={
                "policy_class": TargetCriticDecisionAlignedSACPolicyV43,
            },
        )
        if not isinstance(model.policy, TargetCriticDecisionAlignedSACPolicyV43):
            raise TypeError("checkpoint migration did not install the v4.3 policy")
        space_contract = validate_model_environment_spaces(model, env)
        started = time.perf_counter()
        report, diagnostics = evaluate_with_action_diagnostics_v4_3(
            model,
            env,
            episodes=args.evaluation_episodes,
            seed=args.evaluation_seed_start,
            trace_path=output / "action_diagnostics_decisions.jsonl",
        )
        evaluation_wall_seconds = time.perf_counter() - started
        provenance = build_evaluation_provenance_v2(
            model=model,
            env=env,
            report=report,
            algorithm="topo_v4_3_target_critic_decoder",
            scenario=args.scenario,
            traffic_protocol="frozen_60_20_20",
            evaluation_split="validation",
            episode_limit_profile="source",
            evaluation_seed_start=args.evaluation_seed_start,
            environment_factory=(
                "tools.replay_v4_3_checkpoint._make_env"
            ),
            space_contract=space_contract,
        )
    finally:
        env.close()

    method_metadata = {
        "implementation_id": V43_IMPLEMENTATION_IDS[
            "topo_v4_3_target_critic_decoder"
        ],
        "algorithm": "topo_v4_3_target_critic_decoder",
        "parent_checkpoint_algorithm": source_detailed.get("algorithm"),
        "parent_checkpoint_implementation_id": source_detailed.get(
            "implementation_id"
        ),
        "single_change": "deterministic_target_critic_lane_decoder",
        "deterministic_lane_decoder": "argmax_feasible_min_target_twin_q",
        "stochastic_training_policy_changed": False,
        "weights_changed": False,
        "checkpoint_selection_changed": False,
    }
    _write_json(output / "method_metadata.json", method_metadata)
    _write_json(output / "final_evaluation.json", report.summary.to_dict())
    _write_json(output / "action_diagnostics.json", diagnostics)
    detailed = {
        "schema_version": "topo-scene-v4.3.checkpoint-replay-evaluation/v1",
        "computed_from_real_run": True,
        "fabricated_values": False,
        "algorithm": "topo_v4_3_target_critic_decoder",
        "implementation_id": method_metadata["implementation_id"],
        "method_metadata": method_metadata,
        "scenario": args.scenario,
        "source_training_seed": args.source_training_seed,
        "evaluation_seed_start": args.evaluation_seed_start,
        "evaluation_split": "validation",
        "evaluation_provenance": provenance,
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "attribution_sha256": args.attribution_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
        "source_checkpoint_receipt_sha256": _sha256(
            output / "source_checkpoint_receipt.json"
        ),
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "completion_time_population_std_ddof": 0,
        **report.to_dict(),
    }
    _write_json(output / "paper_evaluation_detailed.json", detailed)
    print(
        json.dumps(
            {
                "output": str(output),
                "source_model_sha256": source_model_sha,
                **report.summary.to_dict(),
                "exact_target_critic_argmax_rate": diagnostics.get(
                    "exact_target_critic_argmax_rate"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
