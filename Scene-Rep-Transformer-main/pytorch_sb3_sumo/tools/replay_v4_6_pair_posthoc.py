"""Replay one frozen v4.6 checkpoint/decoder pair for post-hoc attribution.

The replay is deliberately excluded from every development, promotion, and
formal-test gate.  It only permits train or validation traffic and never writes
inside the frozen source run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45
from tools.action_diagnostics_v4_6 import (
    decoder_integrity_passed,
    decoder_integrity_summary,
    evaluate_with_action_diagnostics_v4_6,
    load_model_for_deployment,
)
from tools.checkpoint_decoder_selector_v4_6 import (
    FUSION_DECODER,
    TARGET_DECODER,
)
from tools.checkpoint_selector_v4_4 import verify_checkpoint_zip
from tools.paper_evaluation_contract import validate_model_environment_spaces
from tools.train_paper_sb3_sumo_v4_5 import _make_paper_env_v4_5


ALLOWED_PARTITIONS = ("train", "validation")
ALLOWED_DECODERS = (TARGET_DECODER, FUSION_DECODER)
ALLOWED_CHECKPOINT_KINDS = ("highest_training_success", "exact_final")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--run-dir", required=True, type=Path)
    value.add_argument("--model-file", required=True)
    value.add_argument("--checkpoint-kind", required=True, choices=ALLOWED_CHECKPOINT_KINDS)
    value.add_argument("--checkpoint-sha256", required=True)
    value.add_argument("--decoder", required=True, choices=ALLOWED_DECODERS)
    value.add_argument("--traffic-partition", required=True, choices=ALLOWED_PARTITIONS)
    value.add_argument("--evaluation-seed-start", required=True, type=int)
    value.add_argument("--episodes", required=True, type=int)
    value.add_argument("--device", default="cuda")
    value.add_argument("--output-dir", required=True, type=Path)
    value.add_argument("--experiment-contract-sha256", required=True)
    value.add_argument("--implementation-freeze-sha256", required=True)
    value.add_argument("--development-decision-sha256", required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.episodes <= 0 or args.evaluation_seed_start < 0:
        raise ValueError("episodes must be positive and seed start non-negative")

    run_dir = args.run_dir.resolve()
    model_path = (run_dir / args.model_file).resolve()
    arguments_path = run_dir / "arguments.json"
    if not model_path.is_file() or not arguments_path.is_file():
        raise FileNotFoundError(f"missing checkpoint or arguments under {run_dir}")
    verify_checkpoint_zip(model_path)
    checkpoint_sha = _sha256(model_path)
    if checkpoint_sha != args.checkpoint_sha256.lower():
        raise ValueError("checkpoint hash does not match the frozen selector receipt")

    source_arguments = _load_json(arguments_path)
    requested = dict(source_arguments["requested_raw_steps"])
    if requested.get("evaluation_split") != "validation":
        raise ValueError("source run did not use the frozen validation protocol")
    requested["evaluation_split"] = args.traffic_partition
    env_args = SimpleNamespace(**requested)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    trace_path = output_dir / "decisions.jsonl"
    env = _make_paper_env_v4_5(env_args, evaluation=True)
    try:
        model = load_model_for_deployment(
            ConfidentActorFusionSACV45,
            model_path,
            decoder=args.decoder,
            env=env,
            device=args.device,
        )
        model.policy.set_training_mode(False)
        space_contract = validate_model_environment_spaces(model, env)
        started = time.perf_counter()
        report, diagnostics = evaluate_with_action_diagnostics_v4_6(
            model,
            env,
            deployment_decoder=args.decoder,
            episodes=args.episodes,
            seed=args.evaluation_seed_start,
            trace_path=trace_path,
        )
        evaluation_wall_seconds = time.perf_counter() - started
    finally:
        env.close()

    if not decoder_integrity_passed(diagnostics):
        raise ValueError("post-hoc decoder integrity failed")
    _write_json(output_dir / "action_diagnostics.json", diagnostics)
    payload = {
        "schema_version": "topo-scene-v4.6.pair-posthoc-replay/v1",
        "post_hoc_diagnostic_only": True,
        "counted_for_development_gate": False,
        "counted_for_promotion_gate": False,
        "computed_from_real_closed_loop_rollout": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
        "mutated_checkpoint": False,
        "source_run_directory": str(run_dir),
        "source_arguments_sha256": _sha256(arguments_path),
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint": str(model_path),
        "checkpoint_sha256": checkpoint_sha,
        "decoder": args.decoder,
        "scenario": requested["scenario"],
        "training_seed": int(requested["seed"]),
        "traffic_partition": args.traffic_partition,
        "evaluation_seed_start": args.evaluation_seed_start,
        "evaluation_episodes": args.episodes,
        "action_repeat": int(requested["action_repeat"]),
        "discount": float(requested["discount"]),
        "traffic_protocol": requested["traffic_protocol"],
        "episode_limit_profile": requested["episode_limit_profile"],
        "space_contract": space_contract,
        "decoder_integrity": decoder_integrity_summary(diagnostics),
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "outcomes": report.summary.to_dict(),
        "per_episode": report.to_dict()["episode_records"],
        "trace_sha256": _sha256(trace_path),
        "action_diagnostics_sha256": _sha256(output_dir / "action_diagnostics.json"),
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
        "development_decision_sha256": args.development_decision_sha256,
    }
    _write_json(output_dir / "attribution_result.json", payload)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "checkpoint_kind": args.checkpoint_kind,
                "decoder": args.decoder,
                **payload["outcomes"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
