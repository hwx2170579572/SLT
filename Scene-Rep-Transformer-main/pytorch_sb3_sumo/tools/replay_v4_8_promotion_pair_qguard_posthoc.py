"""Closed-loop v4.8 promotion attribution with a 0.05 critic-regret guard.

This diagnostic replay is excluded from every scientific gate.  It holds the
checkpoint, validation episode block, environment, speed action, and all model
weights fixed while changing only deterministic lane decoding.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.sb3_torch.hybrid_policy_v4_9_qguard import (
    CriticRegretGuardActorFusionSACPolicyV49A,
    DECODER,
)
from algos.sb3_torch.sac_v4_9_qguard import CriticRegretGuardActorFusionSACV49A
from tools import replay_v4_6_pair_posthoc as base
from tools import replay_v4_8_promotion_pair_posthoc as frozen_v48
from tools.action_diagnostics_v4_9_qguard import (
    evaluate_with_action_diagnostics_v4_9_qguard,
    q_guard_decoder_integrity_passed,
    q_guard_decoder_integrity_summary,
)
from tools.checkpoint_selector_v4_4 import verify_checkpoint_zip
from tools.paper_evaluation_contract import validate_model_environment_spaces
from tools.train_paper_sb3_sumo_v4_5 import _make_paper_env_v4_5


def parser():
    value = base.parser()
    value.description = __doc__
    decoder_action = next(
        action for action in value._actions if action.dest == "decoder"
    )
    decoder_action.choices = (DECODER,)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.episodes <= 0 or args.evaluation_seed_start < 0:
        raise ValueError("episodes must be positive and seed start non-negative")
    provenance = frozen_v48.validate_source(args)

    run_dir = args.run_dir.resolve()
    model_path = (run_dir / args.model_file).resolve()
    arguments_path = run_dir / "arguments.json"
    verify_checkpoint_zip(model_path)
    checkpoint_sha = frozen_v48._sha256(model_path)
    source_arguments = frozen_v48._load(arguments_path)
    requested = dict(source_arguments["requested_raw_steps"])
    requested["evaluation_split"] = args.traffic_partition
    env_args = SimpleNamespace(**requested)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    trace_path = output_dir / "decisions.jsonl"
    env = _make_paper_env_v4_5(env_args, evaluation=True)
    try:
        model = CriticRegretGuardActorFusionSACV49A.load(
            model_path,
            env=env,
            device=args.device,
            custom_objects={
                "policy_class": CriticRegretGuardActorFusionSACPolicyV49A
            },
        )
        model.policy.set_training_mode(False)
        space_contract = validate_model_environment_spaces(model, env)
        started = time.perf_counter()
        report, diagnostics = evaluate_with_action_diagnostics_v4_9_qguard(
            model,
            env,
            episodes=args.episodes,
            seed=args.evaluation_seed_start,
            trace_path=trace_path,
        )
        evaluation_wall_seconds = time.perf_counter() - started
    finally:
        env.close()

    if not q_guard_decoder_integrity_passed(diagnostics):
        raise ValueError("post-hoc Q-guard decoder integrity failed")
    diagnostics_path = output_dir / "action_diagnostics.json"
    _write_json(diagnostics_path, diagnostics)
    payload: dict[str, Any] = {
        "schema_version": "topo-scene-v4.9a.q-regret-guard-posthoc/v1",
        "scientific_version": "v4.9a_q_regret_0_05_hypothesis",
        "post_hoc_diagnostic_only": True,
        "counted_for_development_gate": False,
        "counted_for_promotion_gate": False,
        "counted_for_formal_gate": False,
        "computed_from_real_closed_loop_rollout": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
        "promotion_result_reinterpretation_forbidden": True,
        "mutated_checkpoint": False,
        "source_run_directory": str(run_dir),
        "source_arguments_sha256": frozen_v48._sha256(arguments_path),
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint": str(model_path),
        "checkpoint_sha256": checkpoint_sha,
        "decoder": DECODER,
        "only_intervention": "require_actor_target_q_regret_at_most_0.05",
        "maximum_actor_target_q_regret": 0.05,
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
        "decoder_integrity": q_guard_decoder_integrity_summary(diagnostics),
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "outcomes": report.summary.to_dict(),
        "per_episode": report.to_dict()["episode_records"],
        "trace_sha256": frozen_v48._sha256(trace_path),
        "action_diagnostics_sha256": frozen_v48._sha256(diagnostics_path),
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
        "development_decision_sha256": args.development_decision_sha256,
        "source_algorithm_verified": frozen_v48.EXPECTED_ALGORITHM,
        "source_return_n_step_verified": 16,
        "source_bootstrap_discount_verified": "gamma_power_actual_horizon",
        "source_return_estimator_diagnostics": str(
            provenance["diagnostics_path"].resolve()
        ),
        "source_return_estimator_diagnostics_sha256": frozen_v48._sha256(
            provenance["diagnostics_path"]
        ),
        "source_selector_receipt": str(provenance["selector_path"].resolve()),
        "source_selector_receipt_sha256": frozen_v48._sha256(
            provenance["selector_path"]
        ),
    }
    result_path = output_dir / "attribution_result.json"
    _write_json(result_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "result_sha256": frozen_v48._sha256(result_path),
                "decoder": DECODER,
                "outcomes": payload["outcomes"],
                "q_regret_veto_records": diagnostics["q_regret_veto_records"],
                "post_hoc_diagnostic_only": True,
                "formal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
