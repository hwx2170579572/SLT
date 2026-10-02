"""Closed-loop validation replay for post-hoc v4.3 decoder attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch.hybrid_policy_v4 import DecisionAlignedHybridActor
from algos.sb3_torch.hybrid_policy_v4_3 import TargetCriticDecisionAlignedSACPolicyV43
from algos.sb3_torch.sac_v4_3 import TargetCriticLaneDecoderSACV43
from envs.sumo.paper_env_v4 import PaperSumoSceneEnvV4
from tools.action_diagnostics_v4_2 import evaluate_with_action_diagnostics_v4_2
from tools.action_diagnostics_v4_3 import evaluate_with_action_diagnostics_v4_3
from tools.paper_evaluation_contract import validate_model_environment_spaces


class ActorDeterministicAttributionPolicy(TargetCriticDecisionAlignedSACPolicyV43):
    """Restore the checkpoint actor's original deterministic argmax decoder."""

    deterministic_lane_decoder = "actor_argmax_post_hoc"

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        actor = self.actor
        if not isinstance(actor, DecisionAlignedHybridActor):
            raise TypeError("attribution replay requires DecisionAlignedHybridActor")
        return actor(observation, deterministic=deterministic)


class ActorConfidenceFusionAttributionPolicy(TargetCriticDecisionAlignedSACPolicyV43):
    """Use confident non-keep actor choices, otherwise target-twin-Q argmax."""

    deterministic_lane_decoder = "actor_confident_non_keep_else_target_critic_post_hoc"
    actor_non_keep_confidence_threshold = 0.90

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        actor = self.actor
        if not isinstance(actor, DecisionAlignedHybridActor):
            raise TypeError("attribution replay requires DecisionAlignedHybridActor")
        if not deterministic:
            return actor(observation, deterministic=False)

        batch = actor.all_action_samples(observation, deterministic_speed=True)
        features = self.critic_target.extract_features(
            observation, self.critic_target.features_extractor
        )
        q_heads = self.critic_target.all_q_from_features(features, batch.actions)
        minimum_q = th.stack(q_heads, dim=-1).min(dim=-1).values.squeeze(-1)
        masked_q = minimum_q.masked_fill(~batch.action_mask, -th.inf)
        target_indices = masked_q.argmax(dim=1)
        maxima = masked_q.max(dim=1, keepdim=True).values
        keep_tied = masked_q[:, 1:2] == maxima
        target_indices = th.where(
            keep_tied.squeeze(1), th.ones_like(target_indices), target_indices
        )

        actor_indices = batch.lane_probabilities.argmax(dim=1)
        actor_confidence = batch.lane_probabilities.gather(
            1, actor_indices[:, None]
        ).squeeze(1)
        override = (actor_indices != 1) & (
            actor_confidence >= float(self.actor_non_keep_confidence_threshold)
        )
        selected = th.where(override, actor_indices, target_indices)
        return actor._select(batch.actions, selected)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_env(
    scenario: str,
    action_repeat: int,
    discount: float,
    traffic_partition: str,
) -> PaperSumoSceneEnvV4:
    return PaperSumoSceneEnvV4(
        scenario=scenario,
        action_repeat=action_repeat,
        reward_discount=discount,
        traffic_partition=traffic_partition,
        episode_limit_profile="source",
        include_state_lstm=False,
        state_lstm_only=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--decoder", choices=("target", "actor", "fusion"), required=True
    )
    parser.add_argument("--actor-confidence-threshold", type=float, default=0.90)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--evaluation-seed-start", type=int, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument(
        "--traffic-partition",
        choices=("train", "validation"),
        default="validation",
    )
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.evaluation_seed_start < 0:
        raise ValueError("episodes and evaluation seed block must be valid")
    if not 0.0 <= args.actor_confidence_threshold <= 1.0:
        raise ValueError("actor confidence threshold must be in [0,1]")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    policy_classes = {
        "target": TargetCriticDecisionAlignedSACPolicyV43,
        "actor": ActorDeterministicAttributionPolicy,
        "fusion": ActorConfidenceFusionAttributionPolicy,
    }
    policy_class = policy_classes[args.decoder]
    env = _make_env(
        args.scenario,
        args.action_repeat,
        args.discount,
        args.traffic_partition,
    )
    try:
        model = TargetCriticLaneDecoderSACV43.load(
            checkpoint,
            env=env,
            device=args.device,
            custom_objects={"policy_class": policy_class},
        )
        if args.decoder == "fusion":
            model.policy.actor_non_keep_confidence_threshold = float(
                args.actor_confidence_threshold
            )
        space_contract = validate_model_environment_spaces(model, env)
        trace_path = output / "action_diagnostics_decisions.jsonl"
        started = time.perf_counter()
        if args.decoder == "target":
            report, diagnostics = evaluate_with_action_diagnostics_v4_3(
                model,
                env,
                episodes=args.episodes,
                seed=args.evaluation_seed_start,
                trace_path=trace_path,
            )
        else:
            report, diagnostics = evaluate_with_action_diagnostics_v4_2(
                model,
                env,
                episodes=args.episodes,
                seed=args.evaluation_seed_start,
                trace_path=trace_path,
            )
        wall_seconds = time.perf_counter() - started
    finally:
        env.close()

    metadata = {
        "schema_version": "topo-scene-v4.3.decoder-attribution/v1",
        "post_hoc_diagnostic_only": True,
        "counted_for_development_gate": False,
        "computed_from_real_closed_loop_rollout": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "training_seed": args.training_seed,
        "scenario": args.scenario,
        "decoder": args.decoder,
        "actor_non_keep_confidence_threshold": (
            args.actor_confidence_threshold if args.decoder == "fusion" else None
        ),
        "evaluation_seed_start": args.evaluation_seed_start,
        "evaluation_episodes": args.episodes,
        "action_repeat": args.action_repeat,
        "discount": args.discount,
        "traffic_partition": args.traffic_partition,
        "episode_limit_profile": "source",
        "space_contract": space_contract,
        "evaluation_wall_seconds": wall_seconds,
    }
    _write_json(output / "attribution_metadata.json", metadata)
    _write_json(output / "final_evaluation.json", report.summary.to_dict())
    _write_json(output / "action_diagnostics.json", diagnostics)
    result = {
        **metadata,
        "outcomes": report.summary.to_dict(),
        "lane_command_rates": diagnostics["lane_command_rates"],
        "route_action_window_match_rate": diagnostics[
            "route_action_window_match_rate"
        ],
        "actual_speed_mps": diagnostics["actual_speed_mps"],
        "trace_sha256": _sha256(trace_path),
    }
    _write_json(output / "attribution_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
