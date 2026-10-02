"""Replay frozen v4.2 checkpoints for post-hoc C1 seed/block attribution.

This tool never trains or mutates a checkpoint.  It records actor-probability,
twin-critic-Q, route-event, and outcome evidence on validation traffic only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch.sac_v4_2 import FactorizedEntropyHybridSACV42
from envs.sumo.decision_alignment_v4 import LANE_COMMANDS
from tools.action_diagnostics_v4_2 import evaluate_with_action_diagnostics_v4_2
from tools.analyze_v4_1_deterministic_gap import _route_window_attribution, _sha256
from tools.run_topo_v4_2_r1_experiments import route_event_metrics
from tools.train_paper_sb3_sumo_v4_2 import _make_paper_env_v4_2


class V42DiagnosticDecoder:
    """Actor, online-critic, or target-critic deterministic lane decoder."""

    def __init__(self, model: FactorizedEntropyHybridSACV42, decoder: str) -> None:
        allowed = {"actor_argmax", "critic_greedy", "critic_target_greedy"}
        if decoder not in allowed:
            raise ValueError(f"unsupported decoder: {decoder}")
        self.model = model
        self.decoder = decoder
        self.records: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def predict(
        self,
        observation: Any,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any]:
        del episode_start
        if not deterministic:
            raise ValueError("attribution decoder is deterministic-only")
        tensor_observation, vectorized = self.model.policy.obs_to_tensor(observation)
        actor, online_critic, target_critic = self.model._hybrid_modules()
        selected_critic = (
            target_critic if self.decoder == "critic_target_greedy" else online_critic
        )
        with torch.no_grad():
            batch = actor.all_action_samples(
                tensor_observation, deterministic_speed=True
            )
            features = selected_critic.extract_features(
                tensor_observation, selected_critic.features_extractor
            )
            q_heads = selected_critic.all_q_from_features(features, batch.actions)
            min_q = torch.stack(q_heads, dim=-1).min(dim=-1).values.squeeze(-1)
            masked_q = min_q.masked_fill(~batch.action_mask, -torch.inf)
            actor_indices = batch.lane_probabilities.argmax(dim=1)
            critic_indices = masked_q.argmax(dim=1)
            selected_indices = (
                actor_indices if self.decoder == "actor_argmax" else critic_indices
            )
            selected_actions = actor._select(batch.actions, selected_indices)

        for row_index in range(selected_actions.shape[0]):
            valid = batch.action_mask[row_index].detach().cpu().tolist()
            q_values = min_q[row_index].detach().cpu().tolist()
            self.records.append(
                {
                    "lane_probabilities": batch.lane_probabilities[row_index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "valid_lane_actions": valid,
                    "min_twin_q": [
                        float(value) if is_valid else None
                        for value, is_valid in zip(q_values, valid)
                    ],
                    "actor_argmax_lane": int(
                        LANE_COMMANDS[int(actor_indices[row_index])]
                    ),
                    "critic_greedy_lane": int(
                        LANE_COMMANDS[int(critic_indices[row_index])]
                    ),
                    "selected_lane": int(
                        LANE_COMMANDS[int(selected_indices[row_index])]
                    ),
                }
            )

        actions = selected_actions.detach().cpu().numpy()
        if not vectorized:
            actions = actions.squeeze(axis=0)
        return actions, state


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(allow_abbrev=False)
    value.add_argument("--run-dir", required=True, type=Path)
    value.add_argument("--model-file", default="final_model.zip")
    value.add_argument(
        "--decoder",
        required=True,
        choices=("actor_argmax", "critic_greedy", "critic_target_greedy"),
    )
    value.add_argument("--episodes", required=True, type=int)
    value.add_argument("--evaluation-seed-start", required=True, type=int)
    value.add_argument("--device", default="cuda")
    value.add_argument("--output-dir", required=True, type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    model_path = (run_dir / args.model_file).resolve()
    arguments_path = run_dir / "arguments.json"
    if not model_path.is_file() or not arguments_path.is_file():
        raise FileNotFoundError(f"missing model or arguments under {run_dir}")
    if args.episodes <= 0 or args.evaluation_seed_start < 0:
        raise ValueError("episodes must be positive and seed start non-negative")

    arguments = json.loads(arguments_path.read_text(encoding="utf-8"))
    requested = dict(arguments["requested_raw_steps"])
    requested["evaluation_split"] = "validation"
    env_args = SimpleNamespace(**requested)

    model = FactorizedEntropyHybridSACV42.load(model_path, device=args.device)
    model.policy.set_training_mode(False)
    predictor = V42DiagnosticDecoder(model, args.decoder)
    env = _make_paper_env_v4_2(env_args, evaluation=True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_label = {
        "final_model": "final",
        "best_training_success_model": "best_train",
    }.get(model_path.stem, model_path.stem[:20])
    stem = (
        f"train_s{requested['seed']}__{model_label}__{args.decoder}__"
        f"v{args.evaluation_seed_start}"
    )
    trace_path = output_dir / f"{stem}__decisions.jsonl"
    try:
        report, action_diagnostics = evaluate_with_action_diagnostics_v4_2(
            predictor,
            env,
            episodes=args.episodes,
            seed=args.evaluation_seed_start,
            trace_path=trace_path,
        )
    finally:
        env.close()

    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payload = {
        "schema_version": "topo-scene-v4.2.r1.failure-attribution-replay/v1",
        "post_hoc_diagnostic_only": True,
        "formal_test_accessed": False,
        "mutated_checkpoint": False,
        "run_directory": str(run_dir),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "arguments_sha256": _sha256(arguments_path),
        "decoder": args.decoder,
        "scenario": requested["scenario"],
        "training_seed": requested["seed"],
        "evaluation_split": "validation",
        "evaluation_seed_start": args.evaluation_seed_start,
        "evaluation_episodes": args.episodes,
        "outcomes": report.summary.to_dict(),
        "action_diagnostics": action_diagnostics,
        "route_event_metrics": route_event_metrics(trace_rows),
        "decoder_attribution": _route_window_attribution(
            predictor.records, trace_rows
        ),
        "trace": {
            "path": str(trace_path),
            "sha256": _sha256(trace_path),
            "records": len(trace_rows),
        },
    }
    output_path = output_dir / f"{stem}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output_path), **payload["outcomes"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
