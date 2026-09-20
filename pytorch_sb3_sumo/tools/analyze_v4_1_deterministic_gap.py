"""Post-hoc v4.1 attribution for stochastic-train/deterministic-eval mismatch.

This tool never trains or mutates a checkpoint.  It replays a frozen v4.1 model
on the validation partition while comparing the actor-mode lane decision with a
twin-critic minimum-Q lane decision at the actor's deterministic per-lane speed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch.sac_v4 import DecisionAlignedHybridSACV4
from envs.sumo.decision_alignment_v4 import LANE_COMMANDS, lane_command_index
from tools.action_diagnostics_v4 import evaluate_with_action_diagnostics_v4
from tools.train_paper_sb3_sumo_v4 import _make_paper_env_v4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


class DiagnosticDeterministicDecoder:
    """Model-compatible predictor with actor and critic-greedy lane decoders."""

    def __init__(self, model: DecisionAlignedHybridSACV4, decoder: str) -> None:
        if decoder not in {"actor_argmax", "critic_greedy"}:
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
        actor, critic, _ = self.model._hybrid_modules()
        with torch.no_grad():
            batch = actor.all_action_samples(
                tensor_observation, deterministic_speed=True
            )
            features = critic.extract_features(
                tensor_observation, critic.features_extractor
            )
            q_heads = critic.all_q_from_features(features, batch.actions)
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
                    "actor_argmax_lane": int(LANE_COMMANDS[int(actor_indices[row_index])]),
                    "critic_greedy_lane": int(LANE_COMMANDS[int(critic_indices[row_index])]),
                    "selected_lane": int(LANE_COMMANDS[int(selected_indices[row_index])]),
                }
            )

        actions = selected_actions.detach().cpu().numpy()
        if not vectorized:
            actions = actions.squeeze(axis=0)
        return actions, state


def _route_window_attribution(
    decoder_records: list[dict[str, Any]], trace_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(decoder_records) != len(trace_rows):
        raise ValueError(
            "decoder/action trace length mismatch: "
            f"{len(decoder_records)} != {len(trace_rows)}"
        )
    route_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for decoder_row, trace_row in zip(decoder_records, trace_rows):
        if bool(trace_row["pre_action_route_intent_valid"]) and int(
            trace_row["pre_action_route_intent"]
        ) != 0:
            route_rows.append((decoder_row, trace_row))

    probability_margins: list[float] = []
    q_margins: list[float] = []
    actor_matches = 0
    critic_matches = 0
    selected_matches = 0
    for decoder_row, trace_row in route_rows:
        route_lane = int(trace_row["pre_action_route_intent"])
        route_index = lane_command_index(route_lane)
        keep_index = lane_command_index(0)
        probabilities = decoder_row["lane_probabilities"]
        q_values = decoder_row["min_twin_q"]
        probability_margins.append(
            float(probabilities[route_index]) - float(probabilities[keep_index])
        )
        route_q = q_values[route_index]
        keep_q = q_values[keep_index]
        if route_q is None or keep_q is None:
            raise ValueError("route window contains an infeasible route/keep action")
        q_margins.append(float(route_q) - float(keep_q))
        actor_matches += int(decoder_row["actor_argmax_lane"] == route_lane)
        critic_matches += int(decoder_row["critic_greedy_lane"] == route_lane)
        selected_matches += int(decoder_row["selected_lane"] == route_lane)

    count = len(route_rows)
    disagreements = sum(
        int(row["actor_argmax_lane"] != row["critic_greedy_lane"])
        for row in decoder_records
    )
    return {
        "decision_records": len(decoder_records),
        "actor_critic_disagreements": disagreements,
        "actor_critic_disagreement_rate": (
            disagreements / len(decoder_records) if decoder_records else None
        ),
        "route_window_samples": count,
        "actor_argmax_route_match_rate": actor_matches / count if count else None,
        "critic_greedy_route_match_rate": critic_matches / count if count else None,
        "selected_route_match_rate": selected_matches / count if count else None,
        "actor_route_minus_keep_probability_margin": _finite_summary(
            probability_margins
        ),
        "critic_route_minus_keep_q_margin": _finite_summary(q_margins),
        "critic_route_q_above_keep_rate": (
            sum(value > 0.0 for value in q_margins) / count if count else None
        ),
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model-file", default="final_model.zip")
    parser.add_argument(
        "--decoder",
        required=True,
        choices=("actor_argmax", "critic_greedy"),
    )
    parser.add_argument("--episodes", required=True, type=int)
    parser.add_argument("--evaluation-seed-start", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
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

    model = DecisionAlignedHybridSACV4.load(model_path, device=args.device)
    model.policy.set_training_mode(False)
    predictor = DiagnosticDeterministicDecoder(model, args.decoder)
    env = _make_paper_env_v4(env_args, evaluation=True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_dir.name}__{model_path.stem}__{args.decoder}"
    trace_path = output_dir / f"{stem}__decisions.jsonl"
    try:
        report, action_diagnostics = evaluate_with_action_diagnostics_v4(
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
        "schema_version": "topo-scene-v4.attribution-deterministic-gap/v1",
        "post_hoc_diagnostic_only": True,
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
    print(json.dumps({"output": str(output_path), **payload["outcomes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
