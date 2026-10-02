"""Read-only learned-score probe for v4.9.2 continuous-speed proposals.

The deployed policy still acts exactly as frozen.  At each visited observation,
the probe asks the frozen reward and collision target critics to score a fixed
diagnostic speed grid for every feasible lane.  Grid actions are never emitted
to the environment, so this is attribution evidence rather than a controller.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch as th

from algos.sb3_torch.evaluation import evaluate_model_detailed
from algos.sb3_torch.hybrid_policy_v4 import DecisionAlignedHybridActor
from algos.sb3_torch.hybrid_policy_v4_9_model import (
    CollisionConstrainedTargetCriticSACPolicyV49,
)
from algos.sb3_torch.sac_v4_9_model import CollisionConstrainedFusionSACV49
from envs.sumo.decision_alignment_v4 import LANE_COMMANDS
from tools.action_diagnostics_v2 import _ActionTracingEnvironment
from tools.action_diagnostics_v4_9_model import load_model_for_deployment_v4_9
from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER
from tools.train_paper_sb3_sumo_v4_5 import _make_paper_env_v4_5


DEFAULT_ATTRIBUTION = (
    ROOT
    / "results_topo_v4_9_2_promotion"
    / "attribution"
    / "promotion_model_attribution.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results_topo_v4_10_dev"
    / "stage_0_speed_proposal_probe"
    / "speed_proposal_coverage.json"
)
DEFAULT_TRACE_ROOT = DEFAULT_OUTPUT.parent / "traces"
DEFAULT_SPEED_GRID = (-0.95, -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75, 0.95)
FLOAT_TOLERANCE = 1e-7


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _selected_index(scores: th.Tensor, mask: th.Tensor) -> int:
    masked = scores.masked_fill(~mask, -th.inf)
    maximum = masked.max()
    if bool(mask[1]) and bool(th.isclose(masked[1], maximum, atol=0.0, rtol=0.0)):
        return 1
    return int(masked.argmax())


def learned_speed_proposal_snapshot(
    model: Any,
    observation: Mapping[str, np.ndarray],
    *,
    speed_grid: Iterable[float] = DEFAULT_SPEED_GRID,
) -> dict[str, Any]:
    """Score diagnostic speeds without changing the action sent to SUMO."""

    policy = model.policy
    if not isinstance(policy, CollisionConstrainedTargetCriticSACPolicyV49):
        raise TypeError("speed proposal probe requires the v4.9 target policy")
    actor = policy.actor
    if not isinstance(actor, DecisionAlignedHybridActor):
        raise TypeError("speed proposal probe requires the hybrid actor")
    grid_values = tuple(float(value) for value in speed_grid)
    if not grid_values or any(value < -1.0 or value > 1.0 for value in grid_values):
        raise ValueError("diagnostic speed grid must be non-empty and within [-1,1]")
    if tuple(sorted(set(grid_values))) != grid_values:
        raise ValueError("diagnostic speed grid must be strictly increasing")

    tensor_observation, _ = policy.obs_to_tensor(observation)
    with th.no_grad():
        actor_batch = actor.all_action_samples(
            tensor_observation, deterministic_speed=True
        )
        reward_q, collision_value, actor_score = policy._risk_adjusted_target_values(
            tensor_observation, actor_batch
        )
        action_mask = actor_batch.action_mask[0]
        baseline_index = _selected_index(actor_score[0], action_mask)

        grid = th.as_tensor(
            grid_values,
            device=actor_batch.actions.device,
            dtype=actor_batch.actions.dtype,
        )
        lane_codes = actor.lane_codes.to(grid)
        grid_speeds = grid.repeat(len(LANE_COMMANDS))
        grid_lanes = lane_codes.repeat_interleave(grid.numel())
        grid_actions = th.stack((grid_speeds, grid_lanes), dim=1)
        actor_actions = actor_batch.actions[0]
        candidate_actions = th.cat((grid_actions, actor_actions), dim=0)

        reward_target = policy.critic_target
        reward_features = reward_target.extract_features(
            tensor_observation, reward_target.features_extractor
        ).repeat(candidate_actions.shape[0], 1)
        reward_heads = reward_target.forward_from_features(
            reward_features, candidate_actions
        )
        candidate_reward = th.stack(reward_heads, dim=-1).min(
            dim=-1
        ).values.squeeze(-1)

        collision_target = policy.collision_critic_target
        collision_features = collision_target.extract_features(
            tensor_observation, collision_target.features_extractor
        ).repeat(candidate_actions.shape[0], 1)
        collision_heads = collision_target.forward_from_features(
            collision_features, candidate_actions
        )
        candidate_collision = th.stack(
            [th.sigmoid(head) for head in collision_heads], dim=-1
        ).max(dim=-1).values.squeeze(-1)
        candidate_score = (
            candidate_reward
            - float(policy.collision_risk_coef) * candidate_collision
        )

        proposals_per_lane = grid.numel()
        best_indices: list[int] = []
        lane_rows: list[dict[str, Any] | None] = []
        for lane_index, lane_code in enumerate(LANE_COMMANDS):
            if not bool(action_mask[lane_index]):
                lane_rows.append(None)
                best_indices.append(-1)
                continue
            grid_start = lane_index * proposals_per_lane
            indices = list(
                range(grid_start, grid_start + proposals_per_lane)
            )
            actor_action_index = len(LANE_COMMANDS) * proposals_per_lane + lane_index
            maximum = max(float(candidate_score[index]) for index in indices)
            actor_lane_score = float(candidate_score[actor_action_index])
            if actor_lane_score >= maximum - FLOAT_TOLERANCE:
                best_index = actor_action_index
                source = "actor"
            else:
                best_index = max(indices, key=lambda index: float(candidate_score[index]))
                source = "diagnostic_grid"
            best_indices.append(best_index)
            lane_rows.append(
                {
                    "lane_index": lane_index,
                    "lane": int(lane_code),
                    "source": source,
                    "actor_speed_normalized": float(actor_actions[lane_index, 0]),
                    "actor_speed_mps": float((actor_actions[lane_index, 0] + 1.0) * 5.0),
                    "actor_reward_q": float(reward_q[0, lane_index]),
                    "actor_collision_value": float(collision_value[0, lane_index]),
                    "actor_score": float(actor_score[0, lane_index]),
                    "best_speed_normalized": float(candidate_actions[best_index, 0]),
                    "best_speed_mps": float((candidate_actions[best_index, 0] + 1.0) * 5.0),
                    "best_reward_q": float(candidate_reward[best_index]),
                    "best_collision_value": float(candidate_collision[best_index]),
                    "best_score": float(candidate_score[best_index]),
                    "score_gain": float(
                        candidate_score[best_index] - actor_score[0, lane_index]
                    ),
                    "collision_value_reduction": float(
                        collision_value[0, lane_index]
                        - candidate_collision[best_index]
                    ),
                }
            )

        feasible_best = [
            index for index in best_indices if index >= 0
        ]
        best_score_value = max(float(candidate_score[index]) for index in feasible_best)
        baseline_lane_best_index = best_indices[baseline_index]
        if (
            float(candidate_score[baseline_lane_best_index])
            >= best_score_value - FLOAT_TOLERANCE
        ):
            global_best_index = baseline_lane_best_index
        else:
            global_best_index = max(
                feasible_best, key=lambda index: float(candidate_score[index])
            )
        global_lane_index = int(
            round(float(candidate_actions[global_best_index, 1]))
        ) + 1
        baseline_row = lane_rows[baseline_index]
        global_row = lane_rows[global_lane_index]
        if baseline_row is None or global_row is None:
            raise AssertionError("selected learned proposal must be feasible")

    return {
        "diagnostic_only": True,
        "action_rewritten": False,
        "speed_grid_normalized": list(grid_values),
        "speed_grid_mps": [float((value + 1.0) * 5.0) for value in grid_values],
        "baseline_selected_lane_index": baseline_index,
        "baseline_selected_lane": int(LANE_COMMANDS[baseline_index]),
        "baseline_speed_normalized": float(actor_actions[baseline_index, 0]),
        "baseline_speed_mps": float((actor_actions[baseline_index, 0] + 1.0) * 5.0),
        "baseline_reward_q": float(reward_q[0, baseline_index]),
        "baseline_collision_value": float(collision_value[0, baseline_index]),
        "baseline_score": float(actor_score[0, baseline_index]),
        "same_lane_best": baseline_row,
        "global_best": global_row,
        "same_lane_score_gain": float(baseline_row["score_gain"]),
        "same_lane_collision_value_reduction": float(
            baseline_row["collision_value_reduction"]
        ),
        "same_lane_speed_delta_mps": float(
            baseline_row["best_speed_mps"] - baseline_row["actor_speed_mps"]
        ),
        "global_score_gain": float(
            global_row["best_score"] - float(actor_score[0, baseline_index])
        ),
        "global_collision_value_reduction": float(
            collision_value[0, baseline_index] - global_row["best_collision_value"]
        ),
        "global_speed_delta_mps": float(
            global_row["best_speed_mps"]
            - float((actor_actions[baseline_index, 0] + 1.0) * 5.0)
        ),
        "global_lane_changed": global_lane_index != baseline_index,
        "per_lane": lane_rows,
    }


class _ProposalProbeEnvironment(_ActionTracingEnvironment):
    def __init__(
        self,
        env: Any,
        model: Any,
        *,
        speed_grid: tuple[float, ...],
    ) -> None:
        super().__init__(env, model)
        self.speed_grid = speed_grid
        self._current_observation: Mapping[str, np.ndarray] | None = None

    def reset(self, *args: Any, **kwargs: Any):
        observation, info = super().reset(*args, **kwargs)
        self._current_observation = observation
        return observation, info

    def step(self, action: Any):
        if self._current_observation is None:
            raise RuntimeError("proposal probe step before reset")
        snapshot = learned_speed_proposal_snapshot(
            self.model,
            self._current_observation,
            speed_grid=self.speed_grid,
        )
        output = super().step(action)
        self.records[-1]["learned_speed_proposal_probe"] = snapshot
        self._current_observation = output[0]
        return output


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "population_std": None,
            "minimum": None,
            "p10": None,
            "median": None,
            "p90": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "population_std": float(array.std()),
        "minimum": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "maximum": float(array.max()),
    }


def summarize_probe_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    probes = [row["learned_speed_proposal_probe"] for row in rows]
    if not probes:
        return {"decision_count": 0}
    same_gain = [float(row["same_lane_score_gain"]) for row in probes]
    same_risk = [
        float(row["same_lane_collision_value_reduction"]) for row in probes
    ]
    same_speed = [float(row["same_lane_speed_delta_mps"]) for row in probes]
    global_gain = [float(row["global_score_gain"]) for row in probes]
    global_risk = [
        float(row["global_collision_value_reduction"]) for row in probes
    ]
    global_speed = [float(row["global_speed_delta_mps"]) for row in probes]
    improved = [value > FLOAT_TOLERANCE for value in same_gain]
    safer_improved = [
        gain > FLOAT_TOLERANCE and risk > FLOAT_TOLERANCE
        for gain, risk in zip(same_gain, same_risk)
    ]
    return {
        "decision_count": len(probes),
        "same_lane_diagnostic_grid_selected_rate": sum(
            row["same_lane_best"]["source"] == "diagnostic_grid" for row in probes
        )
        / len(probes),
        "same_lane_strict_score_improvement_rate": sum(improved) / len(probes),
        "same_lane_safer_and_higher_score_rate": sum(safer_improved) / len(probes),
        "same_lane_lower_speed_when_improved_rate": (
            sum(speed < 0.0 for speed, flag in zip(same_speed, improved) if flag)
            / max(sum(improved), 1)
        ),
        "global_lane_changed_rate": sum(
            bool(row["global_lane_changed"]) for row in probes
        )
        / len(probes),
        "same_lane_score_gain": _numeric_summary(same_gain),
        "same_lane_collision_value_reduction": _numeric_summary(same_risk),
        "same_lane_speed_delta_mps": _numeric_summary(same_speed),
        "global_score_gain": _numeric_summary(global_gain),
        "global_collision_value_reduction": _numeric_summary(global_risk),
        "global_speed_delta_mps": _numeric_summary(global_speed),
    }


def _annotate_outcomes(
    rows: list[dict[str, Any]], report: Mapping[str, Any]
) -> None:
    episodes = report["episode_records"]
    for row in rows:
        record = episodes[int(row["episode"])]
        if bool(record["collision"]):
            outcome = "collision"
        elif bool(record["success"]):
            outcome = "success"
        elif bool(record["off_route"]):
            outcome = "off_route"
        else:
            outcome = "timeout"
        row["eventual_episode_outcome"] = outcome


def summarize_probe(
    rows: list[dict[str, Any]], report: Mapping[str, Any]
) -> dict[str, Any]:
    _annotate_outcomes(rows, report)
    output: dict[str, Any] = {
        "all_decisions": summarize_probe_rows(rows),
        "by_episode_outcome": {},
        "collision_terminal_windows": {},
    }
    outcomes = sorted({str(row["eventual_episode_outcome"]) for row in rows})
    for outcome in outcomes:
        subset = [
            row for row in rows if row["eventual_episode_outcome"] == outcome
        ]
        output["by_episode_outcome"][outcome] = summarize_probe_rows(subset)
    collision_episodes = sorted(
        {
            int(row["episode"])
            for row in rows
            if row["eventual_episode_outcome"] == "collision"
        }
    )
    for window in (1, 5, 10):
        terminal: list[dict[str, Any]] = []
        for episode in collision_episodes:
            episode_rows = [row for row in rows if int(row["episode"]) == episode]
            terminal.extend(episode_rows[-window:])
        output["collision_terminal_windows"][str(window)] = summarize_probe_rows(
            terminal
        )
    return output


def evaluation_outcomes(report: Mapping[str, Any]) -> dict[str, float]:
    """Extract the public rates from a detailed evaluation report."""

    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("detailed evaluation report is missing its summary")
    return {
        key: float(summary[key])
        for key in (
            "success_rate",
            "collision_rate",
            "off_route_rate",
            "timeout_rate",
        )
    }


def _candidate_pairs(attribution: Mapping[str, Any]) -> list[dict[str, Any]]:
    if attribution.get("decision_allowed") is not True:
        raise ValueError("complete attribution decision is not allowed")
    if attribution.get("partial_matrix") is not False:
        raise ValueError("speed proposal probe refuses partial promotion evidence")
    if int(attribution.get("pair_count", 0)) != 6:
        raise ValueError("speed proposal probe requires all six promotion pairs")
    pairs = list(attribution["pairs"])
    return sorted(pairs, key=lambda row: (row["scenario"], int(row["seed"])))


def run_pair(
    pair: Mapping[str, Any],
    *,
    episodes: int,
    device: str,
    speed_grid: tuple[float, ...],
    trace_root: Path,
) -> dict[str, Any]:
    scenario = str(pair["scenario"])
    seed = int(pair["seed"])
    run_dir = Path(pair["candidate"]["run_dir"])
    arguments_path = run_dir / "arguments.json"
    checkpoint = run_dir / "selected_model.zip"
    if not arguments_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"missing selected v4.9.2 evidence in {run_dir}")
    arguments = _load_json(arguments_path)
    requested = dict(arguments["requested_raw_steps"])
    requested["evaluation_split"] = "validation"
    args = SimpleNamespace(**requested)
    evaluation_seed = int(requested["evaluation_seed_start"])

    env = _make_paper_env_v4_5(args, evaluation=True)
    model = None
    try:
        model = load_model_for_deployment_v4_9(
            CollisionConstrainedFusionSACV49,
            checkpoint,
            decoder=TARGET_DECODER,
            env=None,
            device=device,
        )
        tracing_env = _ProposalProbeEnvironment(
            env, model, speed_grid=speed_grid
        )
        report = evaluate_model_detailed(
            model,
            tracing_env,
            episodes=episodes,
            seed=evaluation_seed,
            deterministic=True,
            policy_action_hold=1,
        ).to_dict()
        rows = tracing_env.records
        if len(report["episode_records"]) != episodes or not rows:
            raise ValueError("proposal probe rollout is incomplete")
        for row in rows:
            probe = row["learned_speed_proposal_probe"]
            if int(row["lane_command"]) != int(probe["baseline_selected_lane"]):
                raise ValueError("diagnostic probe changed or mismatched deployed lane action")
            observed_speed = float(row["action_longitudinal"])
            if abs(observed_speed - float(probe["baseline_speed_normalized"])) > 1e-5:
                raise ValueError("diagnostic probe changed or mismatched deployed speed")
        summary = summarize_probe(rows, report)
        trace_path = trace_root / f"{scenario}__seed{seed}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        return {
            "scenario": scenario,
            "seed": seed,
            "source_run": _relative(run_dir),
            "source_arguments_sha256": _sha256(arguments_path),
            "source_checkpoint_sha256": _sha256(checkpoint),
            "evaluation_seed_start": evaluation_seed,
            "episodes": episodes,
            "outcomes": evaluation_outcomes(report),
            "promotion_outcomes": pair["candidate"]["evaluation_summary"],
            "probe": summary,
            "trace": _relative(trace_path),
            "trace_sha256": _sha256(trace_path),
            "trace_records": len(rows),
            "deployed_actions_changed_by_probe": False,
        }
    finally:
        env.close()
        del model
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        all_decisions = run["probe"]["all_decisions"]
        terminal10 = run["probe"]["collision_terminal_windows"]["10"]
        promotion = run["promotion_outcomes"]
        rows.append(
            {
                "scenario": run["scenario"],
                "seed": run["seed"],
                "promotion_success_rate": promotion["success_rate"],
                "promotion_collision_rate": promotion["collision_rate"],
                "same_lane_score_improvement_rate": all_decisions[
                    "same_lane_strict_score_improvement_rate"
                ],
                "same_lane_safer_and_higher_score_rate": all_decisions[
                    "same_lane_safer_and_higher_score_rate"
                ],
                "same_lane_mean_score_gain": all_decisions[
                    "same_lane_score_gain"
                ]["mean"],
                "collision_terminal10_decisions": terminal10["decision_count"],
                "collision_terminal10_safer_and_higher_score_rate": (
                    terminal10.get("same_lane_safer_and_higher_score_rate")
                ),
                "collision_terminal10_mean_score_gain": (
                    terminal10.get("same_lane_score_gain", {}).get("mean")
                ),
                "collision_terminal10_mean_collision_value_reduction": (
                    terminal10.get(
                        "same_lane_collision_value_reduction", {}
                    ).get("mean")
                ),
                "deployed_actions_changed_by_probe": False,
            }
        )
    return {
        "per_run": rows,
        "all_runs_completed": len(rows) == 6,
        "run_count": len(rows),
        "interpretation_boundary": (
            "critic self-consistency identifies proposal coverage only; it does "
            "not prove that a counterfactual grid action is safe in closed loop"
        ),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--attribution", type=Path, default=DEFAULT_ATTRIBUTION)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    value.add_argument("--episodes", type=int, default=12)
    value.add_argument("--device", default="cuda")
    value.add_argument(
        "--speed-grid",
        default=",".join(str(value) for value in DEFAULT_SPEED_GRID),
    )
    value.add_argument("--pair", default="all")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    speed_grid = tuple(float(value) for value in args.speed_grid.split(","))
    attribution_path = args.attribution.resolve()
    attribution = _load_json(attribution_path)
    pairs = _candidate_pairs(attribution)
    if args.pair != "all":
        requested = {
            item.strip() for item in str(args.pair).split(",") if item.strip()
        }
        pairs = [
            row
            for row in pairs
            if f"{row['scenario']}__seed{int(row['seed'])}" in requested
        ]
        if len(pairs) != len(requested):
            raise ValueError(f"unknown or duplicate pair selector: {args.pair}")

    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for pair in pairs:
        identity = f"{pair['scenario']}__seed{int(pair['seed'])}"
        print(json.dumps({"pair": identity, "status": "running"}), flush=True)
        try:
            run = run_pair(
                pair,
                episodes=args.episodes,
                device=args.device,
                speed_grid=speed_grid,
                trace_root=args.trace_root.resolve(),
            )
            runs.append(run)
            print(json.dumps({"pair": identity, "status": "complete"}), flush=True)
        except Exception as exc:
            failures.append(
                {
                    "pair": identity,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                json.dumps(
                    {"pair": identity, "status": "failed", "error": failures[-1]["error"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    payload = {
        "schema_version": "topo-scene-v4.10.stage0-speed-proposal-coverage/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_kind": "read_only_frozen_learned_critic_counterfactual_query",
        "computed_from_real_checkpoints_and_validation_rollouts": True,
        "fabricated_values": False,
        "scientific_model_changed": False,
        "source_run_trees_modified": False,
        "formal_test_accessed": False,
        "diagnostic_grid_used_for_deployment": False,
        "deployed_actions_changed_by_probe": False,
        "attribution": _relative(attribution_path),
        "attribution_sha256": _sha256(attribution_path),
        "speed_grid_normalized": list(speed_grid),
        "speed_grid_mps": [float((value + 1.0) * 5.0) for value in speed_grid],
        "episodes_per_pair": args.episodes,
        "execution_policy": "attempt_all_selected_pairs_then_summarize",
        "failure_does_not_cancel_later_pairs": True,
        "runs": runs,
        "failures": failures,
        "aggregate": aggregate_runs(runs),
        "decision_allowed": len(failures) == 0 and len(runs) == len(pairs),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["decision_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
