"""Trace-backed attribution for the v4.9 learned collision-value model.

The analysis is deliberately observational.  It measures whether the learned
collision value separates collision and non-collision trajectories and whether
the deployed learned score selected an action with higher predicted collision
value than another feasible action.  It never proposes or applies an action
override, geometric label, kinematic threshold, or safety projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence


WINDOWS = (1, 5, 10)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _decoder(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    target = row.get("target_critic_decoder")
    if isinstance(target, dict):
        return "target_critic", target
    fusion = row.get("fusion_decoder")
    if isinstance(fusion, dict):
        return "fusion", fusion
    guarded = row.get("q_guard_fusion_decoder")
    if isinstance(guarded, dict):
        return "q_guard_fusion", guarded
    raise KeyError("trace row has no recognized learned-model decoder")


def _components(row: dict[str, Any]) -> dict[str, Any]:
    decoder_kind, decoder = _decoder(row)
    valid = [bool(item) for item in decoder["valid_lane_actions"]]
    collision = decoder["maximum_target_twin_collision_value"]
    reward = decoder["minimum_target_twin_q"]
    score = decoder["risk_adjusted_target_score"]
    if not (len(valid) == len(collision) == len(reward) == len(score) == 3):
        raise ValueError("learned decoder must expose exactly three lane actions")
    feasible = [index for index, allowed in enumerate(valid) if allowed]
    if not feasible:
        raise ValueError("learned decoder has no feasible action")
    collision_values = {
        index: _finite(collision[index], name="collision value")
        for index in feasible
    }
    reward_values = {
        index: _finite(reward[index], name="reward value") for index in feasible
    }
    score_values = {
        index: _finite(score[index], name="risk-adjusted score")
        for index in feasible
    }
    coefficient = _finite(decoder["collision_risk_coef"], name="risk coefficient")
    for index in feasible:
        expected = reward_values[index] - coefficient * collision_values[index]
        if not math.isclose(score_values[index], expected, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("risk-adjusted score equation drifted")
    selected_lane = int(row["lane_command"])
    lane_codes = (-1, 0, 1)
    try:
        selected_index = lane_codes.index(selected_lane)
    except ValueError as error:
        raise ValueError(f"unsupported lane command: {selected_lane}") from error
    if selected_index not in feasible:
        raise ValueError("selected action is not feasible")
    safest_index = min(feasible, key=lambda index: (collision_values[index], index))
    return {
        "decoder_kind": decoder_kind,
        "selected_index": selected_index,
        "safest_index": safest_index,
        "selected_collision_value": collision_values[selected_index],
        "minimum_feasible_collision_value": collision_values[safest_index],
        "collision_value_regret": (
            collision_values[selected_index] - collision_values[safest_index]
        ),
        "selected_reward_q": reward_values[selected_index],
        "safest_action_reward_q": reward_values[safest_index],
        "selected_risk_adjusted_score": score_values[selected_index],
        "safest_action_risk_adjusted_score": score_values[safest_index],
        "selected_differs_from_minimum_collision_action": (
            selected_index != safest_index
        ),
    }


def _rank_auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """Return tie-aware P(positive > negative) without third-party packages."""

    if not positive or not negative:
        return None
    wins = 0.0
    for positive_value in positive:
        for negative_value in negative:
            if positive_value > negative_value:
                wins += 1.0
            elif positive_value == negative_value:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def _mean_or_none(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def _load_trace(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics = _read_json(run_dir / "action_diagnostics.json")
    trace_metadata = diagnostics.get("trace")
    if not isinstance(trace_metadata, dict):
        raise TypeError(f"missing trace metadata: {run_dir}")
    trace_path = Path(str(trace_metadata["path"]))
    if not trace_path.is_absolute():
        trace_path = run_dir / trace_path
    if not trace_path.is_file():
        fallback = run_dir / "action_diagnostics_decisions.jsonl"
        if not fallback.is_file():
            raise FileNotFoundError(trace_path)
        trace_path = fallback
    expected_sha = str(trace_metadata.get("sha256", ""))
    actual_sha = _sha256(trace_path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"trace SHA-256 mismatch: {trace_path}")
    rows: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"trace line {line_number} is not an object")
            rows.append(value)
    if len(rows) != int(trace_metadata["records"]):
        raise ValueError(f"trace record count drifted: {trace_path}")
    return rows, diagnostics


def _group_episodes(
    rows: Iterable[dict[str, Any]],
) -> list[tuple[int, list[dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode"]), []).append(row)
    result: list[tuple[int, list[dict[str, Any]]]] = []
    for episode, episode_rows in sorted(grouped.items()):
        ordered = sorted(episode_rows, key=lambda row: int(row["decision"]))
        if not ordered:
            continue
        terminal = ordered[-1]
        if not bool(terminal.get("terminated") or terminal.get("truncated")):
            raise ValueError(f"episode {episode} has no terminal trace row")
        result.append((episode, ordered))
    return result


def analyze_run(run_dir: Path) -> dict[str, Any]:
    rows, diagnostics = _load_trace(run_dir)
    episodes = _group_episodes(rows)
    episode_records: list[dict[str, Any]] = []
    for episode, episode_rows in episodes:
        terminal = episode_rows[-1]
        record: dict[str, Any] = {
            "episode": episode,
            "seed": int(terminal["seed"]),
            "collision": bool(terminal.get("collision")),
            "success": bool(terminal.get("is_success")),
            "off_route": bool(terminal.get("off_route")),
            "timeout": bool(terminal.get("timeout")),
            "decisions": len(episode_rows),
            "windows": {},
        }
        for window in WINDOWS:
            selected_rows = episode_rows[-window:]
            components = [_components(row) for row in selected_rows]
            record["windows"][str(window)] = {
                "observed_rows": len(components),
                "mean_selected_collision_value": fmean(
                    item["selected_collision_value"] for item in components
                ),
                "maximum_selected_collision_value": max(
                    item["selected_collision_value"] for item in components
                ),
                "mean_collision_value_regret": fmean(
                    item["collision_value_regret"] for item in components
                ),
                "maximum_collision_value_regret": max(
                    item["collision_value_regret"] for item in components
                ),
                "minimum_collision_action_disagreement_rate": fmean(
                    float(item["selected_differs_from_minimum_collision_action"])
                    for item in components
                ),
                "mean_selected_reward_advantage_over_minimum_collision_action": fmean(
                    item["selected_reward_q"] - item["safest_action_reward_q"]
                    for item in components
                ),
            }
        episode_records.append(record)

    separation: dict[str, Any] = {}
    for window in WINDOWS:
        key = str(window)
        positive_mean = [
            float(record["windows"][key]["mean_selected_collision_value"])
            for record in episode_records
            if record["collision"]
        ]
        negative_mean = [
            float(record["windows"][key]["mean_selected_collision_value"])
            for record in episode_records
            if not record["collision"]
        ]
        positive_maximum = [
            float(record["windows"][key]["maximum_selected_collision_value"])
            for record in episode_records
            if record["collision"]
        ]
        negative_maximum = [
            float(record["windows"][key]["maximum_selected_collision_value"])
            for record in episode_records
            if not record["collision"]
        ]
        positive_regret = [
            float(record["windows"][key]["mean_collision_value_regret"])
            for record in episode_records
            if record["collision"]
        ]
        negative_regret = [
            float(record["windows"][key]["mean_collision_value_regret"])
            for record in episode_records
            if not record["collision"]
        ]
        positive_disagreement = [
            float(
                record["windows"][key][
                    "minimum_collision_action_disagreement_rate"
                ]
            )
            for record in episode_records
            if record["collision"]
        ]
        negative_disagreement = [
            float(
                record["windows"][key][
                    "minimum_collision_action_disagreement_rate"
                ]
            )
            for record in episode_records
            if not record["collision"]
        ]
        positive_reward_advantage = [
            float(
                record["windows"][key][
                    "mean_selected_reward_advantage_over_minimum_collision_action"
                ]
            )
            for record in episode_records
            if record["collision"]
        ]
        negative_reward_advantage = [
            float(
                record["windows"][key][
                    "mean_selected_reward_advantage_over_minimum_collision_action"
                ]
            )
            for record in episode_records
            if not record["collision"]
        ]
        separation[key] = {
            "collision_episode_count": len(positive_mean),
            "noncollision_episode_count": len(negative_mean),
            "collision_mean_selected_value": _mean_or_none(positive_mean),
            "noncollision_mean_selected_value": _mean_or_none(negative_mean),
            "mean_value_gap": (
                fmean(positive_mean) - fmean(negative_mean)
                if positive_mean and negative_mean
                else None
            ),
            "mean_value_rank_auc": _rank_auc(positive_mean, negative_mean),
            "maximum_value_rank_auc": _rank_auc(
                positive_maximum, negative_maximum
            ),
            "collision_mean_collision_value_regret": _mean_or_none(
                positive_regret
            ),
            "noncollision_mean_collision_value_regret": _mean_or_none(
                negative_regret
            ),
            "collision_minimum_value_action_disagreement_rate": _mean_or_none(
                positive_disagreement
            ),
            "noncollision_minimum_value_action_disagreement_rate": _mean_or_none(
                negative_disagreement
            ),
            "collision_selected_reward_advantage_over_minimum_value_action": (
                _mean_or_none(positive_reward_advantage)
            ),
            "noncollision_selected_reward_advantage_over_minimum_value_action": (
                _mean_or_none(negative_reward_advantage)
            ),
        }

    label_path = run_dir / "collision_label_diagnostics.json"
    training_path = run_dir / "training_diagnostics.json"
    labels = _read_json(label_path)
    training = _read_json(training_path)
    statistics = training.get("statistics", {})
    if not isinstance(statistics, dict):
        raise TypeError("training diagnostics statistics must be an object")

    def statistic_mean(name: str) -> float | None:
        item = statistics.get(name)
        if not isinstance(item, dict) or item.get("mean") is None:
            return None
        return _finite(item["mean"], name=name)

    selected_decoder = diagnostics.get("selected_deployment_decoder")
    return {
        "run_dir": str(run_dir.resolve()),
        "scenario": training.get("scenario"),
        "algorithm": training.get("algorithm"),
        "selected_deployment_decoder": selected_decoder,
        "rule_free_deployment_decoder": selected_decoder == "target_critic",
        "episodes": len(episode_records),
        "collision_episodes": sum(
            int(record["collision"]) for record in episode_records
        ),
        "success_episodes": sum(
            int(record["success"]) for record in episode_records
        ),
        "label_and_scale": {
            "sampled_positive_label_rate": labels.get(
                "sampled_positive_label_rate"
            ),
            "sampled_mean_discounted_collision_cost": labels.get(
                "sampled_mean_discounted_collision_cost"
            ),
            "training_mean_collision_cost_target": statistic_mean(
                "risk/collision_cost_target"
            ),
            "training_mean_replay_prediction": statistic_mean(
                "risk/collision_cost_replay_prediction"
            ),
            "training_mean_policy_expected_collision_cost": statistic_mean(
                "risk/policy_expected_collision_cost"
            ),
            "training_last_policy_expected_collision_cost": (
                statistics.get("risk/policy_expected_collision_cost", {}).get(
                    "last"
                )
                if isinstance(
                    statistics.get("risk/policy_expected_collision_cost"), dict
                )
                else None
            ),
        },
        "terminal_window_separation": separation,
        "episode_records": episode_records,
        "evidence": {
            "action_diagnostics_sha256": _sha256(
                run_dir / "action_diagnostics.json"
            ),
            "trace_sha256": diagnostics["trace"]["sha256"],
            "collision_label_diagnostics_sha256": _sha256(label_path),
            "training_diagnostics_sha256": _sha256(training_path),
        },
    }


def summarize(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scale_values = [
        float(run["label_and_scale"]["training_mean_policy_expected_collision_cost"])
        for run in runs
        if run["label_and_scale"]["training_mean_policy_expected_collision_cost"]
        is not None
    ]
    scale_ratio = None
    if scale_values and min(scale_values) > 0.0:
        scale_ratio = max(scale_values) / min(scale_values)
    return {
        "schema_version": "topo-scene-v4.9.learned-model-attribution/v1",
        "analysis_kind": "observational_trace_attribution_no_action_override",
        "forbidden_mechanisms_used": {
            "external_kinematic_projection": False,
            "headway_or_ttc_threshold": False,
            "lane_change_veto": False,
            "manual_geometry_label": False,
            "counterfactual_action_rewrite": False,
        },
        "run_count": len(runs),
        "all_deployment_decoders_rule_free": all(
            bool(run["rule_free_deployment_decoder"]) for run in runs
        ),
        "policy_expected_collision_value_scale": {
            "minimum": min(scale_values) if scale_values else None,
            "maximum": max(scale_values) if scale_values else None,
            "max_to_min_ratio": scale_ratio,
        },
        "runs": list(runs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        type=Path,
        help="Accepted v4.9 run directory; repeat for multiple runs.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = summarize([analyze_run(path) for path in args.run_dir])
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
