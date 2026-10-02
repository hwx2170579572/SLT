"""Seal the post-hoc attribution for the failed v4.7 G3 development cell."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_ROOT = ROOT / "results_topo_v4_7_2_dev"
DEVELOPMENT = RESULT_ROOT / "development"
RUNS = DEVELOPMENT / "runs"
G1 = RUNS / "G1__cand__cross__s6__pe97ef91a"
G3 = RUNS / "G3__cand__cross__s8__pe97ef91a"
ATTRIBUTION = RESULT_ROOT / "attribution" / "g3_failure"
CONTRACT = ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_7.yaml"
FREEZE = RESULT_ROOT / "engineering" / "implementation_freeze.json"
DECISION = DEVELOPMENT / "development_decision.json"
REPLAY_TOOL = ROOT / "tools" / "replay_v4_7_g3_posthoc.py"

EXPECTED_CONTRACT = "e97ef91a85ddeb60bee50fb6fdccdff4afb4e49f5a9d6e52ffee24431890ea7a"
EXPECTED_FREEZE = "8b38eb76de4e0287867e1a588ab1e65895b01d5a7331f24c9ce9debd25f7e49a"
EXPECTED_DECISION = "21808a9a1ec3ec4ff30e892cb962b8dc4cf235a5b8212e30deeaea0cd8b7e0f1"
PAIR_NAMES = (
    "highest_training_success__target_critic",
    "highest_training_success__fusion_0_90",
    "exact_final__target_critic",
    "exact_final__fusion_0_90",
)


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid JSONL: {path}")
    return rows


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _counts(summary: dict[str, Any]) -> dict[str, int]:
    episodes = int(summary["episodes"])
    return {
        "episodes": episodes,
        "success_count": round(float(summary["success_rate"]) * episodes),
        "collision_count": round(float(summary["collision_rate"]) * episodes),
        "off_route_count": round(float(summary["off_route_rate"]) * episodes),
        "timeout_count": round(float(summary["timeout_rate"]) * episodes),
    }


def _signature(rows: list[dict[str, Any]]) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        (int(row["episode"]), int(row["seed"]), str(row["traffic_variant"]))
        for row in rows
    )


def _posthoc(directory: Path) -> dict[str, Any]:
    result_path = directory / "attribution_result.json"
    action_path = directory / "action_diagnostics.json"
    trace_path = directory / "decisions.jsonl"
    result = _load(result_path)
    action = _load(action_path)
    for key, expected in {
        "post_hoc_diagnostic_only": True,
        "counted_for_development_gate": False,
        "counted_for_promotion_gate": False,
        "computed_from_real_closed_loop_rollout": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
        "mutated_checkpoint": False,
        "cross_block_result_is_gate_input": False,
    }.items():
        _require(result.get(key) is expected, f"{directory.name}: {key} drifted")
    _require(result.get("experiment_contract_sha256") == EXPECTED_CONTRACT, "contract drifted")
    _require(result.get("implementation_freeze_sha256") == EXPECTED_FREEZE, "freeze drifted")
    _require(result.get("development_decision_sha256") == EXPECTED_DECISION, "decision drifted")
    _require(action.get("outcomes") == result.get("outcomes"), "action outcomes drifted")
    _require(_sha256(trace_path) == result.get("trace_sha256"), "trace hash drifted")
    return {
        **result,
        "action_diagnostics": action,
        "trace_rows": _load_jsonl(trace_path),
        "source": _source(result_path),
        "action_source": _source(action_path),
        "trace_source": _source(trace_path),
    }


def _actual(run: Path) -> dict[str, Any]:
    detailed_path = run / "paper_evaluation_detailed.json"
    action_path = run / "action_diagnostics.json"
    trace_path = run / "action_diagnostics_decisions.jsonl"
    detailed = _load(detailed_path)
    action = _load(action_path)
    _require(action.get("outcomes") == detailed.get("summary"), "actual outcomes drifted")
    return {
        "outcomes": detailed["summary"],
        "per_episode": detailed["episode_records"],
        "action_diagnostics": action,
        "trace_rows": _load_jsonl(trace_path),
        "selected_checkpoint_kind": detailed["selected_checkpoint_kind"],
        "selected_deployment_decoder": detailed["selected_deployment_decoder"],
        "source": _source(detailed_path),
        "action_source": _source(action_path),
        "trace_source": _source(trace_path),
    }


def _paired(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    left = control["per_episode"]
    right = treatment["per_episode"]
    _require(_signature(left) == _signature(right), "paired signatures differ")
    transitions: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for lrow, rrow in zip(left, right):
        left_event = "success" if lrow["success"] else "collision" if lrow["collision"] else "timeout" if lrow["timeout"] else "other"
        right_event = "success" if rrow["success"] else "collision" if rrow["collision"] else "timeout" if rrow["timeout"] else "other"
        transitions[f"{left_event}_to_{right_event}"] += 1
        rows.append(
            {
                "episode": int(lrow["episode"]),
                "seed": int(lrow["seed"]),
                "traffic_variant": lrow["traffic_variant"],
                "control_event": left_event,
                "treatment_event": right_event,
            }
        )
    return {"transition_counts": dict(sorted(transitions.items())), "rows": rows}


def _training(run: Path) -> dict[str, Any]:
    diagnostics_path = run / "training_diagnostics.json"
    monitor_path = run / "train_monitor.csv"
    diagnostics = _load(diagnostics_path)
    stats = diagnostics["statistics"]
    lines = monitor_path.read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(lines[1:]))
    return {
        "episodes": len(rows),
        "success_count": sum(row["is_success"] == "True" for row in rows),
        "collision_count": sum(row["collision"] == "True" for row in rows),
        "timeout_count": sum(row["max_time"] == "True" for row in rows),
        "last_20_success_count": sum(row["is_success"] == "True" for row in rows[-20:]),
        "actor_lane_entropy_mean": float(stats["hybrid/lane_entropy"]["mean"]),
        "actor_lane_keep_probability_mean": float(stats["hybrid/lane_probability_keep"]["mean"]),
        "critic_loss_mean": float(stats["train/critic_loss"]["mean"]),
        "critic_loss_last": float(stats["train/critic_loss"]["last"]),
        "return_estimator": diagnostics["return_estimator"],
        "diagnostics_source": _source(diagnostics_path),
        "monitor_source": _source(monitor_path),
    }


def _action_summary(value: dict[str, Any]) -> dict[str, Any]:
    action = value["action_diagnostics"]
    return {
        "outcomes": value["outcomes"],
        "lane_keep_rate": float(action["lane_command_rates"]["keep"]),
        "lane_change_applied_rate": float(action["lane_change_applied_rate"]),
        "route_action_window_match_rate": float(action["route_action_window_match_rate"]),
        "selected_decoder_exact_rule_match_rate": float(action["selected_decoder_exact_rule_match_rate"]),
        "selected_action_mask_feasible_rate": float(action["selected_action_mask_feasible_rate"]),
    }


def _combine(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    a = _counts(first)
    b = _counts(second)
    episodes = a["episodes"] + b["episodes"]
    return {
        "episodes": episodes,
        "success_count": a["success_count"] + b["success_count"],
        "collision_count": a["collision_count"] + b["collision_count"],
        "off_route_count": a["off_route_count"] + b["off_route_count"],
        "timeout_count": a["timeout_count"] + b["timeout_count"],
        "mean_return": (
            float(first["mean_return"]) * a["episodes"]
            + float(second["mean_return"]) * b["episodes"]
        ) / episodes,
    }


def _markdown(payload: dict[str, Any]) -> str:
    cross = payload["crossed_weight_block_intervention"]
    selector = payload["selector_failure"]
    training = payload["training_seed_instability"]
    second = payload["tie_only_second_train_calibration"]
    return f"""# v4.7 G3 失败深层归因

状态：post-hoc attribution only；不计入 development/promotion/formal gate；formal test 未访问。

## 结论

G3 的工件、16-step 回报、decoder 和绑定均通过，科学 gate 因 `5/12` 成功、
`6/12` 碰撞而失败。直接必要原因是首个 train-only 校准块上四组合 outcome
完全平局，旧规则偏好 `exact_final + target_critic`；同一 checkpoint 的
`fusion_0_90` 在完全相同 validation episodes 上达到 `8/12` 成功、`4/12`
碰撞，三例失败全部改善为成功且无成功退化。

## 排除 validation block 主因

- G1 权重：block 67000 为 {cross['g1_block67000_success_count']}/12，block 68000 为 {cross['g1_block68000_success_count']}/12；
- G3 权重：block 67000 为 {cross['g3_block67000_success_count']}/12，block 68000 为 {cross['g3_block68000_success_count']}/12。

两 block 平均成功数相同，而 G1 与 G3 权重平均相差
`{cross['weight_average_success_count_gap']:.1f}/12`，因此主效应来自权重/decoder
交互而非 block 难度。

## 上游机制

G3 训练期 lane keep 概率均值为
`{training['g3']['actor_lane_keep_probability_mean']:.3f}`，G1 为
`{training['g1']['actor_lane_keep_probability_mean']:.3f}`；G3 最后 20 个训练
episode 仅 `{training['g3']['last_20_success_count']}/20` 成功。闭环 target
decoder 的 route-window match 从 G1 的
`{selector['g1_target_action']['route_action_window_match_rate']:.3f}` 降到 G3 的
`{selector['g3_target_action']['route_action_window_match_rate']:.3f}`。这解释了
为何 target tie preference 在 seed8 上脆弱，但不否定 16-step 在 seed6 的
配对增益。

## 第二 train 校准块

第二块使用不同 SUMO seeds、同一冻结 train-variant 支撑并对所有组合严格
配对。合并 24 episodes 后：

| 组合 | success | collision | timeout |
| --- | ---: | ---: | ---: |
""" + "\n".join(
        f"| {name} | {row['success_count']}/24 | {row['collision_count']}/24 | {row['timeout_count']}/24 |"
        for name, row in second["combined_24"].items()
    ) + f"""

因此数据驱动排序会选择
`{second['selected_by_combined_24']}`，不需要查看 validation。

## v4.8 单一变化

保留 v4.7 的网络、16-step `gamma^h`、训练、checkpoint 和 decoder 定义。
仅当首轮 12 episodes 的结果排序项完全并列时，对并列首位组合运行第二个
预注册 train seed block；按合并 24 episodes 重排，仍平局才使用旧的
exact-final / target 偏好。新版本必须在未访问的新 validation block 和新鲜
训练 seed 上确认，不能把本归因回放计入门禁。
"""


def main() -> int:
    _require(_sha256(CONTRACT) == EXPECTED_CONTRACT, "contract hash drifted")
    _require(_sha256(FREEZE) == EXPECTED_FREEZE, "freeze hash drifted")
    _require(_sha256(DECISION) == EXPECTED_DECISION, "decision hash drifted")

    decision = _load(DECISION)
    g3_gate = decision["jobs"]["G3"]["gate"]
    _require(decision.get("decision") == "fail", "development did not stop")
    _require(g3_gate.get("outcome_passed") is False, "G3 outcome did not fail")
    _require(g3_gate.get("mechanism_passed") is True, "G3 mechanism failed")
    _require(g3_gate.get("selector_passed") is True, "G3 selector integrity failed")
    _require(decision.get("formal_test_unlocked") is False, "formal test was unlocked")
    _require(not (RUNS / "G4__cand__carla__s5__pe97ef91a").exists(), "G4 unexpectedly ran")
    _require(not (RUNS / "G5__cand__ram__s3__pe97ef91a").exists(), "G5 unexpectedly ran")

    g1_actual = _actual(G1)
    g3_target = _actual(G3)
    g1_68000 = _posthoc(ATTRIBUTION / "cross_block" / "G1_weights_block68000")
    g3_67000 = _posthoc(ATTRIBUTION / "cross_block" / "G3_weights_block67000")
    best_target = _posthoc(ATTRIBUTION / "pair_matrix" / "best_target_block68000")
    best_fusion = _posthoc(ATTRIBUTION / "pair_matrix" / "best_fusion_block68000")
    final_fusion = _posthoc(ATTRIBUTION / "pair_matrix" / "final_fusion_block68000")

    _require(_signature(g1_actual["per_episode"])[0][1] == 67000, "G1 block drifted")
    _require(_signature(g1_68000["per_episode"])[0][1] == 68000, "G1 cross block drifted")
    _require(_signature(g3_target["per_episode"])[0][1] == 68000, "G3 block drifted")
    _require(_signature(g3_67000["per_episode"])[0][1] == 67000, "G3 cross block drifted")
    _require(_signature(g3_target["per_episode"]) == _signature(final_fusion["per_episode"]), "decoder matrix is not paired")
    _require(_signature(best_target["per_episode"]) == _signature(final_fusion["per_episode"]), "checkpoint matrix is not paired")
    _require(_signature(best_fusion["per_episode"]) == _signature(final_fusion["per_episode"]), "pair matrix is not paired")

    receipt = _load(G3 / "selector" / "receipt.json")
    original: dict[str, dict[str, Any]] = {}
    for candidate in receipt["candidates"]:
        name = f"{candidate['checkpoint_kind']}__{candidate['deployment_decoder']}"
        original[name] = candidate["summary"]
    _require(tuple(original) == PAIR_NAMES, "candidate ordering drifted")
    data_keys = {
        (
            _counts(summary)["success_count"],
            -_counts(summary)["collision_count"],
            -_counts(summary)["off_route_count"],
            -_counts(summary)["timeout_count"],
            float(summary["mean_return"]),
        )
        for summary in original.values()
    }
    _require(len(data_keys) == 1, "G3 initial candidates were not fully tied")
    _require(receipt["selected_checkpoint_kind"] == "exact_final", "selected checkpoint drifted")
    _require(receipt["selected_deployment_decoder"] == "target_critic", "selected decoder drifted")

    second_dirs = {
        "highest_training_success__target_critic": "best_target",
        "highest_training_success__fusion_0_90": "best_fusion",
        "exact_final__target_critic": "final_target",
        "exact_final__fusion_0_90": "final_fusion",
    }
    second = {
        name: _posthoc(ATTRIBUTION / "tie_break_train_block77100" / directory)
        for name, directory in second_dirs.items()
    }
    signatures = {_signature(value["per_episode"]) for value in second.values()}
    _require(len(signatures) == 1, "second calibration candidates are not paired")
    initial_signature = tuple(
        (int(row["episode"]), int(row["seed"]), str(row["traffic_variant"]))
        for row in receipt["candidates"][0]["episode_pair_signature"]
    )
    second_signature = next(iter(signatures))
    validation_signature = _signature(g3_target["per_episode"])
    _require({row[1] for row in initial_signature}.isdisjoint({row[1] for row in second_signature}), "calibration seeds overlap")
    _require({row[1] for row in second_signature}.isdisjoint({row[1] for row in validation_signature}), "second calibration and validation seeds overlap")
    _require({row[2] for row in initial_signature} == {row[2] for row in second_signature}, "paired train variant support drifted")

    combined = {
        name: _combine(original[name], second[name]["outcomes"])
        for name in PAIR_NAMES
    }
    ranked = sorted(
        PAIR_NAMES,
        key=lambda name: (
            combined[name]["success_count"],
            -combined[name]["collision_count"],
            -combined[name]["off_route_count"],
            -combined[name]["timeout_count"],
            combined[name]["mean_return"],
            int(name.startswith("exact_final")),
            int(name.endswith("target_critic")),
        ),
        reverse=True,
    )
    _require(ranked[0] == "exact_final__fusion_0_90", "second calibration did not select fusion")

    g1_counts = [_counts(g1_actual["outcomes"])["success_count"], _counts(g1_68000["outcomes"])["success_count"]]
    g3_counts = [_counts(g3_67000["outcomes"])["success_count"], _counts(g3_target["outcomes"])["success_count"]]
    selector_failure = {
        "initial_calibration_complete_data_tie": True,
        "old_selected_pair": "exact_final__target_critic",
        "old_selected_validation": _action_summary(g3_target),
        "validation_eligible_alternative": "exact_final__fusion_0_90",
        "alternative_validation": _action_summary(final_fusion),
        "paired_target_to_fusion": _paired(g3_target, final_fusion),
        "success_count_gain": _counts(final_fusion["outcomes"])["success_count"] - _counts(g3_target["outcomes"])["success_count"],
        "collision_count_reduction": _counts(g3_target["outcomes"])["collision_count"] - _counts(final_fusion["outcomes"])["collision_count"],
        "timeout_count_reduction": _counts(g3_target["outcomes"])["timeout_count"] - _counts(final_fusion["outcomes"])["timeout_count"],
        "best_target_validation": _action_summary(best_target),
        "best_fusion_validation": _action_summary(best_fusion),
        "g1_target_action": _action_summary(g1_actual),
        "g3_target_action": _action_summary(g3_target),
        "interpretation": "an eligible passing pair existed; the inherited deterministic preference resolved a finite-sample calibration tie toward the failing decoder",
    }
    payload = {
        "schema_version": "topo-scene-v4.7.g3-failure-attribution/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "post_hoc_diagnostic_only": True,
        "counted_for_development_gate": False,
        "counted_for_promotion_gate": False,
        "formal_test_accessed": False,
        "failed_job": "G3__cand__cross__s8__pe97ef91a",
        "failure_gate": {
            "artifact_accepted": True,
            "outcome_passed": False,
            "mechanism_passed": True,
            "selector_integrity_passed": True,
            "success_count": 5,
            "collision_count": 6,
            "timeout_count": 1,
            "G4_G5_executed": False,
            "formal_test_unlocked": False,
        },
        "primary_attribution": {
            "mechanism": "finite_sample_train_calibration_tie_resolved_by_inherited_target_decoder_preference_despite_passing_fusion_pair",
            "confidence": "strong_paired_closed_loop_posthoc_intervention",
            "upstream_risk": "training_seed_dependent_actor_critic_lane_preference_instability",
            "rejected_as_primary": [
                "artifact_or_binding_failure",
                "return_estimator_implementation_failure",
                "validation_block_difficulty",
                "absence_of_any_passing_checkpoint_decoder_pair",
                "best_checkpoint_would_have_rescued_target_decoder",
            ],
        },
        "selector_failure": selector_failure,
        "crossed_weight_block_intervention": {
            "g1_block67000_success_count": g1_counts[0],
            "g1_block68000_success_count": g1_counts[1],
            "g3_block67000_success_count": g3_counts[0],
            "g3_block68000_success_count": g3_counts[1],
            "g1_weight_average_success_count": sum(g1_counts) / 2.0,
            "g3_weight_average_success_count": sum(g3_counts) / 2.0,
            "weight_average_success_count_gap": (sum(g1_counts) - sum(g3_counts)) / 2.0,
            "block67000_average_success_count": (g1_counts[0] + g3_counts[0]) / 2.0,
            "block68000_average_success_count": (g1_counts[1] + g3_counts[1]) / 2.0,
        },
        "training_seed_instability": {"g1": _training(G1), "g3": _training(G3)},
        "initial_calibration_matrix": original,
        "validation_pair_matrix": {
            "highest_training_success__target_critic": _action_summary(best_target),
            "highest_training_success__fusion_0_90": _action_summary(best_fusion),
            "exact_final__target_critic": _action_summary(g3_target),
            "exact_final__fusion_0_90": _action_summary(final_fusion),
        },
        "tie_only_second_train_calibration": {
            "seed_start": 77100,
            "episodes_per_tied_pair": 12,
            "paired_episode_signatures": True,
            "simulator_seed_disjoint_from_initial_calibration": True,
            "same_frozen_train_variant_support_replicated": True,
            "validation_seed_disjoint": True,
            "second_block": {name: value["outcomes"] for name, value in second.items()},
            "combined_24": combined,
            "selected_by_combined_24": ranked[0],
            "selection_uses_validation": False,
        },
        "selected_next_hypothesis": {
            "version": "v4.8_tie_only_replicated_train_calibration",
            "single_change": "when top candidates tie on all empirical outcome fields after the first 12 paired train episodes, evaluate only the tied set on a second 12-episode train seed block and rank by combined 24-episode outcomes",
            "fallback_if_still_tied": "retain exact_final_then_target_critic deterministic preferences",
            "training_changed": False,
            "return_estimator_changed": False,
            "checkpoint_set_changed": False,
            "decoder_definitions_changed": False,
            "fresh_unseen_validation_required": True,
            "fresh_training_seed_confirmation_required": True,
            "formal_test_eligible": False,
        },
        "risk_register": [
            {"risk": "v4.8 was motivated after seeing G3 validation", "mitigation": "use a new unseen validation seed block plus a fresh training seed; old posthoc results never enter a gate"},
            {"risk": "second calibration repeats the same 12 train traffic variants", "mitigation": "declare this as simulator-seed replication, keep all pairs strictly paired, and require fresh cross-scenario evidence"},
            {"risk": "adaptive sample size favors tied candidates", "mitigation": "only candidates exactly tied for first place are eligible; lower-ranked candidates cannot be promoted by unseen extra samples"},
            {"risk": "fusion may create unsafe off-window overrides", "mitigation": "outcome ranking remains primary, decoder integrity remains exact, and CARLA collision hard gate is non-compensable"},
        ],
        "source_integrity": {
            "experiment_contract": _source(CONTRACT),
            "implementation_freeze": _source(FREEZE),
            "development_decision": _source(DECISION),
            "posthoc_replay_tool": _source(REPLAY_TOOL),
            "g3_selector_receipt": _source(G3 / "selector" / "receipt.json"),
        },
    }
    summary = ATTRIBUTION / "attribution_summary.json"
    markdown = ATTRIBUTION / "deep_attribution.md"
    _write(summary, payload)
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary),
                "summary_sha256": _sha256(summary),
                "markdown": str(markdown),
                "markdown_sha256": _sha256(markdown),
                "primary_attribution": payload["primary_attribution"],
                "selected_next_hypothesis": payload["selected_next_hypothesis"]["version"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
