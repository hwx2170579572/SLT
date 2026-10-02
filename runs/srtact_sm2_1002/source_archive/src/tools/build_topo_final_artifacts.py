"""Compile evidence-bound attribution, decision, summary, and final report.

This compiler never fills missing experiment cells.  It accepts only complete
real-run summaries and paired diagnostic contracts, derives all comparisons
from those files, and writes the S8/S9 CCFA artifacts atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("scene_rep", "temporal_graph", "topo_scene", "topo_scene_balanced")
METHOD_LABELS = {
    "scene_rep": "MST+SLT",
    "temporal_graph": "TemporalGraph+Graph-SLT",
    "topo_scene": "Full",
    "topo_scene_balanced": "Full+BalancedSlots",
}
CODE_PATHS = (
    "algos/sb3_torch/__init__.py",
    "algos/sb3_torch/callbacks.py",
    "algos/sb3_torch/evaluation.py",
    "algos/sb3_torch/graph_representation.py",
    "algos/sb3_torch/sac.py",
    "algos/sb3_torch/topo_temporal_features.py",
    "configs/sb3_configs.py",
    "envs/sumo/topology_graph.py",
    "tools/audit_topology_graphs.py",
    "tools/audit_topo_final_artifacts.py",
    "tools/check_ccfa_contract.py",
    "tools/diagnose_slot_action_sensitivity.py",
    "tools/diagnose_topology_attention.py",
    "tools/eval_paper_sb3_sumo.py",
    "tools/eval_sb3.py",
    "tools/plot_topo_results.py",
    "tools/profile_topo_compute.py",
    "tools/recover_interrupted_paper_evaluation.py",
    "tools/reevaluate_checkpoint_curve.py",
    "tools/run_latent_probes.py",
    "tools/run_topo_experiments.py",
    "tools/train_sb3.py",
    "tools/build_topo_final_artifacts.py",
    "tests_sb3_sumo/test_topology_graph.py",
    "tests_sb3_sumo/test_topo_temporal.py",
    "tests_sb3_sumo/test_topo_result_artifacts.py",
    "experiments/topo_scene/experiment_contract.yaml",
    "experiments/topo_scene/README.md",
    "README.md",
    "ccfa.yaml",
    "MIGRATION_ORIGINAL_SHA256.json",
    "requirements.txt",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required evidence is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Evidence is not a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _validate_summary(payload: dict[str, Any], phase: str) -> None:
    _require(payload.get("schema_version") == "topo-scene.results/v1", f"{phase}: wrong summary schema")
    _require(payload.get("phase") == phase, f"{phase}: phase mismatch")
    _require(payload.get("computed_from_real_runs") is True, f"{phase}: not computed from real runs")
    _require(payload.get("fabricated_values") is False, f"{phase}: fabricated flag is not false")
    _require(payload.get("missing_jobs") == [], f"{phase}: missing experiment jobs")
    _require(payload.get("complete_jobs") == payload.get("expected_jobs"), f"{phase}: incomplete experiment jobs")


def _one_run(payload: dict[str, Any], method: str) -> dict[str, Any]:
    matches = [row for row in payload["per_run"] if row["method"] == method]
    _require(len(matches) == 1, f"Expected one {method} result, found {len(matches)}")
    row = matches[0]
    numeric = (
        "success_rate",
        "collision_rate",
        "mean_return",
        "parameter_count",
        "train_ms_per_gradient_step",
        "inference_ms_per_action",
        "peak_gpu_memory_mb",
    )
    _require(all(math.isfinite(float(row[key])) for key in numeric), f"{method}: non-finite result")
    return row


def _result_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": METHOD_LABELS[row["method"]],
        "scenario": row["scenario"],
        "training_seed": row["seed"],
        "raw_steps": row["raw_steps"],
        "evaluation_episodes": len(row["episode_records"]),
        "evaluation_seeds": [episode["seed"] for episode in row["episode_records"]],
        "success_rate": row["success_rate"],
        "collision_rate": row["collision_rate"],
        "off_route_rate": row["off_route_rate"],
        "timeout_rate": row["timeout_rate"],
        "mean_return": row["mean_return"],
        "successful_completion_time_seconds": row["successful_completion_time_seconds"],
        "parameter_count": row["parameter_count"],
        "train_ms_per_gradient_step": row["train_ms_per_gradient_step"],
        "inference_ms_per_action": row["inference_ms_per_action"],
        "peak_gpu_memory_mb": row["peak_gpu_memory_mb"],
    }


def _delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in (
        "success_rate",
        "collision_rate",
        "mean_return",
        "successful_completion_time_seconds",
    ):
        candidate_value = candidate[metric]
        reference_value = reference[metric]
        result[metric] = (
            None
            if candidate_value is None or reference_value is None
            else float(candidate_value) - float(reference_value)
        )
    return result


def _probe_view(probes: dict[str, Any], method: str) -> dict[str, Any]:
    row = probes["results"][method]
    view = {
        "full_latent_std": row["full"]["latent_std_mean"],
        "minimum_ttc_r2": row["full"]["minimum_ttc"]["r2_mean"],
        "minimum_distance_r2": row["full"]["minimum_distance"]["r2_mean"],
        "ego_dynamics_r2": row["full"]["ego_dynamics"]["r2_mean"],
    }
    if all(slot in row for slot in ("ego", "social", "route")):
        view["slots"] = {
            slot: {
                "latent_std": row[slot]["latent_std_mean"],
                "minimum_ttc_r2": row[slot]["minimum_ttc"]["r2_mean"],
                "route_progress_balanced_accuracy": row[slot]["route_progress_class"]["balanced_accuracy"],
            }
            for slot in ("ego", "social", "route")
        }
    return view


def _action_view(actions: dict[str, Any], method: str) -> dict[str, Any]:
    row = actions["models"][method]
    critical = row["action_summary_by_risk"]["critical_ttc_le_3s"]
    result = {
        "all_target_speed_mps": row["action_summary_by_risk"]["all"]["target_speed_mps_mean"],
        "critical_target_speed_mps": critical["target_speed_mps_mean"],
        "critical_positive_lane_command_rate": critical["lane_command_positive_rate"],
    }
    if row.get("structured_slots"):
        result["critical_slot_speed_sensitivity_mps"] = {
            slot: row["slot_mean_ablation"][slot]["critical_ttc_le_3s"]
            ["mean_absolute_target_speed_delta_mps"]
            for slot in ("ego", "social", "route")
        }
    return result


def _timeline_key(name: str) -> tuple[str, int] | None:
    aliases = {
        "topo": "full",
        "full": "full",
        "topo_scene": "full",
        "balanced": "balanced",
        "topo_scene_balanced": "balanced",
    }
    for prefix, family in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        marker = f"{prefix}_"
        if not name.startswith(marker):
            continue
        suffix = name[len(marker) :]
        if suffix.endswith("k") and suffix[:-1].isdigit():
            return family, int(suffix[:-1]) * 1000
        if suffix.isdigit():
            return family, int(suffix)
    return None


def _checkpoint_mechanism(
    probes: dict[str, Any], actions: dict[str, Any]
) -> dict[str, Any]:
    index: dict[str, dict[int, str]] = {"full": {}, "balanced": {}}
    for name in probes["models"]:
        parsed = _timeline_key(name)
        _require(parsed is not None, f"Unrecognized checkpoint diagnostic key: {name}")
        family, step = parsed
        index[family][step] = name
    _require(set(index["full"]) == set(index["balanced"]), "Full and balanced checkpoint grids differ")
    _require(set(actions["models"]) == set(probes["models"]), "Checkpoint action/probe model sets differ")

    trajectories: dict[str, list[dict[str, Any]]] = {}
    for family in ("full", "balanced"):
        rows = []
        for step, name in sorted(index[family].items()):
            probe = probes["results"][name]
            action = actions["models"][name]
            rows.append(
                {
                    "raw_steps": step,
                    "ego_latent_std": probe["ego"]["latent_std_mean"],
                    "social_latent_std": probe["social"]["latent_std_mean"],
                    "route_latent_std": probe["route"]["latent_std_mean"],
                    "critical_target_speed_mps": action["action_summary_by_risk"]
                    ["critical_ttc_le_3s"]["target_speed_mps_mean"],
                    "critical_positive_lane_command_rate": action["action_summary_by_risk"]
                    ["critical_ttc_le_3s"]["lane_command_positive_rate"],
                    "critical_route_slot_speed_sensitivity_mps": action["slot_mean_ablation"]
                    ["route"]["critical_ttc_le_3s"]["mean_absolute_target_speed_delta_mps"],
                }
            )
        trajectories[family] = rows

    full_rows = trajectories["full"]
    jumps = [
        {
            "from_raw_steps": left["raw_steps"],
            "to_raw_steps": right["raw_steps"],
            "critical_target_speed_change_mps": right["critical_target_speed_mps"]
            - left["critical_target_speed_mps"],
            "positive_lane_command_rate_change": right["critical_positive_lane_command_rate"]
            - left["critical_positive_lane_command_rate"],
        }
        for left, right in zip(full_rows, full_rows[1:])
    ]
    largest = max(jumps, key=lambda row: abs(row["critical_target_speed_change_mps"]))
    return {
        "same_observations_for_all_checkpoints": True,
        "trajectories": trajectories,
        "full_largest_critical_speed_regime_change": largest,
    }


def _choose_decision(
    results: dict[str, dict[str, Any]],
    *,
    maximum_collision_rate_increase: float = 0.02,
    minimum_success_rate_increase: float = 0.05,
) -> tuple[str, str]:
    baseline = results["scene_rep"]
    temporal = results["temporal_graph"]
    balanced = results["topo_scene_balanced"]
    temporal_safe = (
        temporal["collision_rate"]
        <= baseline["collision_rate"] + maximum_collision_rate_increase
        and temporal["success_rate"] >= baseline["success_rate"]
    )
    balanced_beats_temporal = (
        balanced["collision_rate"]
        <= temporal["collision_rate"] + maximum_collision_rate_increase
        and (
            balanced["success_rate"]
            >= temporal["success_rate"] + minimum_success_rate_increase
            or balanced["mean_return"] > temporal["mean_return"]
        )
    )
    if temporal_safe and not balanced_beats_temporal:
        return (
            "retain_temporal_only",
            "在图方法内部，TemporalGraph 通过 left-turn 安全筛查，而含拓扑迭代没有建立相对它的主指标收益；MST+SLT 仍是经验部署参考。",
        )
    if balanced_beats_temporal:
        return (
            "confirm",
            "BalancedSlots 通过局部安全筛查，并在合同主指标/回报上优于 TemporalGraph；仍需多 seed 确认。",
        )
    if all(math.isfinite(float(results[name]["mean_return"])) for name in METHODS):
        return (
            "redesign_topology",
            "运行数值有限，但原始和平衡化拓扑路径都未建立安全收益；扩大规模前应重设计拓扑—策略耦合。",
        )
    return "stop", "最终结果出现非有限值，违反实验停止规则。"


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _report_markdown(final: dict[str, Any], attribution: dict[str, Any], ledger: dict[str, Any]) -> str:
    rows = final["method_results"]
    table_lines = [
        "| 方法 | 成功率 | 碰撞率 | 回报 | 成功完成时间/s | 参数量 | 训练 ms/update | 推理 ms | 峰值 GPU/MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = rows[method]
        table_lines.append(
            "| {label} | {success} | {collision} | {ret} | {completion} | {params:,} | {train} | {infer} | {memory} |".format(
                label=row["label"],
                success=_format_float(row["success_rate"]),
                collision=_format_float(row["collision_rate"]),
                ret=_format_float(row["mean_return"]),
                completion=_format_float(row["successful_completion_time_seconds"]),
                params=int(row["parameter_count"]),
                train=_format_float(row["train_ms_per_gradient_step"]),
                infer=_format_float(row["inference_ms_per_action"]),
                memory=_format_float(row["peak_gpu_memory_mb"]),
            )
        )
    full_delta = final["comparisons"]["topo_scene_minus_scene_rep"]
    balanced_delta = final["comparisons"]["topo_scene_balanced_minus_scene_rep"]
    mechanism = attribution["deep_attribution"]
    mechanism_timeline = attribution["checkpoint_mechanism"]["trajectories"]
    break_step = mechanism["checkpoint_regime_change"]["to_raw_steps"]
    full_break = next(row for row in mechanism_timeline["full"] if row["raw_steps"] == break_step)
    balanced_break = next(row for row in mechanism_timeline["balanced"] if row["raw_steps"] == break_step)
    representation_tradeoff = mechanism["balanced_representation_tradeoff"]
    evaluation_seeds = final["scope"]["final_evaluation_seeds"]
    training_seeds = final["scope"]["training_seeds"]
    verification = final["engineering_verification"]
    full_overhead = final["efficiency_ratios_vs_scene_rep"]["topo_scene"]
    return f"""# Topology-Temporal Graph-SLT L1 最终报告

## 最终结果

原始 Full 没有通过匹配开发门禁：相对 MST+SLT，成功率变化 {_format_float(full_delta['success_rate'])}，碰撞率变化 {_format_float(full_delta['collision_rate'])}，回报变化 {_format_float(full_delta['mean_return'])}。因此没有启动确认矩阵。证据驱动的 BalancedSlots 迭代相对 MST+SLT 的成功率变化为 {_format_float(balanced_delta['success_rate'])}、碰撞率变化为 {_format_float(balanced_delta['collision_rate'])}；按冻结停止规则，最终方法决策为 **{ledger['final_decision']}**。

{chr(10).join(table_lines)}

上表全部最终率指标使用相同的 `{final['scope']['scenario']}` 评估车流顺序与确定性 episode seeds {evaluation_seeds[0]}..{evaluation_seeds[-1]}。已完成训练 seeds 为 {training_seeds}；这是开发规模证据，不支持总体级或跨场景结论。

## 深层归因

{mechanism['conclusion']}

- 保留时序车辆图但移除拓扑后，成功率/碰撞率从 Full 的 {_format_float(rows['topo_scene']['success_rate'])}/{_format_float(rows['topo_scene']['collision_rate'])} 变为 {_format_float(rows['temporal_graph']['success_rate'])}/{_format_float(rows['temporal_graph']['collision_rate'])}。
- 拓扑注意力本身通过局部性门禁：top-1 距离 p90 为 {_format_float(mechanism['attention_locality']['top1_distance_p90_m'])} m，40 m 内平均注意力质量为 {_format_float(mechanism['attention_locality']['mean_mass_within_40m'])}。
- Full 最大策略断点发生在 {mechanism['checkpoint_regime_change']['from_raw_steps']}→{mechanism['checkpoint_regime_change']['to_raw_steps']} raw steps，临界观测目标速度变化 {_format_float(mechanism['checkpoint_regime_change']['critical_target_speed_change_mps'])} m/s。
- 在同一 {break_step} raw-step 检查点和同一观测集上，Full/Balanced 的临界目标速度为 {_format_float(full_break['critical_target_speed_mps'])}/{_format_float(balanced_break['critical_target_speed_mps'])} m/s，正向换道指令率为 {_format_float(full_break['critical_positive_lane_command_rate'])}/{_format_float(balanced_break['critical_positive_lane_command_rate'])}。
- Balanced 的最终整体 latent std 从 Full 的 {_format_float(representation_tradeoff['full_latent_std'])} 降为 {_format_float(representation_tradeoff['balanced_latent_std'])}，route 槽从 {_format_float(representation_tradeoff['full_route_latent_std'])} 降为 {_format_float(representation_tradeoff['balanced_route_latent_std'])}；这是稳定性—信息收缩权衡，不是新的主指标增益。
- 潜变量与动作探针使用同一观测数据集。它们用于定位策略耦合，不是反事实闭环安全估计。

## Claim 判定

| Claim | 状态 | 证据边界内结论 |
| --- | --- | --- |
| C1 Full 改善闭环驾驶 | {final['claims']['C1']['status']} | {final['claims']['C1']['conclusion']} |
| C2 时序车辆图带来收益 | {final['claims']['C2']['status']} | {final['claims']['C2']['conclusion']} |
| C3 拓扑查询增加价值 | {final['claims']['C3']['status']} | {final['claims']['C3']['conclusion']} |
| C4 计算开销有界 | {final['claims']['C4']['status']} | {final['claims']['C4']['conclusion']} |

相对 MST+SLT，原始 Full 使用 {full_overhead['parameter_count']:.2f}x 参数、{full_overhead['train_ms_per_gradient_step']:.2f}x 实测 update 时间、{full_overhead['inference_ms_per_action']:.2f}x 推理延迟和 {full_overhead['peak_gpu_memory_mb']:.2f}x 峰值 GPU 显存。

## 证据边界

- 已完成：四个 50k raw-step、seed-0 的 `left_turn` 运行；20 episode 最终配对评估；显式种子 checkpoint 复评；同观测探针；拓扑注意力和性能分析。
- 按设计未执行：1M-step、3 场景、3 seed 的确认矩阵，因为开发安全门禁失败。
- 历史 SB3 periodic `evaluations.npz` 仅保留为探索记录；主学习曲线为 `diagnosis/fixed_checkpoint_curves.json`。
- 时间指标在共享 GPU 上测得，只适合作为相对工程背景，不是无竞争硬件基准。

## 工程验证

- 完整回归：{verification['tests_passed']} tests passed，{verification['tests_failed']} failed。
- 确定性拓扑审计：{verification['topology_maps']} 个场景地图全部位于冻结的 64-node/256-edge 容量内，无静默截断。
- 原迁移保护：保留 {verification['protected_baseline_files']} 个受保护文件；source invariant=`{verification['migration_source_invariant']}`。
- 真实 SUMO 集成检查：`{verification['sumo_smoke_status']}`。

## 复现

```powershell
conda run -n llm_pipeline python tools/run_topo_experiments.py validate
conda run -n llm_pipeline python tools/reevaluate_checkpoint_curve.py --help
conda run -n llm_pipeline python tools/plot_topo_results.py
conda run -n llm_pipeline python tools/check_ccfa_contract.py --evaluate --stage S8 --stage S9
```

机器可读证据位于 `results_topo_scene/final/final_summary.json`、`results_topo_scene/diagnosis/attribution.json`、`results_topo_scene/diagnosis/iteration_ledger.json`、`results_topo_scene/final/code_manifest.json` 和 `results_topo_scene/final/figures/dashboard_receipt.json`。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-summary", type=Path, default=PROJECT_ROOT / "results_topo_scene/development/summary.json")
    parser.add_argument("--development-gate", type=Path, default=PROJECT_ROOT / "results_topo_scene/development/gate_decision.json")
    parser.add_argument("--diagnostic-summary", type=Path, default=PROJECT_ROOT / "results_topo_scene/diagnostic/summary.json")
    parser.add_argument("--iteration-summary", type=Path, default=PROJECT_ROOT / "results_topo_scene/iteration/summary.json")
    parser.add_argument("--final-probes", type=Path, default=PROJECT_ROOT / "results_topo_scene/diagnosis/latent_probes_final.json")
    parser.add_argument("--final-actions", type=Path, default=PROJECT_ROOT / "results_topo_scene/diagnosis/slot_action_sensitivity_final.json")
    parser.add_argument("--checkpoint-probes", type=Path, default=PROJECT_ROOT / "results_topo_scene/diagnosis/checkpoint_latent_probes.json")
    parser.add_argument("--checkpoint-actions", type=Path, default=PROJECT_ROOT / "results_topo_scene/diagnosis/checkpoint_action_sensitivity.json")
    parser.add_argument("--curves", type=Path, default=PROJECT_ROOT / "results_topo_scene/diagnosis/fixed_checkpoint_curves.json")
    parser.add_argument("--verification-report", type=Path, default=PROJECT_ROOT / "artifacts/ccfa/verification_report.json")
    parser.add_argument("--topology-scan", type=Path, default=PROJECT_ROOT / "artifacts/ccfa/topology_scan.json")
    parser.add_argument("--baseline-freeze", type=Path, default=PROJECT_ROOT / "artifacts/ccfa/baseline_freeze.json")
    parser.add_argument("--sumo-check", type=Path, default=PROJECT_ROOT / "artifacts/ccfa/sumo_check.json")
    parser.add_argument("--diagnosis-dir", type=Path, default=PROJECT_ROOT / "results_topo_scene/diagnosis")
    parser.add_argument("--final-dir", type=Path, default=PROJECT_ROOT / "results_topo_scene/final")
    args = parser.parse_args()

    source_paths = [
        args.development_summary.resolve(),
        args.development_gate.resolve(),
        args.diagnostic_summary.resolve(),
        args.iteration_summary.resolve(),
        args.final_probes.resolve(),
        args.final_actions.resolve(),
        args.checkpoint_probes.resolve(),
        args.checkpoint_actions.resolve(),
        args.curves.resolve(),
        args.verification_report.resolve(),
        args.topology_scan.resolve(),
        args.baseline_freeze.resolve(),
        args.sumo_check.resolve(),
    ]
    development, gate, diagnostic, iteration, final_probes, final_actions, checkpoint_probes, checkpoint_actions, curves, verification_report, topology_scan, baseline_freeze, sumo_check = [
        _read(path) for path in source_paths
    ]
    _validate_summary(development, "development")
    _validate_summary(diagnostic, "diagnostic")
    _validate_summary(iteration, "iteration")
    _require(gate.get("computed_from_real_runs") is True and gate.get("claim_allowed") is False, "Development gate must be a real failed screen")
    _require(final_probes.get("computed_from_real_rollout") is True and final_probes.get("fabricated_values") is False, "Final probes are not real")
    _require(final_probes.get("same_observations_for_all_models") is True, "Final probes are not observation-paired")
    _require(final_actions.get("computed_from_real_observations") is True and final_actions.get("fabricated_values") is False, "Final action diagnostics are not real")
    _require(final_probes.get("dataset_sha256") == final_actions.get("dataset_sha256"), "Final probe/action datasets differ")
    _require(set(final_probes["models"]) == set(METHODS), "Final probe method set is incomplete")
    _require(set(final_actions["models"]) == set(METHODS), "Final action method set is incomplete")
    _require(checkpoint_probes.get("same_observations_for_all_models") is True, "Checkpoint probes are not paired")
    _require(checkpoint_actions.get("same_observations_for_all_models") is True, "Checkpoint action diagnostics are not paired")
    _require(checkpoint_probes.get("dataset_sha256") == checkpoint_actions.get("dataset_sha256"), "Checkpoint probe/action datasets differ")
    _require(checkpoint_probes.get("dataset_sha256") == final_probes.get("dataset_sha256"), "Final/checkpoint diagnostic datasets differ")
    _require(curves.get("computed_from_real_rollouts") is True and curves.get("fabricated_values") is False, "Checkpoint curves are not real")
    _require(curves.get("fixed_checkpoint_curves_are_primary") is True and isinstance(curves.get("paired_protocol"), dict), "Checkpoint curve protocol is not primary/paired")
    _require(set(run["algorithm"] for run in curves["runs"]) == set(METHODS), "Checkpoint curve method set is incomplete")
    _require(verification_report.get("passed") is True and verification_report.get("tests_failed") == 0, "Full regression report is not passed")
    _require(topology_scan.get("all_within_capacity") is True, "Topology scan exceeds frozen capacity")
    _require(baseline_freeze.get("migration_source_invariant", {}).get("ok") is True, "Migration source invariant is not frozen/passed")
    _require(baseline_freeze.get("pre_change_tests", {}).get("exit_code") == 0, "Pre-change baseline test receipt failed")
    _require(sumo_check.get("status") == "ok", "Live SUMO smoke check did not pass")

    runs = {
        "scene_rep": _one_run(development, "scene_rep"),
        "temporal_graph": _one_run(diagnostic, "temporal_graph"),
        "topo_scene": _one_run(development, "topo_scene"),
        "topo_scene_balanced": _one_run(iteration, "topo_scene_balanced"),
    }
    results = {method: _result_view(row) for method, row in runs.items()}
    thresholds = gate["thresholds"]
    decision, decision_reason = _choose_decision(
        results,
        maximum_collision_rate_increase=float(
            thresholds["maximum_collision_rate_increase"]
        ),
        minimum_success_rate_increase=float(
            thresholds["positive_effect_any"]["minimum_success_rate_increase"]
        ),
    )
    mechanism_timeline = _checkpoint_mechanism(checkpoint_probes, checkpoint_actions)
    attention = runs["topo_scene"]["attention_diagnostics"]
    _require(attention and attention.get("computed_from_real_rollout") is True, "Full attention diagnostic is missing")
    sources = [{"path": _relative(path), "sha256": _sha256(path)} for path in source_paths]
    final_probe_views = {method: _probe_view(final_probes, method) for method in METHODS}
    final_action_views = {method: _action_view(final_actions, method) for method in METHODS}
    balanced_contracted = (
        final_probe_views["topo_scene_balanced"]["full_latent_std"]
        < final_probe_views["topo_scene"]["full_latent_std"]
    )
    balanced_tradeoff_text = (
        "BalancedSlots 的跨观测 latent std 更低，且最终 route 槽显著收缩，说明稳定性修复伴随信息收缩权衡。"
        if balanced_contracted
        else "BalancedSlots 未表现出相对原 Full 的跨观测 latent 收缩。"
    )

    deep_attribution = {
        "strength": "left_turn/seed0 范围内的消融与配对观测诊断三角互证",
        "conclusion": (
            "证据不支持把故障简单归因于图容量溢出或弥散注意力：含拓扑分支改变了策略利用方式，后期结构化槽位的尺度/耦合漂移与突发高速动作区间同步。"
            "移除拓扑可恢复安全；逐槽无仿射归一化检验了该耦合机制，但不能单独证明拓扑增加了任务价值。"
            + balanced_tradeoff_text
        ),
        "topology_removal_contrast": {
            "full": {metric: results["topo_scene"][metric] for metric in ("success_rate", "collision_rate", "mean_return")},
            "temporal_graph": {metric: results["temporal_graph"][metric] for metric in ("success_rate", "collision_rate", "mean_return")},
            "temporal_graph_minus_full": _delta(results["temporal_graph"], results["topo_scene"]),
        },
        "attention_locality": {
            "passed": attention["locality_gate"]["passed"],
            "top1_distance_p90_m": attention["metrics"]["top1_attention_distance_m"]["p90"],
            "mean_mass_within_40m": attention["metrics"]["attention_mass_within_40m"]["mean"],
        },
        "checkpoint_regime_change": mechanism_timeline["full_largest_critical_speed_regime_change"],
        "balanced_representation_tradeoff": {
            "cross_observation_contraction": balanced_contracted,
            "full_latent_std": final_probe_views["topo_scene"]["full_latent_std"],
            "balanced_latent_std": final_probe_views["topo_scene_balanced"]["full_latent_std"],
            "full_route_latent_std": final_probe_views["topo_scene"]["slots"]["route"]["latent_std"],
            "balanced_route_latent_std": final_probe_views["topo_scene_balanced"]["slots"]["route"]["latent_std"],
            "full_minimum_ttc_r2": final_probe_views["topo_scene"]["minimum_ttc_r2"],
            "balanced_minimum_ttc_r2": final_probe_views["topo_scene_balanced"]["minimum_ttc_r2"],
        },
        "interpretation_limit": (
            "消融定位了该运行失败所需的实现差异；潜变量/动作探针属于观测诊断，不能证明唯一的微观因果机制。"
        ),
    }
    reference_seeds = results["scene_rep"]["evaluation_seeds"]
    for method in METHODS[1:]:
        _require(
            results[method]["evaluation_seeds"] == reference_seeds,
            f"{method}: final evaluation seeds are not paired",
        )
        _require(
            results[method]["scenario"] == results["scene_rep"]["scenario"],
            f"{method}: final scenario is not paired",
        )
        _require(
            results[method]["raw_steps"] == results["scene_rep"]["raw_steps"],
            f"{method}: final raw-step budget is not paired",
        )
    attribution = {
        "contract": "topo-scene.attribution/v1",
        "evidence_bound": True,
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "scope": {
            "scenario": results["scene_rep"]["scenario"],
            "training_seeds": sorted({results[method]["training_seed"] for method in METHODS}),
            "raw_steps": results["scene_rep"]["raw_steps"],
            "final_evaluation_seeds": reference_seeds,
        },
        "method_results": results,
        "final_latent_probes": final_probe_views,
        "final_action_diagnostics": final_action_views,
        "checkpoint_mechanism": mechanism_timeline,
        "deep_attribution": deep_attribution,
        "sources": sources,
    }

    iteration_ledger = {
        "contract": "topo-scene.iteration-ledger/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "iterations": [
            {
                "id": "I0_original_full",
                "change": "Topology query + temporal vehicle graph + structured Graph-SLT",
                "result": {metric: results["topo_scene"][metric] for metric in ("success_rate", "collision_rate", "mean_return")},
                "decision": "diagnose",
                "reason": "数值与局部性门禁通过，但开发正效应和碰撞护栏失败。",
            },
            {
                "id": "I1_topology_ablation",
                "change": "Remove topology query while retaining temporal vehicle graph and Graph-SLT",
                "result": {metric: results["temporal_graph"][metric] for metric in ("success_rate", "collision_rate", "mean_return")},
                "decision": "localize_to_topology_path",
                "reason": "在匹配最终评估下恢复安全。",
            },
            {
                "id": "I2_balanced_slots",
                "change": "Add independent non-affine LayerNorm after the 32/64/32 ego/social/route projections only",
                "preserved": ["PyTorch", "Stable-Baselines3", "SUMO", "SAC", "reward", "action_space", "one_step_replay", "topology_inputs"],
                "result": {metric: results["topo_scene_balanced"][metric] for metric in ("success_rate", "collision_rate", "mean_return")},
                "decision": decision,
                "reason": decision_reason,
            },
        ],
        "final_decision": decision,
        "decision_reason": decision_reason,
        "confirmation_started": False,
        "confirmation_blocker": "原冻结开发门禁未通过，未使用门禁覆盖。",
        "sources": sources,
    }

    claims = {
        "C1": {
            "status": "rejected_in_development",
            "conclusion": "匹配 left-turn 筛查中，原始 Full 相对 MST+SLT 同时恶化成功率与碰撞率。",
        },
        "C2": {
            "status": "not_supported_as_an_improvement",
            "conclusion": "TemporalGraph 恢复了安全闭环行为，但没有建立相对天花板基线的主指标或探针优势。",
        },
        "C3": {
            "status": "rejected_for_original_full",
            "conclusion": "拓扑没有证明超越 TemporalGraph 的价值；原始 Full 更差，BalancedSlots 属于事后稳定性检验。",
        },
        "C4": {
            "status": "measured_without_acceptance_threshold",
            "conclusion": "已报告参数、训练、推理与显存开销，但合同未预定义可接受的工程阈值。",
        },
    }
    baseline_efficiency = results["scene_rep"]
    efficiency_ratios = {
        method: {
            metric: float(results[method][metric]) / float(baseline_efficiency[metric])
            for metric in (
                "parameter_count",
                "train_ms_per_gradient_step",
                "inference_ms_per_action",
                "peak_gpu_memory_mb",
            )
        }
        for method in METHODS
    }
    final_summary = {
        "contract": "topo-scene.final-summary/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "claim_boundary_recorded": True,
        "scope": attribution["scope"],
        "method_results": results,
        "comparisons": {
            "topo_scene_minus_scene_rep": _delta(results["topo_scene"], results["scene_rep"]),
            "temporal_graph_minus_scene_rep": _delta(results["temporal_graph"], results["scene_rep"]),
            "topo_scene_balanced_minus_scene_rep": _delta(results["topo_scene_balanced"], results["scene_rep"]),
            "topo_scene_balanced_minus_temporal_graph": _delta(results["topo_scene_balanced"], results["temporal_graph"]),
        },
        "claims": claims,
        "efficiency_ratios_vs_scene_rep": efficiency_ratios,
        "final_decision": decision,
        "decision_reason": decision_reason,
        "confirmation": {
            "completed": False,
            "reason": "开发门禁失败；合同禁止启动 1M-step 确认矩阵。",
        },
        "limitations": [
            "仅一个训练 seed 和一个场景，不支持跨 seed 或跨场景泛化结论。",
            "20 个配对评估 episode 可量化 episode 结果，但不能替代独立训练 seeds。",
            "潜变量探针与槽位均值消融是观测/局部诊断，不是反事实闭环证据。",
            "性能计时采集于共享 GPU。",
        ],
        "primary_learning_curve": "results_topo_scene/diagnosis/fixed_checkpoint_curves.json",
        "historical_periodic_curves": "exploratory_only",
        "engineering_verification": {
            "tests_passed": verification_report["tests_passed"],
            "tests_failed": verification_report["tests_failed"],
            "warnings": verification_report["warnings"],
            "topology_maps": len(topology_scan["scenarios"]),
            "topology_all_within_capacity": topology_scan["all_within_capacity"],
            "protected_baseline_files": len(baseline_freeze["protected_files"]),
            "migration_source_invariant": baseline_freeze["migration_source_invariant"]["ok"],
            "sumo_smoke_status": sumo_check["status"],
        },
        "sources": sources,
    }

    diagnosis_dir = args.diagnosis_dir.resolve()
    final_dir = args.final_dir.resolve()
    _atomic_json(diagnosis_dir / "attribution.json", attribution)
    _atomic_json(diagnosis_dir / "iteration_ledger.json", iteration_ledger)
    _atomic_json(final_dir / "final_summary.json", final_summary)
    _atomic_text(final_dir / "FINAL_REPORT.md", _report_markdown(final_summary, attribution, iteration_ledger))
    code_files = [PROJECT_ROOT / relative for relative in CODE_PATHS]
    missing_code = [relative for relative, path in zip(CODE_PATHS, code_files) if not path.is_file()]
    _require(not missing_code, f"Code-manifest files are missing: {missing_code}")
    _atomic_json(
        final_dir / "code_manifest.json",
        {
            "contract": "topo-scene.code-manifest/v1",
            "complete": True,
            "files": [
                {"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size}
                for relative, path in zip(CODE_PATHS, code_files)
            ],
            "baseline_freeze": "artifacts/ccfa/baseline_freeze.json",
        },
    )
    print(
        json.dumps(
            {
                "contract": final_summary["contract"],
                "final_decision": decision,
                "outputs": [
                    _relative(diagnosis_dir / "attribution.json"),
                    _relative(diagnosis_dir / "iteration_ledger.json"),
                    _relative(final_dir / "final_summary.json"),
                    _relative(final_dir / "FINAL_REPORT.md"),
                    _relative(final_dir / "code_manifest.json"),
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
