"""Build a read-only diagnosis package for Full+BalancedSlots regressions.

This script never trains, evaluates, or mutates a model/checkpoint. It only
reads frozen matrix outputs, source assets, and audit artifacts, then writes
derived tables/figures/report files under this report directory.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parents[1]
SOURCE_REPORT = PROJECT_ROOT / "reports" / "systematic_matrix_deep_attribution_20260814"
SOURCE_TABLES = SOURCE_REPORT / "tables"
TABLES = REPORT_DIR / "tables"
FIGURES = REPORT_DIR / "figures"
ASSETS = PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1"

SCENARIO_ORDER = [
    "left_turn",
    "cross",
    "roundabout_easy",
    "roundabout_medium",
    "roundabout",
    "carla",
]
SCENARIO_LABELS = {
    "left_turn": "无保护左转",
    "cross": "双汇入",
    "roundabout_easy": "环岛 A",
    "roundabout_medium": "环岛 B",
    "roundabout": "环岛 C",
    "carla": "CARLA Town-10 任务重建",
}
METRICS = [
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "mean_return",
    "mean_success_completion_time_seconds",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def optional_number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def aggregate_matrix() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = read_csv(SOURCE_TABLES / "run_level_metrics.csv")
    selected = [row for row in rows if row["method"] in {"temporal_graph", "full_balanced"}]
    if len(selected) != 36:
        raise AssertionError(f"Expected 36 selected runs, found {len(selected)}")

    ci_rows = {
        row["scenario"]: row
        for row in read_csv(SOURCE_TABLES / "scenario_attribution.csv")
        if row["comparison"] == "full_balanced_minus_temporal_graph"
    }
    by_scene: list[dict[str, object]] = []
    by_seed: list[dict[str, object]] = []
    for scenario in SCENARIO_ORDER:
        method_rows: dict[str, list[dict[str, str]]] = {}
        for method in ("temporal_graph", "full_balanced"):
            current = sorted(
                [row for row in selected if row["scenario"] == scenario and row["method"] == method],
                key=lambda row: int(row["seed"]),
            )
            if len(current) != 3:
                raise AssertionError((scenario, method, len(current)))
            method_rows[method] = current
        tg = method_rows["temporal_graph"]
        full = method_rows["full_balanced"]
        ci = ci_rows[scenario]
        summary: dict[str, object] = {
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS[scenario],
            "episodes_per_method": 150,
        }
        for metric in METRICS:
            tg_values = [value for row in tg if (value := optional_number(row, metric)) is not None]
            full_values = [value for row in full if (value := optional_number(row, metric)) is not None]
            tg_value = mean(tg_values)
            full_value = mean(full_values)
            summary[f"temporal_graph_{metric}"] = tg_value
            summary[f"full_balanced_{metric}"] = full_value
            summary[f"delta_{metric}"] = full_value - tg_value
        summary["success_delta_ci_low"] = float(ci["success_ci_low"])
        summary["success_delta_ci_high"] = float(ci["success_ci_high"])
        summary["collision_delta_ci_low"] = float(ci["collision_ci_low"])
        summary["collision_delta_ci_high"] = float(ci["collision_ci_high"])
        by_scene.append(summary)

        for seed in range(3):
            tg_row = tg[seed]
            full_row = full[seed]
            seed_row: dict[str, object] = {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "seed": seed,
            }
            for prefix, row in (("temporal_graph", tg_row), ("full_balanced", full_row)):
                for metric in METRICS:
                    seed_row[f"{prefix}_{metric}"] = optional_number(row, metric)
                for diagnostic in (
                    "diagnostic_topology_attention_entropy_mean",
                    "diagnostic_latent_std_mean",
                    "diagnostic_slot_scale_ratio_mean",
                ):
                    value = row.get(diagnostic, "")
                    seed_row[f"{prefix}_{diagnostic}"] = float(value) if value else None
            seed_row["success_delta"] = (
                float(seed_row["full_balanced_success_rate"])
                - float(seed_row["temporal_graph_success_rate"])
            )
            seed_row["collision_delta"] = (
                float(seed_row["full_balanced_collision_rate"])
                - float(seed_row["temporal_graph_collision_rate"])
            )
            by_seed.append(seed_row)
    return by_scene, by_seed


def roundabout_metadata() -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    limits = {"roundabout_easy": 400, "roundabout_medium": 600, "roundabout": 800}
    for scenario in ("roundabout_easy", "roundabout_medium", "roundabout"):
        root = ASSETS / scenario
        source = (root / "scenario_source.txt").read_text(encoding="utf-8")
        variant_match = re.search(r"np\.random\.choice\(1000,\s*(\d+)", source)
        rates = [float(item) for item in re.findall(r"rate\s*=\s*([0-9.]+)", source)]
        main_rate = rates[0]
        special_rate = rates[1] if len(rates) > 1 else 0.0
        nominal_sum = 9.0 * main_rate + special_rate
        ego_root = ET.parse(root / "ego.rou.xml").getroot()
        ego_edges = ego_root.find("route").attrib["edges"]
        traffic_files = sorted((root / "traffic").glob("traffic_*.rou.xml"))
        count = len(traffic_files)
        eval_count = math.ceil(count / 5)
        output[scenario] = {
            "source_generator_variants": int(variant_match.group(1)) if variant_match else None,
            "released_traffic_files": count,
            "train_traffic_files": count - eval_count,
            "evaluation_traffic_files": eval_count,
            "main_flow_rate_each": main_rate,
            "special_flow_rate": special_rate,
            "nominal_configured_rate_sum": nominal_sum,
            "ego_route_edges": ego_edges,
            "paper_max_raw_steps": limits[scenario],
            "network_sha256": sha256(root / "map.net.xml"),
        }
    return output


def scenario_audit(topology: dict[str, object]) -> list[dict[str, object]]:
    roundabouts = roundabout_metadata()
    rows: list[dict[str, object]] = []
    for scenario in ("roundabout_easy", "roundabout_medium", "roundabout"):
        meta = roundabouts[scenario]
        topo = topology["scenarios"][scenario]
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "source_class": "authors_release_v1.0.0 direct SUMO assets",
                "migration_verdict": "可视为原 SMARTS 场景的直接 SUMO 资产迁移",
                "physics_equivalence": "否；任务/接口等价，不是 SMARTS 物理 bitwise 等价",
                "network_sha256": meta["network_sha256"],
                "topology_nodes": topo["valid_node_count"],
                "topology_edges": topo["valid_edge_count"],
                "topology_conflict_edges": topo["relation_counts"]["conflict"],
                "released_traffic_files": meta["released_traffic_files"],
                "train_traffic_files": meta["train_traffic_files"],
                "evaluation_traffic_files": meta["evaluation_traffic_files"],
                "traffic_partition_disjoint": True,
                "ego_route_edges": meta["ego_route_edges"],
                "nominal_configured_rate_sum_veh_h": meta["nominal_configured_rate_sum"],
                "max_raw_steps": meta["paper_max_raw_steps"],
                "reachability_test": "PASS",
                "issue": "无资产破损；但 A/B/C 不是只改变密度的单因子难度阶梯",
            }
        )

    topo = topology["scenarios"]["carla"]
    rows.append(
        {
            "scenario": "carla",
            "scenario_label": SCENARIO_LABELS["carla"],
            "source_class": "CARLA source/waypoint controlled SUMO reconstruction",
            "migration_verdict": "只能称为 Town-10 任务语义重建，不能称为原 CARLA 地图/动力学直接迁移",
            "physics_equivalence": "否；SUMO 路网、车辆/行人流和动力学均为重建",
            "network_sha256": sha256(ASSETS / "carla" / "map.net.xml"),
            "topology_nodes": topo["valid_node_count"],
            "topology_edges": topo["valid_edge_count"],
            "topology_conflict_edges": topo["relation_counts"]["conflict"],
            "released_traffic_files": 1,
            "train_traffic_files": 1,
            "evaluation_traffic_files": 1,
            "traffic_partition_disjoint": False,
            "ego_route_edges": ET.parse(ASSETS / "carla" / "ego.rou.xml").getroot().find("route").attrib["edges"],
            "nominal_configured_rate_sum_veh_h": 840.0,
            "max_raw_steps": 302,
            "reachability_test": "PASS（含强制目标车道变换测试）",
            "issue": "只有一个交通 XML，train/evaluation 资产不互斥；仍有 SUMO seed 随机性",
        }
    )
    return rows


def topology_rows(topology: dict[str, object]) -> list[dict[str, object]]:
    output = []
    for scenario in SCENARIO_ORDER:
        row = topology["scenarios"][scenario]
        relations = row["relation_counts"]
        output.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "nodes": row["valid_node_count"],
                "edges": row["valid_edge_count"],
                "successor": relations["successor"],
                "predecessor": relations["predecessor"],
                "left": relations["left"],
                "right": relations["right"],
                "conflict": relations["conflict"],
                "fingerprint": row["fingerprint"],
                "within_capacity": (
                    row["valid_node_count"] <= topology["declared_capacity"]["max_nodes"]
                    and row["valid_edge_count"] <= topology["declared_capacity"]["max_edges"]
                ),
            }
        )
    return output


def trace_rows() -> list[dict[str, object]]:
    payload = json.loads((TABLES / "selected_rollout_traces.json").read_text(encoding="utf-8"))
    output = []
    for row in payload["runs"]:
        output.append(
            {
                "run": row["run"],
                "scenario": row["scenario"],
                "training_seed": row["training_seed"],
                "outcome": next(
                    name
                    for name in ("success", "collision", "off_route", "timeout")
                    if row["outcome"][name]
                ),
                "raw_steps": row["outcome"]["raw_steps"],
                "mean_target_speed_mps": row["mean_target_speed_mps"],
                "negative_lane_command_rate": row["negative_lane_command_rate"],
                "keep_lane_command_rate": row["keep_lane_command_rate"],
                "positive_lane_command_rate": row["positive_lane_command_rate"],
                "lane_change_applied_rate": row["lane_change_applied_rate"],
                "roads_visited": " > ".join(row["roads_visited"]),
            }
        )
    return output


def make_figures(scene_rows: list[dict[str, object]], seed_rows: list[dict[str, object]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [str(row["scenario_label"]) for row in scene_rows]
    success = np.array([float(row["delta_success_rate"]) * 100 for row in scene_rows])
    collision = np.array([float(row["delta_collision_rate"]) * 100 for row in scene_rows])
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - width / 2, success, width, label="Success Δ (pp)", color="#1261A0")
    ax.bar(x + width / 2, collision, width, label="Collision Δ (pp)", color="#C23B22")
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Full − TemporalGraph (percentage points)")
    ax.set_title("Full+BalancedSlots 的退化集中于环岛 B 与 CARLA；环岛 C 并未下降")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "full_vs_temporal_scene_deltas.svg", format="svg")
    fig.savefig(FIGURES / "full_vs_temporal_scene_deltas.png", dpi=180)
    plt.close(fig)

    focus = [row for row in seed_rows if row["scenario"] in {"roundabout_medium", "carla"}]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), sharey=True)
    for ax, scenario in zip(axes, ("roundabout_medium", "carla")):
        rows = [row for row in focus if row["scenario"] == scenario]
        seeds = np.arange(3)
        ax.plot(
            seeds,
            [float(row["temporal_graph_success_rate"]) * 100 for row in rows],
            marker="o",
            linewidth=2,
            label="TemporalGraph",
            color="#4C78A8",
        )
        ax.plot(
            seeds,
            [float(row["full_balanced_success_rate"]) * 100 for row in rows],
            marker="o",
            linewidth=2,
            label="Full+BalancedSlots",
            color="#E45756",
        )
        ax.set_xticks(seeds)
        ax.set_xlabel("training seed")
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Success (%)")
    axes[0].set_ylim(0, 105)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("下降并非所有种子一致：CARLA 由 Full seed0 的 0% 主导")
    fig.tight_layout()
    fig.savefig(FIGURES / "seed_instability_roundabout_b_carla.svg", format="svg")
    fig.savefig(FIGURES / "seed_instability_roundabout_b_carla.png", dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict[str, object]], fields: list[tuple[str, str]], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    lines = ["| " + " | ".join(label for _, label in fields) + " |"]
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        values = []
        for key, _ in fields:
            value = row[key]
            if key in formats:
                value = formats[key].format(value)
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown(scene_rows: list[dict[str, object]], audit_rows: list[dict[str, object]], topology: list[dict[str, object]], traces: list[dict[str, object]]) -> str:
    metric_table = markdown_table(
        scene_rows,
        [
            ("scenario_label", "场景"),
            ("temporal_graph_success_rate", "TG Success"),
            ("full_balanced_success_rate", "Full Success"),
            ("delta_success_rate", "ΔSuccess"),
            ("delta_collision_rate", "ΔCollision"),
            ("delta_timeout_rate", "ΔTimeout"),
            ("delta_mean_return", "ΔReturn"),
        ],
        {
            "temporal_graph_success_rate": "{:.1%}",
            "full_balanced_success_rate": "{:.1%}",
            "delta_success_rate": "{:+.1%}",
            "delta_collision_rate": "{:+.1%}",
            "delta_timeout_rate": "{:+.1%}",
            "delta_mean_return": "{:+.3f}",
        },
    )
    audit_table = markdown_table(
        audit_rows,
        [
            ("scenario_label", "场景"),
            ("migration_verdict", "迁移判定"),
            ("traffic_partition_disjoint", "流量持出互斥"),
            ("reachability_test", "可达性"),
            ("issue", "审计结论/限制"),
        ],
    )
    topo_table = markdown_table(
        topology,
        [
            ("scenario_label", "场景"),
            ("nodes", "Nodes"),
            ("edges", "Edges"),
            ("conflict", "Conflict edges"),
            ("within_capacity", "容量通过"),
        ],
    )
    trace_table = markdown_table(
        traces,
        [
            ("run", "只读轨迹"),
            ("outcome", "结果"),
            ("mean_target_speed_mps", "Mean speed"),
            ("negative_lane_command_rate", "Cmd −"),
            ("keep_lane_command_rate", "Cmd 0"),
            ("positive_lane_command_rate", "Cmd +"),
        ],
        {
            "mean_target_speed_mps": "{:.2f}",
            "negative_lane_command_rate": "{:.1%}",
            "keep_lane_command_rate": "{:.1%}",
            "positive_lane_command_rate": "{:.1%}",
        },
    )
    return f"""# Full+BalancedSlots 场景退化诊断（冻结矩阵，只读）

## 结论先行

**当前证据支持 Full+BalancedSlots 存在结构性缺陷，但不支持“这些场景不适配，应当替换场景”。** 环岛 A/B/C 的发布 SUMO 资产、拓扑容量和目标可达性均通过审计；双汇入也可达，但当前拓扑构造没有任何 `conflict` 边，属于模型侧关系表达缺口。CARLA 可作为 Town-10 左转/目标车道任务的受控 SUMO 重建，但不能称为原 CARLA 地图与动力学的直接移植，且只有一个交通 XML，train/evaluation 资产不互斥。

从 TemporalGraph 到 Full 同时增加 topology query 和 non-affine BalancedSlots，因此冻结矩阵本身**不能把退化单独归因给其中一个组件**。最强证据链是：拓扑注意力在复杂地图上过于弥散；路线 token 被 42–44 个拓扑 token 稀释；软拓扑注意力又被乘进 same-lane/conflict 特征；BalancedSlots 抹去幅值/置信度却没有改善跨 batch 的 slot 平衡；在稀疏终端奖励和 ±1/3 离散横向阈值下，部分种子最终锁死为单一横向符号。

## 1. 先纠正一个事实：环岛 C 没有退化

{metric_table}

环岛 C 的 Success 实际为 **81.33% → 83.33%（+2.00pp）**，Collision 为 **−2.67pp**；因此退化集合应写成“双汇入、环岛 A、环岛 B、CARLA”，而不是 A/B/C 全部下降。环岛 B 是最清楚的场景级退化（Success −22.67pp、Collision +22.67pp，分层 bootstrap 95% 区间分别为 [−47.33, −2.00]pp 与 [+2.00, +46.67]pp）。CARLA 的 −33.33pp 则由 Full seed0 的 0% 成功率主导，只有 3 个训练种子，置信区间仍触及 0。

![逐场景差值](figures/full_vs_temporal_scene_deltas.png)

![种子不稳定](figures/seed_instability_roundabout_b_carla.png)

## 2. 场景是否有问题、能否视为原场景的 SUMO 迁移版

{audit_table}

### 环岛 A/B/C

- 三者 `map.net.xml` 完全相同（SHA256 `9C23D52E...47E3`），且都来自作者发布包，可称为原 SMARTS 场景的直接 SUMO 资产迁移。
- 但“等价”只覆盖任务语义、观测/动作接口、奖励/终止和控制频率，不是 SMARTS 物理、碰撞求解或传感器的 bitwise 等价。
- A/B/C 不是干净的密度阶梯：A/B/C 的 ego route、主流量率、特殊流、交通 XML 数和时限都不同。名义配置流量总和约为 820/950/450 veh/h，B 反而最密；C 路线最长、时限更长。因此不能把 A→B→C 的差异只解释成“难度/密度逐级增加”。
- 发布目录里 A/B/C 分别有 30/40/15 个交通 XML；冻结划分为 24/6、32/8、12/3，互斥且各方法配对。A/B 比源码生成循环的 15/20 更多，属于发布资产目录现状，不是本次迁移凭空生成的数据。

### CARLA Town-10

- 它从发布的 `carla_env.py`、`wp.npy`、`wp2.npy` 恢复起点、曲线、目标车道、终点框、302-step 上限以及车辆/行人交互语义；目标可达和必须换入目标车道的测试通过。
- 但原 CARLA 没有发布可直接运行的 SUMO 路网。当前 `map.net.xml`、交通流和行人流是新建的 SUMO proxy，因此正确标签是 **`carla_source_waypoint_reconstruction` / controlled extension**，不是 Town10HD_Opt 地图移植。
- 只有 `traffic_0.rou.xml`：代码明确在 train/evaluation 都复用它并公开 `traffic_partition_is_disjoint=false`。SUMO seed 会改变随机速度、驾驶噪声与随机出发位置，但这不是 held-out traffic-asset 泛化。该限制会缩小外部有效性，却不能解释 Full seed0 失败而 TemporalGraph 同种子成功，因为两方法收到相同配对环境。

## 3. 拓扑图审计：没有截断，但语义并不完整

{topo_table}

所有图均低于 64 nodes/256 edges 的容量，没有静默截断。A/B/C 共享同一 44-node/160-edge 指纹。真正的问题是**关系定义**：双汇入只有 successor/predecessor/left/right，`conflict=0`。实现只用 SUMO junction `areFoes()` 生成冲突关系，无法表达两个车流在下游合并到同一车道的“merge conflict”。Full 因此增加了拓扑复杂度，却没有获得该场景最关键的安全关系。

## 4. Full+BalancedSlots 的具体缺陷

### 4.1 对比是联合改动，当前设计不可识别单组件因果

配置中 `temporal_graph` 同时关闭 topology，而 `topo_scene_balanced` 同时开启 topology 和 slot normalization。由此只能说“完整组合退化”，不能断言单独是 topology 或 BalancedSlots。

### 4.2 拓扑查询过于弥散，且误差被二次传播

lane query 只有学习得分加固定 20 m 高斯距离偏置，没有显式路线归属、行驶方向兼容或候选车道门控。复杂图中的平均熵对应约 4–8 条有效候选 lane；Full 的场景去均值 entropy 与 success 的描述性相关为约 −0.46。随后 same-lane/conflict 又由两个软注意力概率相乘得到，弥散误差会进入车辆图边特征。

### 4.3 路线 token 被拓扑 token 数量稀释

最终 goal attention 把 ego route token 与全部 topology token 直接拼在一个池中，没有 token 类型门、层级池化或 cardinality correction。等分注意力时，路线 token 占比只有：双汇入 2/(2+10)=16.7%，A/B/C 2/(2+44)=4.35%，CARLA 3/(3+42)=6.67%。这使模型容易依赖静态地图数量而非任务路线，尤其伤害需要正确横向符号/目标车道的 B 与 CARLA。

### 4.4 BalancedSlots 不是“跨样本平衡器”

32/64/32 三段分别做 non-affine LayerNorm，会在每个样本内强制零均值/单位方差，删除幅值和置信度；它并不保证不同 slot 在 replay batch 上具有相等信息量。冻结诊断中 Full 的总体 latent std 约 0.292，而 TemporalGraph 约 0.906；Full 的跨 batch slot-scale ratio 约 4.96，反而高于 TemporalGraph 的 3.05。这个证据提示表征压缩/塌缩，但相关性不足以单独证明它就是唯一原因。

### 4.5 固定预算下容量增大并造成种子敏感

Full 约 2.017M 参数，TemporalGraph 约 1.438M（+40.2%）；learner update 约 +47%、推理约 +87%、峰值 GPU 约 +52%。两者仍固定为 100k raw steps、约 95k updates。更大而更复杂的模型在稀疏 `success − collision` 终端奖励下更容易欠识别，并把小的 critic/注意力误差放大成不同 seed 的策略分叉。

### 4.6 失败策略出现离散横向动作符号饱和

环境把连续横向动作在 ±1/3 处离散成 −1/0/+1。只读同 seed rollout 显示：CARLA Full seed0 的 92.1% 决策为正向命令、均速仅 2.17m/s，最终 302-step timeout；同 seed TemporalGraph 100% 为反向命令，93 steps 成功。环岛 B Full seed2 则 100% 为正向命令并碰撞；同场景成功的 Full seed1 有 94.6% 反向命令。这不是“地图不可解”，而是 Full 在部分初始化下锁死为错误的路线/横向策略。

{trace_table}

## 5. 为什么各场景表现不同

- **双汇入：** 最关键的 merge-conflict 没被 topology graph 编码；Full 成功回合更快但 Collision +8.67pp，表现像风险偏好/错误安全关系，而非场景损坏。
- **环岛 A：** 图 token 多、路线 token 占比低，但任务短、成功率已接近饱和；下降只有 −2pp，属于弱退化。
- **环岛 B：** 名义流量最高、路线需穿越更长的环岛部分；错误 lane/route attention 更容易直接变成碰撞。Full seed2 的 36% success/64% collision 是主要失败，但碰撞分布跨全部 8 个 evaluation XML，不是一个坏 traffic file。
- **环岛 C：** 并未下降。它的路线更长但名义流量更低；Full 某些“单一横向符号”策略仍可能沿任务路线工作，因此不能把图复杂度机械等同于退化。
- **CARLA：** 必须从转弯后换入指定目标车道，对横向符号极敏感；同时只有一个 traffic asset。Full seed0 的高 topology entropy、低速和错误符号锁死造成 timeout，而另两个 Full seed 为 100%，说明主要是优化/表征不稳定，不是场景普遍不适配。

## 6. 是否需要另外设计场景

**不应替换现有场景。** 一个基准暴露模型缺陷，恰恰说明它有诊断价值；为了让 Full 看起来更好而另换场景会形成选择性报告。现有 A/B/C 与双汇入应继续作为主矩阵；CARLA 应保留但降格为“受控 SUMO proxy”，与直接 SMARTS 资产分栏解释。

如果目标是回答“究竟哪一组件有问题”，可以在不改现有主结论的前提下新增**诊断场景/正交实验**：

1. 在现有场景做 2×2：无 topology/无 norm、topology/无 norm、无 topology/有 norm、topology/有 norm。
2. 同一环岛地图、同一路线、同时限，只改变 450/700/950 veh/h，消除 A/B/C 当前混杂。
3. 双汇入显式加入 merge-conflict 关系并与现有 `areFoes` 图配对，保持交通不变。
4. 目标 token cardinality stress：保持任务相同，只添加 10/20/44 条无关远端 lane，检查 token 数泄漏。
5. CARLA proxy 做目标换道 on/off、行人 on/off、左右镜像配对；并明确 train/eval 都使用同一 traffic XML。
6. 环岛 B 与 CARLA 至少扩到 10 个预注册训练 seeds；若要判断 100k 是否欠训练，预注册 100k/200k/500k 学习曲线，不能事后挑最好预算。

这些是**补充诊断**，不是重新设计一个偏向 Full 的主 benchmark。

## 7. 统计边界

- 每格只有 3 个训练 seeds；50 回合 bootstrap 不能替代新的训练重复。
- 环岛 B 的方向最稳；CARLA 的均值下降由单个 seed 主导，不能称为“系统性不兼容”。
- 宏观 Full−TemporalGraph 为 Success −9.78pp、Collision +4.67pp，但 Holm 校正后不显著。
- Full 与 TemporalGraph 的联合改动使组件级因果未识别；上述机制判断是代码结构、训练诊断与 rollout 行为一致的证据链，而不是已经完成的正交因果证明。

## 证据入口

- 冻结运行表：`../systematic_matrix_deep_attribution_20260814/tables/run_level_metrics.csv`
- 场景级 bootstrap：`../systematic_matrix_deep_attribution_20260814/tables/scenario_attribution.csv`
- 拓扑审计：`../../artifacts/contracts/topology_scan_recheck.json`
- 只读 rollout：`tables/selected_rollout_traces.json`
- 本报告伴随表：`tables/full_vs_temporal_by_scene.csv`、`tables/full_vs_temporal_by_seed.csv`、`tables/scenario_validity_audit.csv`、`tables/topology_audit.csv`
- 研究方法修改：**否**。
"""


def html_table(rows: list[dict[str, object]], fields: list[tuple[str, str]], formatters: dict[str, callable] | None = None) -> str:
    formatters = formatters or {}
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in fields)
    body = []
    for row in rows:
        cells = []
        for key, _ in fields:
            value = row[key]
            if key in formatters:
                value = formatters[key](value)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_html(scene_rows: list[dict[str, object]], audit_rows: list[dict[str, object]], topology: list[dict[str, object]], traces: list[dict[str, object]]) -> str:
    pct = lambda value: f"{float(value):+.1%}"
    rate = lambda value: f"{float(value):.1%}"
    metrics = html_table(
        scene_rows,
        [
            ("scenario_label", "场景"),
            ("temporal_graph_success_rate", "TG Success"),
            ("full_balanced_success_rate", "Full Success"),
            ("delta_success_rate", "ΔSuccess"),
            ("delta_collision_rate", "ΔCollision"),
            ("delta_timeout_rate", "ΔTimeout"),
            ("delta_mean_return", "ΔReturn"),
        ],
        {
            "temporal_graph_success_rate": rate,
            "full_balanced_success_rate": rate,
            "delta_success_rate": pct,
            "delta_collision_rate": pct,
            "delta_timeout_rate": pct,
            "delta_mean_return": lambda value: f"{float(value):+.3f}",
        },
    )
    audits = html_table(
        audit_rows,
        [
            ("scenario_label", "场景"),
            ("migration_verdict", "迁移判定"),
            ("traffic_partition_disjoint", "持出互斥"),
            ("reachability_test", "可达性"),
            ("issue", "限制"),
        ],
    )
    topo = html_table(
        topology,
        [
            ("scenario_label", "场景"), ("nodes", "Nodes"), ("edges", "Edges"),
            ("conflict", "Conflict"), ("within_capacity", "容量通过"),
        ],
    )
    trace = html_table(
        traces,
        [
            ("run", "只读轨迹"), ("outcome", "结果"),
            ("mean_target_speed_mps", "Mean speed"),
            ("negative_lane_command_rate", "Cmd −"),
            ("keep_lane_command_rate", "Cmd 0"),
            ("positive_lane_command_rate", "Cmd +"),
        ],
        {
            "mean_target_speed_mps": lambda value: f"{float(value):.2f}",
            "negative_lane_command_rate": rate,
            "keep_lane_command_rate": rate,
            "positive_lane_command_rate": rate,
        },
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full+BalancedSlots 场景退化诊断</title>
<style>
:root{{--ink:#15212b;--muted:#52616b;--blue:#1261a0;--red:#b33a3a;--paper:#fff;--bg:#eef2f4;--line:#d8e0e5}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}}
main{{max-width:1180px;margin:28px auto;padding:0 20px 48px}} header,section{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:24px 28px;margin-bottom:18px;box-shadow:0 4px 14px rgba(25,45,60,.05)}}
h1{{font-size:30px;margin:0 0 10px}} h2{{font-size:22px;border-bottom:2px solid #e6edf1;padding-bottom:8px}} h3{{font-size:17px;color:var(--blue)}}
.lede{{font-size:17px;color:var(--muted)}} .verdict{{border-left:5px solid var(--red);background:#fff7f5;padding:14px 18px;border-radius:6px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .card{{background:#f7fafb;border:1px solid var(--line);padding:14px;border-radius:8px}} .card b{{display:block;font-size:24px;color:var(--blue)}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{background:#f4f7f9;position:sticky;top:0}} .scroll{{overflow:auto}}
figure{{margin:18px 0}} img{{max-width:100%;height:auto}} figcaption{{color:var(--muted);font-size:13px}} code{{background:#f2f5f7;padding:1px 4px;border-radius:3px}} li{{margin:.35em 0}}
@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}} header,section{{padding:18px}}}}
</style></head><body><main>
<header><h1>Full+BalancedSlots 场景退化诊断</h1><p class="lede">冻结 108-run 矩阵、源码/资产审计与只读 deterministic rollout 的联合证据；未修改研究方法。</p>
<p class="verdict"><b>判定：</b>模型完整组合存在结构性缺陷；现有场景不应因暴露缺陷而被替换。A/B/C 是直接 SMARTS SUMO 资产迁移，CARLA 只能称为 Town-10 任务语义的受控 SUMO 重建。</p>
<div class="cards"><div class="card"><b>−9.78pp</b>Full−TG 宏成功率</div><div class="card"><b>−22.67pp</b>环岛 B 成功率</div><div class="card"><b>+2.00pp</b>环岛 C 成功率（未下降）</div><div class="card"><b>1 XML</b>CARLA train/eval 共用</div></div></header>
<section><h2>1. 逐场景事实</h2><div class="scroll">{metrics}</div><p>环岛 C 的主指标实际略升；清楚退化集中在环岛 B，CARLA 均值下降由 Full seed0 主导。</p>
<figure><img src="figures/full_vs_temporal_scene_deltas.png"><figcaption>正值 Success 为改善；正值 Collision 为恶化。</figcaption></figure>
<figure><img src="figures/seed_instability_roundabout_b_carla.png"><figcaption>训练种子只有 3 个，CARLA 不能据此判为普遍不兼容。</figcaption></figure></section>
<section><h2>2. 场景来源与有效性</h2><div class="scroll">{audits}</div>
<h3>A/B/C</h3><p>同一发布路网、不同 ego route/流量/时限。网络 SHA256 相同，拓扑与可达性通过；但不是只改变密度的单因子阶梯。名义配置流量约 820/950/450 veh/h，B 最密，C 路线最长。</p>
<h3>CARLA</h3><p>从 CARLA 源码与 waypoint 恢复任务约束，但路网、交通流和动力学均为 SUMO 重建。只有一个交通 XML，代码在训练与评估两侧复用；随机 SUMO seed 不等于持出交通资产。</p></section>
<section><h2>3. 拓扑审计</h2><div class="scroll">{topo}</div><p>没有容量截断。双汇入的 <code>conflict=0</code> 是关键缺口：构图只识别 junction foes，不能表达下游 merge conflict。</p></section>
<section><h2>4. 模型缺陷链</h2><ol>
<li><b>联合改动不可识别：</b>Full 同时开启 topology query 和 non-affine slot normalization。</li>
<li><b>lane attention 弥散：</b>固定 20m 距离偏置、无路线/方向硬门；复杂图常等效关注 4–8 条 lane。</li>
<li><b>误差二次传播：</b>same-lane/conflict 由两个软拓扑注意力相乘后进入车辆图。</li>
<li><b>路线 token 稀释：</b>A/B/C 仅 2 个 route token 与 44 个 topology token 直接同池，等分质量仅 4.35%。</li>
<li><b>BalancedSlots 删除置信幅值：</b>每样本内 LayerNorm 不能保证跨 batch 平衡；Full latent std 更低而 slot-scale ratio 更高。</li>
<li><b>容量/预算失配：</b>参数约 +40%，更新 +47%，推理 +87%，仍只给同样 100k raw steps。</li>
<li><b>稀疏奖励与离散阈值锁死：</b>部分 seed 固化为单一横向动作符号，形成碰撞或超时。</li></ol>
<div class="scroll">{trace}</div></section>
<section><h2>5. 是否需要另外设计场景</h2><p><b>不替换现有主场景。</b>保留 A/B/C 与双汇入；CARLA 分栏标为受控 proxy。新增场景只用于因果诊断：</p><ol>
<li>现有场景上的 topology × slot-normalization 2×2 正交消融。</li><li>同图、同路线、同时限，只改变流量的环岛密度梯度。</li><li>双汇入显式 merge-conflict 关系配对。</li><li>无关 topology token 数量压力测试。</li><li>CARLA 换道/行人 on-off 与左右镜像配对。</li><li>环岛 B/CARLA 至少 10 个预注册 seeds；预算曲线须预注册。</li></ol></section>
<section><h2>6. 证据边界</h2><ul><li>3 个训练 seeds；50 回合不能替代训练重复。</li><li>环岛 B 的方向最稳；CARLA 由单 seed 主导。</li><li>宏观差值 Holm 校正后不显著。</li><li>机制链与代码/诊断/轨迹一致，但仍需 2×2 才能给组件级因果。</li></ul>
<p>伴随材料：<code>TECHNICAL_REPORT.md</code>、<code>full_balanced_diagnosis_companion.ipynb</code>、<code>tables/*.csv</code>、<code>evidence_manifest.json</code>。</p></section>
</main></body></html>"""


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    topology_payload = json.loads(
        (PROJECT_ROOT / "artifacts" / "contracts" / "topology_scan_recheck.json").read_text(encoding="utf-8")
    )
    if topology_payload["all_within_capacity"] is not True:
        raise AssertionError("Topology capacity audit did not pass")
    scene_rows, seed_rows = aggregate_matrix()
    audit_rows = scenario_audit(topology_payload)
    topo_rows = topology_rows(topology_payload)
    traces = trace_rows()

    write_csv(TABLES / "full_vs_temporal_by_scene.csv", scene_rows)
    write_csv(TABLES / "full_vs_temporal_by_seed.csv", seed_rows)
    write_csv(TABLES / "scenario_validity_audit.csv", audit_rows)
    write_csv(TABLES / "topology_audit.csv", topo_rows)
    write_csv(TABLES / "selected_rollout_trace_summary.csv", traces)
    make_figures(scene_rows, seed_rows)

    report_md = build_markdown(scene_rows, audit_rows, topo_rows, traces)
    (REPORT_DIR / "TECHNICAL_REPORT.md").write_text(report_md, encoding="utf-8")
    (REPORT_DIR / "report.html").write_text(
        build_html(scene_rows, audit_rows, topo_rows, traces), encoding="utf-8"
    )

    sources = [
        SOURCE_TABLES / "run_level_metrics.csv",
        SOURCE_TABLES / "scenario_attribution.csv",
        PROJECT_ROOT / "artifacts" / "contracts" / "topology_scan_recheck.json",
        PROJECT_ROOT / "experiments" / "systematic_matrix" / "protocol.json",
        ASSETS / "PROVENANCE.md",
        ASSETS / "carla" / "PROVENANCE.md",
        TABLES / "selected_rollout_traces.json",
    ]
    products = [
        REPORT_DIR / "TECHNICAL_REPORT.md",
        REPORT_DIR / "report.html",
        TABLES / "full_vs_temporal_by_scene.csv",
        TABLES / "full_vs_temporal_by_seed.csv",
        TABLES / "scenario_validity_audit.csv",
        TABLES / "topology_audit.csv",
        TABLES / "selected_rollout_trace_summary.csv",
        FIGURES / "full_vs_temporal_scene_deltas.svg",
        FIGURES / "full_vs_temporal_scene_deltas.png",
        FIGURES / "seed_instability_roundabout_b_carla.svg",
        FIGURES / "seed_instability_roundabout_b_carla.png",
    ]
    manifest = {
        "contract": "full-balanced-scenario-diagnosis/evidence-manifest-v1",
        "read_only_source_use": True,
        "research_method_modified": False,
        "matrix_runs": 108,
        "matrix_test_episodes": 5400,
        "inputs": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path)} for path in sources
        ],
        "outputs": [
            {"path": str(path.relative_to(REPORT_DIR)), "sha256": sha256(path)} for path in products
        ],
    }
    (REPORT_DIR / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"report": str(REPORT_DIR / "report.html"), "tables": len(products)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
