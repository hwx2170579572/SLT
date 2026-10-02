"""Create the reproducible companion notebook for the read-only diagnosis."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


REPORT_DIR = Path(__file__).resolve().parent
NOTEBOOK = REPORT_DIR / "full_balanced_diagnosis_companion.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(
        """
        # Full+BalancedSlots 场景退化诊断

        本 notebook 只读取冻结 108-run 矩阵、场景源资产、拓扑审计和只读 rollout trace。
        它不会训练、评估、选择或覆盖任何模型/checkpoint，也不修改研究方法。

        **核心判定：** A/B/C 的发布资产和可达性通过；CARLA 是受控 SUMO 任务重建而不是
        原 Town10HD_Opt 地图/动力学迁移。Full 完整组合存在场景依赖与种子不稳定，现有场景不应被替换。
        """
    ),
    code(
        """
        from pathlib import Path
        import csv
        import json
        import subprocess
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        from IPython.display import display, HTML, Markdown

        REPORT_DIR = Path.cwd().resolve()
        assert (REPORT_DIR / "build_diagnosis_artifacts.py").is_file(), REPORT_DIR

        def read_csv(path):
            with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))

        def show_table(rows, fields):
            head = "".join(f"<th>{label}</th>" for key, label in fields)
            body = "".join(
                "<tr>" + "".join(f"<td>{row[key]}</td>" for key, label in fields) + "</tr>"
                for row in rows
            )
            display(HTML(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"))
        """
    ),
    md(
        """
        ## 1. 重建派生证据

        下列命令只写本报告目录，并验证 36 个 TemporalGraph/Full run、拓扑容量以及证据入口。
        """
    ),
    code(
        """
        completed = subprocess.run(
            [sys.executable, str(REPORT_DIR / "build_diagnosis_artifacts.py")],
            cwd=REPORT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        print(completed.stdout)
        manifest = json.loads((REPORT_DIR / "evidence_manifest.json").read_text(encoding="utf-8"))
        print("inputs:", len(manifest["inputs"]), "outputs:", len(manifest["outputs"]))
        print("research_method_modified:", manifest["research_method_modified"])
        """
    ),
    md("## 2. Full − TemporalGraph 的逐场景差值"),
    code(
        """
        scene = read_csv(REPORT_DIR / "tables" / "full_vs_temporal_by_scene.csv")
        for row in scene:
            for key in ("temporal_graph_success_rate", "full_balanced_success_rate", "delta_success_rate", "delta_collision_rate"):
                row[key] = f"{float(row[key]):+.1%}" if key.startswith("delta_") else f"{float(row[key]):.1%}"
        show_table(scene, [
            ("scenario_label", "场景"),
            ("temporal_graph_success_rate", "TG Success"),
            ("full_balanced_success_rate", "Full Success"),
            ("delta_success_rate", "ΔSuccess"),
            ("delta_collision_rate", "ΔCollision"),
        ])
        display(Markdown("**注意：环岛 C 为 +2.0pp，并未下降。**"))
        """
    ),
    md("## 3. 场景来源、持出划分与拓扑语义"),
    code(
        """
        audits = read_csv(REPORT_DIR / "tables" / "scenario_validity_audit.csv")
        show_table(audits, [
            ("scenario_label", "场景"),
            ("migration_verdict", "迁移判定"),
            ("traffic_partition_disjoint", "持出互斥"),
            ("reachability_test", "可达性"),
            ("issue", "限制"),
        ])

        topo = read_csv(REPORT_DIR / "tables" / "topology_audit.csv")
        show_table(topo, [
            ("scenario_label", "场景"), ("nodes", "Nodes"), ("edges", "Edges"),
            ("conflict", "Conflict edges"), ("within_capacity", "容量通过"),
        ])
        display(Markdown("双汇入的 `conflict=0` 说明当前 `areFoes` 构图未编码 merge conflict。"))
        """
    ),
    md("## 4. 训练种子不稳定与横向动作饱和"),
    code(
        """
        seeds = read_csv(REPORT_DIR / "tables" / "full_vs_temporal_by_seed.csv")
        focus = [row for row in seeds if row["scenario"] in {"roundabout_medium", "carla"}]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
        for ax, scenario_name in zip(axes, ("roundabout_medium", "carla")):
            rows = [row for row in focus if row["scenario"] == scenario_name]
            x = np.arange(3)
            ax.plot(x, [100*float(r["temporal_graph_success_rate"]) for r in rows], marker="o", label="TemporalGraph")
            ax.plot(x, [100*float(r["full_balanced_success_rate"]) for r in rows], marker="o", label="Full")
            ax.set_xticks(x); ax.set_xlabel("seed"); ax.set_title(rows[0]["scenario_label"]); ax.grid(axis="y", alpha=.25)
        axes[0].set_ylabel("Success (%)"); axes[0].set_ylim(0,105); axes[1].legend(frameon=False)
        plt.tight_layout(); plt.show()

        traces = read_csv(REPORT_DIR / "tables" / "selected_rollout_trace_summary.csv")
        show_table(traces, [
            ("run", "只读轨迹"), ("outcome", "结果"), ("mean_target_speed_mps", "均速"),
            ("negative_lane_command_rate", "Cmd −"), ("keep_lane_command_rate", "Cmd 0"),
            ("positive_lane_command_rate", "Cmd +"),
        ])
        """
    ),
    md(
        """
        ## 5. 结论边界

        - 不替换现有场景；CARLA 分栏标为受控 proxy。
        - 环岛 B 的退化证据最稳，CARLA 由一个 Full seed 主导。
        - TemporalGraph→Full 同时改变 topology query 与 BalancedSlots，组件级因果尚未识别。
        - 后续如需定位组件，使用 topology × slot normalization 的预注册 2×2；新增场景只作诊断，不改主矩阵。
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "analysis_contract": {
            "matrix_runs": 108,
            "test_episodes": 5400,
            "read_only": True,
            "research_method_modified": False,
        },
    },
)
nbf.write(notebook, NOTEBOOK)
print(NOTEBOOK)
