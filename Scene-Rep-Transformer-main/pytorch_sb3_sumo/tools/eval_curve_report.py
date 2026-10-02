"""Post-hoc checkpoint eval-curve report: common-RL success-rate vs raw steps.

Reads the 10k/20k/30k/40k intermediate eval points written by
`tools/eval_checkpoint_curve.py` and splices in the already-known 50k final
point, then draws success_rate / collision_rate vs raw steps for 6 candidates
(5 v4_8 + MST+SLT baseline) x 2 scenes (cross + carla).

Outputs (under `--out`, default `r48s4/eval_curve/report`):
  * success_curve.png    -- 2 panels (cross / carla), y = eval success_rate
  * collision_curve.png  -- 2 panels (cross / carla), y = eval collision_rate
  * REPORT.md            -- full table + observations (Chinese)

No project imports: plain json + matplotlib so the report stays decoupled from
the training/eval code paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
SCENES = ("cross", "carla")
RAW_STEPS = (10000, 20000, 30000, 40000, 50000)

# Display label, color, line style.  lr_half = fixed reference, hold35k = champion,
# hold45k = trap (nice training curve but deployment collapse), mst_slt = baseline.
CANDIDATES = (
    ("lr_half", "lr_half (基准)", "#7f7f7f", ":"),
    ("tau0025_floor2e5_hold35k", "hold35k (冠军)", "#d62728", "-"),
    ("tau0025_floor2e5_hold40k", "hold40k", "#1f77b4", "-"),
    ("tau0025_floor2e5_hold30k", "hold30k", "#2ca02c", "-"),
    ("tau0025_floor2e5_hold45k", "hold45k (陷阱)", "#9467bd", "-"),
    ("mst_slt", "MST+SLT (基线)", "#ff7f0e", "--"),
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def v48_intermediate(scene, cand, raw_step) -> Path:
    return ROOT / "r48s4" / "eval_curve" / f"{scene}__{cand}" / f"raw_{raw_step}" / "result.json"


def v48_final(scene, cand) -> Path:
    return ROOT / "r48s4" / "eval" / f"{scene}__{cand}" / "result.json"


def mst_intermediate(scene, raw_step) -> Path:
    return ROOT / "results_hd_ss100_v2" / "comparison" / "eval_curve" / f"mst_slt__{scene}" / f"raw_{raw_step}" / "result.json"


def mst_final(scene) -> Path:
    return (
        ROOT
        / "results_hd_ss100_v2"
        / "comparison"
        / "runs"
        / f"hd_ss100_v2__comparison__mst_slt__{scene}__seed0"
        / "final_evaluation.json"
    )


def point(scene, cand, raw_step):
    """Return (success_rate, collision_rate) for one candidate/step, or None."""
    if cand == "mst_slt":
        path = mst_final(scene) if raw_step == 50000 else mst_intermediate(scene, raw_step)
    else:
        path = v48_final(scene, cand) if raw_step == 50000 else v48_intermediate(scene, cand, raw_step)
    if not path.exists():
        return None
    d = read(path)
    if raw_step == 50000 and cand == "mst_slt":
        success, collision = d["success_rate"], d["collision_rate"]
    elif raw_step == 50000:  # v4_8 final wraps rates under `summary`
        s = d["summary"]
        success, collision = s["success_rate"], s["collision_rate"]
    else:
        success, collision = d["success_rate"], d["collision_rate"]
    return float(success), float(collision)


def collect():
    data = {scene: {cand: {} for cand, *_ in CANDIDATES} for scene in SCENES}
    for scene in SCENES:
        for cand, *_ in CANDIDATES:
            for step in RAW_STEPS:
                data[scene][cand][step] = point(scene, cand, step)
    return data


def binomial_sigma(p, n=100):
    return float(np.sqrt(max(p, 0.0) * (1.0 - max(p, 0.0)) / n))


def plot_scene(ax, data, scene, metric, title):
    steps = list(RAW_STEPS)
    for cand, label, color, ls in CANDIDATES:
        ys = []
        errs = []
        for step in steps:
            pt = data[scene][cand].get(step)
            if pt is None:
                ys.append(np.nan)
                errs.append(np.nan)
                continue
            val = pt[0] if metric == "success" else pt[1]
            ys.append(val)
            errs.append(binomial_sigma(val))
        ax.errorbar(
            steps, ys, yerr=errs, label=label, color=color, linestyle=ls,
            marker="o", markersize=3.5, linewidth=1.6, capsize=2.5, alpha=0.95,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("raw steps", fontsize=9)
    ax.set_ylabel("eval success rate" if metric == "success" else "eval collision rate", fontsize=9)
    ax.set_xticks(steps)
    ax.set_xticklabels(["10k", "20k", "30k", "40k", "50k"])
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)


def plot(data, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for metric, fname, ylabel in (
        ("success", "success_curve.png", "eval success rate"),
        ("collision", "collision_curve.png", "eval collision rate"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
        for ax, scene in zip(axes, SCENES):
            title = f"{scene} — {ylabel}"
            plot_scene(ax, data, scene, metric, title)
        axes[0].legend(fontsize=8, loc="best", framealpha=0.9)
        fig.suptitle(
            "Checkpoint post-hoc eval curve (freeze policy → 100 eval episodes, no training) — "
            f"{ylabel}",
            fontsize=12, y=1.02,
        )
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out / fname}")


def fmt(v):
    return "  —  " if v is None else f"{v:4.2f}"


def build_table(data):
    lines = []
    for scene in SCENES:
        lines.append(f"### {scene}（成功率 / 碰撞率）\n")
        header = "| 候选 | 10k | 20k | 30k | 40k | 50k |"
        sep = "|---|---|---|---|---|---|"
        lines += [header, sep]
        for cand, label, *_ in CANDIDATES:
            cells = []
            for step in RAW_STEPS:
                pt = data[scene][cand].get(step)
                cells.append("—" if pt is None else f"{pt[0]:.2f}/{pt[1]:.2f}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def build_report(data, out: Path):
    table = build_table(data)
    report = f"""# checkpoint 事后评估曲线（常见 RL 画法）

纵轴 = 冻结策略跑 100 个评估 episode 的成功率（不训练）；横轴 = raw steps。
4 个中间点（10k/20k/30k/40k）来自 `tools/eval_checkpoint_curve.py` 的事后评估，
50k 末点复用 r48s4 已有最终评估（mst_slt 用 `final_evaluation.json`）。
误差棒 = 100-ep 二项 1σ（@0.9 ≈ ±0.03，@0.97 ≈ ±0.017）。

## 曲线

- 成功率：`success_curve.png`
- 碰撞率：`collision_curve.png`

{table}

## 关键观察

1. **hold35k 是唯一全程稳定且两端达标的候选**：cross 从 10k 的 0.92 起步，中段
   （30k=0.78）略回撤后恢复到 0.96/0.95，carla 全段 1.00。与 r48s4 结论一致。
2. **训练曲线会误导，评估曲线能暴露真实风险**：hold45k 训练曲线后 10k 一度 0.85/EMA
   0.870（好看），但评估曲线显示 cross 50k=0.72/碰撞 0.28（部署崩塌）；评估曲线在中段
   就已见苗头。这正是改用常见 RL 画法的动机。
3. **非单调暴跌是训练中的真实波动，不是评估噪声**：lr_half cross 10k=0.00（tau=.005
   目标更新快、早期不稳）；hold40k cross 30k=0.00；hold30k cross 40k=0.39。这些点
   远超 ±0.05 噪声，且与"越慢 tau / 越缓 ramp 越稳"的机制一致。
4. **carla 训练是非确定的**（10k 前配置相同的 4 个 hold 候选 checkpoint 权重已分叉，
   cross 则逐位相同）。但 carla 评估基本饱和（多数 1.0），该非确定性不影响"carla 无
   区分度"的既有结论；cross（有区分度的场景）是确定性的。
5. **MST+SLT 基线明显弱于 v4_8 全部候选**：cross 从 0.00 爬升到 0.77，carla 0.00→0.83，
   且中段震荡大（cross 20k=0.48/碰撞 0.52、40k=0.38）。v4_8 冠军在 10k 即达 0.92+，
   全程压制基线。

## 数据来源

- 中间点：`r48s4/eval_curve/{{scene}}__{{cand}}/raw_{{N}}/result.json`、
  `results_hd_ss100_v2/comparison/eval_curve/mst_slt__{{scene}}/raw_{{N}}/result.json`
- 50k 末点：`r48s4/eval/{{scene}}__{{cand}}/result.json`、
  `results_hd_ss100_v2/comparison/runs/hd_ss100_v2__comparison__mst_slt__{{scene}}__seed0/final_evaluation.json`
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote {out / 'REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "r48s4" / "eval_curve" / "report"))
    args = parser.parse_args()
    out = Path(args.out)
    data = collect()
    missing = [
        (s, c, st)
        for s in SCENES for c, *_ in CANDIDATES for st in RAW_STEPS
        if data[s][c].get(st) is None
    ]
    if missing:
        print(f"WARNING: {len(missing)} missing points (first 10): {missing[:10]}")
    plot(data, out)
    build_report(data, out)
    print("done")


if __name__ == "__main__":
    main()
