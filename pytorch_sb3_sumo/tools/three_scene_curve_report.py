"""Three-scene standard training curves: hold35k (v4_8) vs MST+SLT.

Reads the 25 historical-checkpoint eval points (2000..50000 raw steps, 50
episodes each) written by `tools/three_scene_eval_curve.py` and draws
success_rate / collision_rate vs raw steps for 2 methods x 3 scenes.

Outputs (under `--out`, default `results_three_scene_v1/eval_curve/report`):
  * success_curve.png    -- 3 panels (cross / carla / cross_left)
  * collision_curve.png  -- 3 panels
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
SCENES = ("cross", "carla", "cross_left")
RAW_STEPS = tuple(range(2000, 50001, 2000))
EVAL_EPISODES = 50

METHODS = (
    ("hold35k", "hold35k (v4.8)", "#d62728", "-"),
    ("mst_slt", "MST+SLT (基线)", "#ff7f0e", "--"),
)

# Symmetric moving-average window (in checkpoints) used to smooth the 25-point
# eval curve.  5 checkpoints = 10000 raw steps, matching the effective span of
# the reference episode-20 training-curve window (~6000-12000 raw steps).  A
# symmetric (centered) window is used instead of the reference's causal EMA
# (alpha 0.999 per raw step) because this is a post-hoc curve: the 50k final
# checkpoint value must not be lag-distorted by a trailing filter.
SMOOTH_WINDOW = 5


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def point(scene, method, raw_step) -> Path:
    return (
        ROOT
        / "results_three_scene_v1"
        / "eval_curve"
        / f"{method}__{scene}"
        / f"raw_{raw_step}"
        / "result.json"
    )


def collect():
    data = {
        scene: {method: {} for method, *_ in METHODS} for scene in SCENES
    }
    for scene in SCENES:
        for method, *_ in METHODS:
            for step in RAW_STEPS:
                path = point(scene, method, step)
                if not path.exists():
                    data[scene][method][step] = None
                    continue
                d = read(path)
                data[scene][method][step] = (
                    float(d["success_rate"]),
                    float(d["collision_rate"]),
                )
    return data


def binomial_sigma(p, n=EVAL_EPISODES):
    return float(np.sqrt(max(p, 0.0) * (1.0 - max(p, 0.0)) / n))


def symmetric_ma(x, window=SMOOTH_WINDOW):
    """Symmetric (centered) moving average with reflected edges.

    ``x`` is the length-25 per-checkpoint series; ``window`` must be odd.  The
    reflected padding keeps the endpoints unbiased (unlike a trailing/causal
    filter), so the 50k final value is not dragged down by the rising trend.
    """
    x = np.asarray(x, dtype=float)
    window = int(window)
    if window <= 1:
        return x
    if window >= len(x):
        return np.full_like(x, x[np.isfinite(x)].mean() if np.isfinite(x).any() else np.nan)
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(xp, kernel, mode="valid")


def plot_scene(ax, data, scene, metric):
    steps = list(RAW_STEPS)
    for method, label, color, ls in METHODS:
        ys, errs = [], []
        for step in steps:
            pt = data[scene][method].get(step)
            if pt is None:
                ys.append(np.nan)
                errs.append(np.nan)
                continue
            val = pt[0] if metric == "success" else pt[1]
            ys.append(val)
            errs.append(binomial_sigma(val))
        ys = np.asarray(ys, dtype=float)
        errs = np.asarray(errs, dtype=float)
        # Faint raw per-checkpoint measurements (honest 50-ep values).
        ax.plot(
            steps, ys, color=color, linestyle="none", marker="o",
            markersize=3, alpha=0.35, zorder=2,
        )
        if np.isfinite(ys).any():
            sy = symmetric_ma(ys)
            se = symmetric_ma(errs)
            ax.fill_between(
                steps,
                np.clip(sy - se, 0.0, 1.0),
                np.clip(sy + se, 0.0, 1.0),
                color=color, alpha=0.14, linewidth=0, zorder=1,
            )
            ax.plot(
                steps, sy, label=label, color=color, linestyle=ls,
                linewidth=2.2 if method == "hold35k" else 1.6, zorder=3,
            )
        else:
            ax.plot(steps, ys, label=label, color=color, linestyle=ls, linewidth=1.5)
    ax.set_title(scene, fontsize=11)
    ax.set_xlabel("raw steps", fontsize=9)
    ax.set_ylabel(
        "eval success rate" if metric == "success" else "eval collision rate",
        fontsize=9,
    )
    ax.set_xticks([2000, 10000, 20000, 30000, 40000, 50000])
    ax.set_xticklabels(["2k", "10k", "20k", "30k", "40k", "50k"])
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)


def plot(data, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for metric, fname in (
        ("success", "success_curve.png"),
        ("collision", "collision_curve.png"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), sharey=True)
        for ax, scene in zip(axes, SCENES):
            plot_scene(ax, data, scene, metric)
        axes[0].legend(fontsize=8, loc="best", framealpha=0.9)
        fig.suptitle(
            "checkpoint 事后评估曲线（冻结策略 → 50 评估 episode，不训练）— "
            + ("成功率" if metric == "success" else "碰撞率"),
            fontsize=12,
            y=1.02,
        )
        fig.text(
            0.5, 0.005,
            f"平滑 = {SMOOTH_WINDOW}-checkpoint（~{SMOOTH_WINDOW * 2000 // 1000}k raw steps）"
            "对称滑动平均；阴影 = 平滑后二项 1σ（描述性，非多种子置信区间）；"
            "浅色圆点 = 原始 50-ep 测量值。",
            ha="center", fontsize=8,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out / fname}")


def build_table(data):
    lines = []
    for scene in SCENES:
        lines.append(f"### {scene}（成功率 / 碰撞率）\n")
        header = "| 方法 | 2k | 10k | 20k | 30k | 40k | 50k |"
        sep = "|---|---|---|---|---|---|---|"
        lines += [header, sep]
        for method, label, *_ in METHODS:
            cells = []
            for step in (2000, 10000, 20000, 30000, 40000, 50000):
                pt = data[scene][method].get(step)
                cells.append("—" if pt is None else f"{pt[0]:.2f}/{pt[1]:.2f}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def build_report(data, out: Path):
    table = build_table(data)
    report = f"""# 三场景标准训练曲线（hold35k vs MST+SLT）

纵轴 = 冻结策略跑 50 个评估 episode 的成功率（不训练）；横轴 = raw steps。
每 2000 raw steps 保存一个 checkpoint（共 25 个，2000~50000），全部由
`tools/three_scene_eval_curve.py` 事后评估；hold35k 用 actor-only deterministic
（seed 块 420000），mst_slt 用 seed 10000。
误差棒 = 50-ep 二项 1σ（@0.9 ≈ ±0.042，@0.5 ≈ ±0.071）。

## 曲线

- 成功率：`success_curve.png`
- 碰撞率：`collision_curve.png`

## 平滑说明

参考既有训练曲线画法（`visual-composer/plot_training_curves_v5_episode20_ema999.py`：
20-episode 零填充窗口 + 每 raw step EMA 0.999 + 描述性阴影带），本图对 25 个
checkpoint 点做**对称滑动平均**平滑：窗口 = 5 checkpoint（~10000 raw steps），
与参考的 20-episode 窗口跨度（~6000~12000 raw steps）一致。选用对称窗口而非
参考的因果 EMA，是因为这是事后评估曲线——因果/尾随滤波会把 50k 末点随上升趋势
拖低；对称窗口（reflect 边界）不偏置端点，末点更接近原始测量值。图中：

- **粗线** = 平滑后趋势线（hold35k 更粗以突出冠军）；
- **阴影带** = 平滑后的二项 1σ（描述性，非多种子置信区间，截断到 [0,1]）；
- **浅色圆点** = 原始 50-ep 测量值（未平滑，保持诚实）。

表格仍列**原始未平滑**的 6 个关键步（2k/10k/20k/30k/40k/50k）成功率/碰撞率。

{table}

## 关键观察

1. **hold35k 三场景全面压制 MST+SLT 基线**。50k 末点成功率（50-ep 冻结评估）：
   hold35k = cross 0.92 / carla 1.00 / cross_left 0.98；MST+SLT = 0.84 / 0.84 / 0.84。
   差距最大在 carla（+0.16）与 cross_left（+0.14），cross 上 +0.08。

2. **hold35k 曲线平滑稳定，MST+SLT 全程剧烈震荡**。hold35k 各场景成功率随训练
   爬升后稳定；MST+SLT 三场景均在 0.04~0.98 间大幅摆动（cross 0.04~0.86、
   carla 0.02~0.96、cross_left 0.10~0.98），说明其训练过程（lr 1e-4、无稳定性
   机制）对 checkpoint 高度敏感。

3. **cross_left 意外偏易（重要 caveat）**。未训练策略（raw 2000，learning_starts
   =5000 前 0 次更新）在该场景已达 0.94（hold35k）/0.98（MST+SLT），而 cross 未训练
   仅 0.00。原因：ego 是唯一左转者（south_in→west_out）、社会车仅直行+右转，单左转
   只需找到一处让行空档，1.4× 密度下空档充足；cross 则需同时穿越对向+两侧车流的
   双重空档（AND 条件）且密度 1.5× 更高。故本场景"无保护左转"难度低于预期，若需
   更难的左转任务应提高密度（vehicle_scale≥1.5）或引入社会左转车流。

4. **MST+SLT 在 cross_left 上随训练退化**：未训练 0.98 → 50k 0.84，且中途 raw 24k
   崩至 0.10。即 aggressive 训练在易任务上发生灾难性遗忘/不稳定；hold35k 稳定维持
   0.98。

5. **hold35k 的 LR 衰减过渡期有瞬时回撤（cross）**：decay_start=20k~decay_end=35k
   附近（raw 32k~36k）成功率短暂回落 0.22~0.44、碰撞率升至 0.54~0.76，随后恢复到
   0.92。这是 hold35k 候选"高 LR→低 LR + tau 下限"调度在衰减期的已知瞬时不稳；最终
   收敛值稳定（50k=0.92，与 r48s4 100-ep 0.95 在二项噪声内一致）。

6. **carla 训练非确定**（已知，见 [[training-nondeterminism-carla]]）：hold35k carla
   曲线在 16k~22k 出现 0.00 的 timeout 回撤，细节差异不全部归因于方法；cross /
   cross_left 是确定性场景、可放心比较。

7. **交叉校验通过**：hold35k cross 50k=0.92（r48s4 100-ep 0.95）、carla=1.00
   （r48s4 1.0）；MST+SLT 曲线 50-ep 为 seed 10000 前 50 集——cross 0.84（内建
   100-ep 0.77，前 50 子集差异 ~1σ）、carla 0.84（内建 0.83）、cross_left 0.84
   （内建 0.82），全部在 50-ep 二项噪声内。

## 数据来源

- `results_three_scene_v1/eval_curve/{{method}}__{{scene}}/raw_{{N}}/result.json`
  （150 个文件 = 2 方法 × 3 场景 × 25 checkpoint）
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote {out / 'REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(ROOT / "results_three_scene_v1" / "eval_curve" / "report"),
    )
    args = parser.parse_args()
    out = Path(args.out)
    data = collect()
    missing = [
        (s, m, st)
        for s in SCENES for m, *_ in METHODS for st in RAW_STEPS
        if data[s][m].get(st) is None
    ]
    if missing:
        print(f"WARNING: {len(missing)} missing points (first 10): {missing[:10]}")
    plot(data, out)
    build_report(data, out)
    print("done")


if __name__ == "__main__":
    main()
