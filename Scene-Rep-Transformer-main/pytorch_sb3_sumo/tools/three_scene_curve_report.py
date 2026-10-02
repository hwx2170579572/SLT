"""Three-scene standard training curves: hold35k (v4_8) vs MST+SLT.

Reads the 25 historical-checkpoint eval points (2000..50000 raw steps, 50
episodes each) written by `tools/three_scene_eval_curve.py` and draws
success_rate / collision_rate vs raw steps for 9 methods x 4 trained scenes
(cross / carla / cross_left / cross_left_unreg), plus empty zero-shot panels.

Outputs (under `--out`, default `results_three_scene_v1/eval_curve/report`):
  * success_curve.png    -- panels for every scene in SCENES
  * collision_curve.png  -- panels for every scene in SCENES
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
# The four *trained* scenes plus the two new zero-shot scenes.  merge /
# intersection have no training curve (they were never trained), so their
# panels and table rows are empty (honest "no data"), matching the zero-shot
# visual/video pipeline that reuses the source scenes' final models.
SCENES = ("cross", "carla", "cross_left", "cross_left_unreg", "merge", "intersection")
ZERO_SHOT_SCENES = ("merge", "intersection")
RAW_STEPS = tuple(range(2000, 50001, 2000))
EVAL_EPISODES = 50

METHODS = (
    ("hold35k", "TASAC (hold35k)", "#d62728", "-"),
    ("mst_slt", "MST+SLT", "#ff7f0e", "--"),
    ("hsac_mlp", "HSAC-MLP", "#1f77b4", "-"),
    ("hsac_lstm", "HSAC-LSTM", "#1f77b4", "--"),
    ("gnn_sac", "GNN+SAC", "#2ca02c", "-"),
    ("mst", "MST (no SLT)", "#2ca02c", "--"),
    ("hybrid_dt", "Hybrid-DT", "#9467bd", "-"),
    ("decision_transformer", "Decision Transformer", "#9467bd", "--"),
    ("hyar", "HyAR", "#17becf", "-"),
)

# Reference methods highlighted against the seven comparison baselines.
REFERENCE_METHODS = ("hold35k", "mst_slt")

# Symmetric moving-average window (in checkpoints) used to smooth the 25-point
# eval curve.  5 checkpoints = 10000 raw steps, matching the effective span of
# the reference episode-20 training-curve window (~6000-12000 raw steps).  A
# symmetric (centered) window is used instead of the reference's causal EMA
# (alpha 0.999 per raw step) because this is a post-hoc curve: the 50k final
# checkpoint value must not be lag-distorted by a trailing filter.
SMOOTH_WINDOW = 5

# Second curve style: a zero-padded causal running mean (in checkpoints).  The
# value at checkpoint i is the mean of the last ZERO_PAD_WINDOW checkpoints
# with the left edge zero-padded — i.e. when fewer than ZERO_PAD_WINDOW
# checkpoints precede i, the missing ones count as 0.  10 checkpoints ≈ 20000
# raw steps, matching the reference episode-20 zero-padding training window.
ZERO_PAD_WINDOW = 10

# Third curve style: the full reference training-curve pipeline
# (visual-composer/plot_training_curves_v5_episode20_ema999.py).  The 25
# per-checkpoint success rates stand in for episode outcomes: a zero-padded
# window over the last EMA_WINDOW checkpoints is held over each checkpoint's
# raw-step span and then smoothed once per raw step with EMA alpha=0.999.
EMA_WINDOW = 20
EMA_ALPHA = 0.999
STEPS_PER_CHECKPOINT = 2000
EMA_SAMPLE_STRIDE = 100


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


def zero_padded_mean(x, window=ZERO_PAD_WINDOW):
    """Zero-padded causal running mean (in checkpoints).

    ``x`` is the length-25 per-checkpoint series.  The value at checkpoint ``i``
    is ``mean(x[max(0, i-window+1) : i+1])`` with the left edge padded with
    zeros so the denominator is always ``window``: the first ``window-1``
    checkpoints are averaged against zeros, then the window slides causally.
    This is the 10-checkpoint (~20000 raw steps) zero-padded window matching
    the reference episode-20 zero-padding training-curve scheme.  NaN entries
    (missing checkpoints) count as 0.
    """
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    cs = np.cumsum(x)
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo = i - window
        prev = cs[lo] if lo >= 0 else 0.0
        out[i] = (cs[i] - prev) / window
    return out


def padded_window_stats(values, window=EMA_WINDOW):
    """Zero-padded window mean, population SD and observed-count.

    Mirrors ``padded_episode_window`` in the reference script: the value at
    checkpoint ``i`` averages the last ``window`` checkpoints with explicit
    zero slots for missing early checkpoints (the denominator is always
    ``window``).
    """
    values = np.asarray(values, dtype=float)
    means = np.zeros(len(values), dtype=float)
    stds = np.zeros(len(values), dtype=float)
    counts = np.zeros(len(values), dtype=int)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        observed = values[start : i + 1]
        slots = np.zeros(window, dtype=float)
        slots[-len(observed):] = observed
        means[i] = float(slots.mean())
        stds[i] = float(slots.std(ddof=0))
        counts[i] = int(len(observed))
    return means, stds, counts


def build_ema_curve(
    values,
    *,
    window=EMA_WINDOW,
    ema_alpha=EMA_ALPHA,
    steps_per_checkpoint=STEPS_PER_CHECKPOINT,
    sample_stride=EMA_SAMPLE_STRIDE,
):
    """Full reference pipeline: zero-padded window -> hold over raw steps ->
    EMA per raw step -> sample.

    ``values`` is the length-25 per-checkpoint success/collision series (NaN
    entries are treated as 0).  Returns ``(xs, mean, lower, upper)`` sampled
    every ``sample_stride`` raw steps (plus the terminal step), exactly like
    ``build_time_step_curve`` in the reference script.
    """
    values = np.where(np.isfinite(values), values, 0.0)
    means, stds, _counts = padded_window_stats(values, window)
    se = stds / float(window) ** 0.5
    step_mean = np.repeat(means, steps_per_checkpoint)
    step_se = np.repeat(se, steps_per_checkpoint)
    total = len(values) * steps_per_checkpoint
    ema_mean = np.zeros(total + 1, dtype=float)
    ema_se = np.zeros(total + 1, dtype=float)
    for t in range(total):
        ema_mean[t + 1] = ema_alpha * ema_mean[t] + (1.0 - ema_alpha) * step_mean[t]
        ema_se[t + 1] = ema_alpha * ema_se[t] + (1.0 - ema_alpha) * step_se[t]
    lower = np.clip(ema_mean - ema_se, 0.0, 1.0)
    upper = np.clip(ema_mean + ema_se, 0.0, 1.0)
    times = list(range(0, total + 1, sample_stride))
    if times[-1] != total:
        times.append(total)
    times = np.asarray(times, dtype=int)
    return times, ema_mean[times], lower[times], upper[times]


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
        fig, axes = plt.subplots(
            1, len(SCENES), figsize=(6 * len(SCENES), 5.2), sharey=True
        )
        for ax, scene in zip(axes, SCENES):
            plot_scene(ax, data, scene, metric)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, fontsize=7, loc="lower center",
            ncol=5, framealpha=0.9, bbox_to_anchor=(0.5, -0.04),
        )
        fig.suptitle(
            "checkpoint 事后评估曲线（冻结策略 → 50 评估 episode，不训练）— "
            + ("成功率" if metric == "success" else "碰撞率"),
            fontsize=12,
            y=1.02,
        )
        fig.text(
            0.5, 0.01,
            f"平滑 = {SMOOTH_WINDOW}-checkpoint（~{SMOOTH_WINDOW * 2000 // 1000}k raw steps）"
            "对称滑动平均；阴影 = 平滑后二项 1σ（描述性，非多种子置信区间）；"
            "浅色圆点 = 原始 50-ep 测量值。",
            ha="center", fontsize=8,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out / fname}")


def plot_scene_zero_padded(ax, data, scene, metric):
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
            sy = zero_padded_mean(ys)
            # std of a window-mean = sqrt(sum sigma_i^2) / window.
            se = np.sqrt(zero_padded_mean(errs ** 2) / ZERO_PAD_WINDOW)
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
        "eval success (zero-pad 10-ckpt mean)" if metric == "success"
        else "eval collision (zero-pad 10-ckpt mean)",
        fontsize=9,
    )
    ax.set_xticks([2000, 10000, 20000, 30000, 40000, 50000])
    ax.set_xticklabels(["2k", "10k", "20k", "30k", "40k", "50k"])
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_zero_padded(data, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for metric, fname in (
        ("success", "success_curve_zero10.png"),
        ("collision", "collision_curve_zero10.png"),
    ):
        fig, axes = plt.subplots(
            1, len(SCENES), figsize=(6 * len(SCENES), 5.2), sharey=True
        )
        for ax, scene in zip(axes, SCENES):
            plot_scene_zero_padded(ax, data, scene, metric)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, fontsize=7, loc="lower center",
            ncol=5, framealpha=0.9, bbox_to_anchor=(0.5, -0.04),
        )
        fig.suptitle(
            "checkpoint 事后评估曲线（冻结策略 → 50 评估 episode，不训练）— "
            + (
                "成功率（10-checkpoint 零填充均值）" if metric == "success"
                else "碰撞率（10-checkpoint 零填充均值）"
            ),
            fontsize=12,
            y=1.02,
        )
        fig.text(
            0.5, 0.01,
            f"纵轴 = 当前及之前 {ZERO_PAD_WINDOW - 1} 个检查点的平均"
            f"（不足 {ZERO_PAD_WINDOW} 个补 0）；"
            f"窗口 = {ZERO_PAD_WINDOW} checkpoint"
            f"（~{ZERO_PAD_WINDOW * 2000 // 1000}k raw steps）；"
            "阴影 = 窗口均值二项 1σ；浅色圆点 = 原始 50-ep 测量值。",
            ha="center", fontsize=8,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out / fname}")


def plot_scene_ema(ax, data, scene, metric):
    for method, label, color, ls in METHODS:
        ys = []
        for step in RAW_STEPS:
            pt = data[scene][method].get(step)
            ys.append(np.nan if pt is None else (pt[0] if metric == "success" else pt[1]))
        ys = np.asarray(ys, dtype=float)
        # Faint raw per-checkpoint measurements (honest 50-ep values).
        ax.plot(
            list(RAW_STEPS), ys, color=color, linestyle="none", marker="o",
            markersize=3, alpha=0.35, zorder=2,
        )
        if np.isfinite(ys).any():
            xs, mean, lower, upper = build_ema_curve(ys)
            ax.fill_between(
                xs, lower, upper, color=color, alpha=0.14, linewidth=0, zorder=1,
            )
            ax.plot(
                xs, mean, label=label, color=color, linestyle=ls,
                linewidth=2.2 if method == "hold35k" else 1.6, zorder=3,
            )
        else:
            ax.plot(list(RAW_STEPS), ys, label=label, color=color, linestyle=ls, linewidth=1.5)
    ax.set_title(scene, fontsize=11)
    ax.set_xlabel("raw steps", fontsize=9)
    ax.set_ylabel(
        "EMA success rate (α=0.999)" if metric == "success"
        else "EMA collision rate (α=0.999)",
        fontsize=9,
    )
    ax.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
    ax.set_xticklabels(["0", "10k", "20k", "30k", "40k", "50k"])
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_ema(data, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for metric, fname in (
        ("success", "success_curve_ema999.png"),
        ("collision", "collision_curve_ema999.png"),
    ):
        fig, axes = plt.subplots(
            1, len(SCENES), figsize=(6 * len(SCENES), 5.2), sharey=True
        )
        for ax, scene in zip(axes, SCENES):
            plot_scene_ema(ax, data, scene, metric)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, fontsize=7, loc="lower center",
            ncol=5, framealpha=0.9, bbox_to_anchor=(0.5, -0.04),
        )
        fig.suptitle(
            "checkpoint 事后评估曲线（冻结策略 → 50 评估 episode，不训练）— "
            + (
                "成功率（20-checkpoint 零填充 + EMA α=0.999）" if metric == "success"
                else "碰撞率（20-checkpoint 零填充 + EMA α=0.999）"
            ),
            fontsize=12,
            y=1.02,
        )
        fig.text(
            0.5, 0.01,
            f"参考画法：{EMA_WINDOW}-checkpoint 零填充窗口 → 每 raw step EMA α={EMA_ALPHA}"
            f"（每 {STEPS_PER_CHECKPOINT} 步一个 checkpoint）；"
            "阴影 = 窗口 population SD/√窗口 经同一 EMA 后的 ±1；浅色圆点 = 原始 50-ep 测量值。",
            ha="center", fontsize=8,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 1])
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
    final_rows = []
    for scene in SCENES:
        cells = []
        for method, label, *_ in METHODS:
            pt = data[scene][method].get(50000)
            cells.append(f"{label}={pt[0]:.2f}" if pt else f"{label}=—")
        final_rows.append(f"- **{scene}**: " + "，".join(cells))
    final_summary = "\n".join(final_rows)
    report = f"""# 三场景标准训练曲线（TASAC + MST+SLT + 7 对比基线）

> **新增场景说明**：
> - `cross_left_unreg`（完全无管制四臂交叉口、70 m 车道、四个进口道各支持
>   直行/左转/右转，0.2 veh/s/进口道）是本项目新增的**第 4 个训练场景**，其
>   训练曲线与另三个训练场景一并绘制。
> - `merge`（高速+匝道合流）与 `intersection`（无信号交叉口左转）是另外两个新增
>   SUMO 场景，**没有训练曲线**——它们从未被训练，本报告对其仅作占位（表格显示
>   `—`、曲线图为空面板）。这两个场景的零样本表现由
>   `tools/three_scene_visual_verification.py` / `tools/three_scene_video_capture.py`
>   复用源场景（merge→cross、intersection→cross_left）最终模型回放，不在此处。

纵轴 = 冻结策略跑 50 个评估 episode 的成功率（不训练）；横轴 = 训练步。
每 2000 步保存一个 checkpoint（共 25 个，2000~50000），全部由
`tools/three_scene_eval_curve.py` 事后评估。

- **SB3 在线基线**（hsac_mlp / hsac_lstm / gnn_sac / mst）：横轴 = raw 环境步，
  与 hold35k / mst_slt 一致；actor-only deterministic 评估（v4_8 方法 seed 块
  420000，base 方法 seed 10000）。
- **序列/潜空间离线基线**（hybrid_dt / decision_transformer / hyar）：横轴 = 离线
  梯度步。它们先模仿一个已训练的强基线策略采集成数据集（hybrid_dt / hyar 从
  HSAC-MLP，decision_transformer 从 GNN+SAC，各 100 episode），再离线训练 50000
  梯度步。评估为 return-to-go 条件自回归 rollout（HyAR 为潜空间策略 rollout）。
- 误差棒 = 50-ep 二项 1σ（@0.9 ≈ ±0.042，@0.5 ≈ ±0.071）。

## 方法分类（动作头 × 编码器）

| 方法 | 动作头 | 编码器 |
|---|---|---|
| TASAC (hold35k) | 混合 | 图 + MST（场景表征） |
| MST+SLT | 连续 | MST transformer + SLT |
| HSAC-MLP | 混合 | MLP |
| HSAC-LSTM | 混合 | LSTM |
| GNN+SAC | 连续 | 车辆图卷积（DGN 式） |
| MST (no SLT) | 连续 | MST transformer（无 SLT） |
| Hybrid-DT | 混合 | GPT-2 序列 |
| Decision Transformer | 连续 | GPT-2 序列 |
| HyAR | 混合 | VAE 潜空间 |

## 曲线

- 成功率（对称滑动平均）：`success_curve.png`
- 碰撞率（对称滑动平均）：`collision_curve.png`
- 成功率（10-checkpoint 零填充均值）：`success_curve_zero10.png`
- 碰撞率（10-checkpoint 零填充均值）：`collision_curve_zero10.png`
- 成功率（20-checkpoint 零填充 + EMA）：`success_curve_ema999.png`
- 碰撞率（20-checkpoint 零填充 + EMA）：`collision_curve_ema999.png`

## 平滑说明

对 25 个 checkpoint 点做**对称滑动平均**平滑：窗口 = 5 checkpoint（~10000 步）。
选用对称窗口（reflect 边界）而非因果 EMA，是因为这是事后评估曲线——因果/尾随滤波
会把 50k 末点随上升趋势拖低；对称窗口不偏置端点。图中：

- **粗线** = 平滑后趋势线（hold35k 加粗以突出冠军）；
- **阴影带** = 平滑后的二项 1σ（描述性，非多种子置信区间，截断到 [0,1]）；
- **浅色圆点** = 原始 50-ep 测量值（未平滑，保持诚实）。

表格仍列**原始未平滑**的 6 个关键步（2k/10k/20k/30k/40k/50k）成功率/碰撞率。

### 第二种画法：10-checkpoint 零填充均值

第二组图（`*_zero10.png`）把纵轴改为「当前及之前 9 个检查点（不足 10 个补 0）的平均
成功率」：第 i 个检查点的值 = mean(x[max(0,i-9)..i]，左侧用 0 补齐到 10 项）。前 9 个
检查点因分母固定为 10、缺项记 0 而被系统性压低，第 10 个检查点（raw 20000）起是完整
10 项均值（窗口 ~20000 raw steps）。这种零填充窗口比对称滑动平均更贴合在线训练曲线的
画法（参考 `plot_training_curves_v5_episode20_ema999.py` 的 20-episode 零填充），能更
清楚地看出「从训练开始累计」的平均成功率爬升；代价是早期点被压低、末点反映的是最近
10 个检查点的平均而非该检查点本身。阴影带 = 窗口均值的二项 1σ（√(Σσᵢ²)/10，较窄）。

### 第三种画法：20-checkpoint 零填充 + EMA α=0.999（参考训练曲线画法）

第三组图（`*_ema999.png`）完整复刻参考脚本 `plot_training_curves_v5_episode20_ema999.py`
的「训练过程中平均成功率」画法：把 25 个 checkpoint 的成功率当作 episode 结果，先做
20-checkpoint 零填充窗口（分母固定 20、缺项记 0），再把每个 checkpoint 的窗口统计量
held 到它的 2000 步 raw-step 跨度上，随后**每 raw step** 做 EMA（`ema_t = 0.999·ema_{{t-1}}
+ 0.001·current_t`，初始 0），最后每 100 步采样画图。阴影带 = 窗口 population SD/√20 经
同一 EMA 后的 ±1。EMA 初始为 0 且半衰期 ~693 步，故曲线前半段有 warm-up 爬升、整体比
前两种画法更滞后平滑，贴合在线训练曲线的视觉习惯。

{table}

## 关键观察（50k 末点成功率，50-ep 冻结评估）

{final_summary}

## 数据来源

- `results_three_scene_v1/eval_curve/{{method}}__{{scene}}/raw_{{N}}/result.json`
  （900 个文件 = 9 方法 × 4 训练场景 × 25 checkpoint；merge/intersection 无训练曲线）
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
    plot_zero_padded(data, out)
    plot_ema(data, out)
    build_report(data, out)
    print("done")


if __name__ == "__main__":
    main()
