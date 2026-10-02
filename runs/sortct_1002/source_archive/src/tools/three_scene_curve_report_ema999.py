"""Reference-faithful EMA-0.999 version of the three-scene eval curves (对照版).

Contrast with `tools/three_scene_curve_report.py` (symmetric moving-average
window).  This script reproduces the released training-curve transform from
`visual-composer/plot_training_curves_v5_episode20_ema999.py` verbatim, applied
to the 25-checkpoint eval series (2000..50000 raw steps, 50 episodes each):

  * each checkpoint's 50-ep success/collision rate is zero-order held over its
    *preceding* 2000 raw-step interval (mirroring "hold the episode window over
    that episode's observed raw steps");
  * a causal EMA ``y[t] = 0.999*y[t-1] + 0.001*x[t]``, initial value 0, is
    applied once per raw step -- the reference's exact "EMA 0.999 at every raw
    step";
  * the binomial 1-sigma SE is EMA-smoothed identically and clipped to [0,1].

The reference's 20-episode zero-padded window has no separate analogue here:
the 50-episode aggregate already plays that role.  This version therefore shows
the reference transform's own behaviour on sparse checkpoint data -- the early
zero-depression (curve rises from 0) and a mild trailing lag -- for side-by-side
comparison with the symmetric window of the sibling script.

Outputs (under `results_three_scene_v1/eval_curve/report_ema999`):
  * success_curve_ema999.png / collision_curve_ema999.png
  * README.md -- one-line comparison note
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tools.three_scene_curve_report import (
    EVAL_EPISODES,
    METHODS,
    RAW_STEPS,
    ROOT,
    SCENES,
    binomial_sigma,
    point,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EMA_ALPHA = 0.999   # old-value weight, per raw step (reference-exact)
SAMPLE_STRIDE = 100  # render every 100 raw steps (reference-exact)


def series(data, scene, method, metric):
    """Return (x=raw steps, y=rate, se=binomial 1σ) for one method/scene."""
    xs = list(RAW_STEPS)
    vals, ses = [], []
    for step in xs:
        pt = data[scene][method].get(step)
        if pt is None:
            raise FileNotFoundError(f"missing point {method}__{scene} raw_{step}")
        val = pt[0] if metric == "success" else pt[1]
        vals.append(float(val))
        ses.append(binomial_sigma(val, EVAL_EPISODES))
    return (
        np.asarray(xs, dtype=float),
        np.asarray(vals, dtype=float),
        np.asarray(ses, dtype=float),
    )


def ema999_curve(steps, values, ses):
    """Zero-order hold + per-raw-step causal EMA (alpha=0.999), init 0.

    Returns dense (sample_x, ema_y, band_lower, band_upper) arrays sampled every
    ``SAMPLE_STRIDE`` raw steps, exactly as the reference transform.
    """
    total = int(steps[-1])
    held = np.zeros(total, dtype=float)
    held_se = np.zeros(total, dtype=float)
    prev = 0
    for s, v, se in zip(steps, values, ses):
        held[prev:int(s)] = v
        held_se[prev:int(s)] = se
        prev = int(s)
    ema = np.zeros(total + 1, dtype=float)
    ema_se = np.zeros(total + 1, dtype=float)
    for t in range(total):
        ema[t + 1] = EMA_ALPHA * ema[t] + (1.0 - EMA_ALPHA) * held[t]
        ema_se[t + 1] = EMA_ALPHA * ema_se[t] + (1.0 - EMA_ALPHA) * held_se[t]
    sample_times = list(range(0, total + 1, SAMPLE_STRIDE))
    if sample_times[-1] != total:
        sample_times.append(total)
    xs = np.asarray(sample_times, dtype=float)
    y = ema[xs.astype(int)]
    se = ema_se[xs.astype(int)]
    return xs, y, np.clip(y - se, 0.0, 1.0), np.clip(y + se, 0.0, 1.0)


def plot_scene(ax, data, scene, metric):
    for method, label, color, ls in METHODS:
        xs, vals, ses = series(data, scene, method, metric)
        sx, sy, lo, hi = ema999_curve(xs, vals, ses)
        ax.fill_between(sx, lo, hi, color=color, alpha=0.14, linewidth=0)
        ax.plot(
            sx, sy, label=label, color=color, linestyle=ls,
            linewidth=2.2 if method == "hold35k" else 1.6,
        )
    ax.set_title(scene, fontsize=11)
    ax.set_xlabel("raw steps", fontsize=9)
    ax.set_ylabel(
        "eval success rate" if metric == "success" else "eval collision rate",
        fontsize=9,
    )
    ax.set_xlim(0, RAW_STEPS[-1])
    ax.set_xticks(np.arange(0, RAW_STEPS[-1] + 1, 10000))
    ax.set_xticklabels(["0", "10k", "20k", "30k", "40k", "50k"])
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)


def plot(data, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for metric, fname in (
        ("success", "success_curve_ema999.png"),
        ("collision", "collision_curve_ema999.png"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), sharey=True)
        for ax, scene in zip(axes, SCENES):
            plot_scene(ax, data, scene, metric)
        axes[0].legend(fontsize=8, loc="best", framealpha=0.9)
        fig.suptitle(
            "checkpoint 事后评估曲线（参考画法：每 raw step 因果 EMA 0.999，不训练）— "
            + ("成功率" if metric == "success" else "碰撞率"),
            fontsize=12,
            y=1.02,
        )
        fig.text(
            0.5, 0.005,
            "zero-order hold + EMA y[t]=0.999·y[t-1]+0.001·x[t]（初值 0，每 raw step）；"
            "阴影 = EMA 后的二项 1σ（描述性，非多种子置信区间）。",
            ha="center", fontsize=8,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out / fname}")


def main():
    from tools.three_scene_curve_report import collect

    out = ROOT / "results_three_scene_v1" / "eval_curve" / "report_ema999"
    data = collect()
    plot(data, out)
    (out / "README.md").write_text(
        "EMA-0.999 参考画法对照版：与同目录上层 `report/`（对称滑动平均 window=5）"
        "使用同一 25 点 checkpoint 数据，仅平滑变换不同。\n\n"
        "本版忠实复现 `visual-composer/plot_training_curves_v5_episode20_ema999.py`"
        "的因果 EMA（每 raw step y=0.999·y_prev+0.001·x，初值 0）：稀疏 checkpoint 下"
        "表现为前段从 0 逐渐抬升 + 尾随滞后。对照可见对称窗口版端点不偏置、更平滑。\n",
        encoding="utf-8",
    )
    print(f"wrote {out / 'README.md'}")
    print("done")


if __name__ == "__main__":
    main()
