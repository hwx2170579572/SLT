"""Reference-exact plotting transform and explicit within-run diagnostics."""
from __future__ import annotations

import csv
import math

import numpy as np

from tools.report_phase2_curves_reference_style_v2 import transform


def read_monitor(path, budget=50000):
    rows, total = [], 0
    with path.open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(line for line in f if not line.startswith('#')):
            length = int(row['raw_simulation_steps'])
            if length <= 0:
                raise ValueError('Invalid raw episode length')
            total += length
            if total > budget:
                break
            if row['is_success'].lower() not in ('true', 'false', '1', '0'):
                raise ValueError('Invalid success value')
            reward = float(row['r'])
            if not math.isfinite(reward):
                raise ValueError('Non-finite training return')
            rows.append(dict(r=reward, is_success=row['is_success'].lower() in ('true', '1'),
                             raw_simulation_steps=length))
    if not rows:
        raise ValueError('No complete episodes in training log')
    return rows


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                variance=float(a.var()) if len(a) else None,
                std=float(a.std()) if len(a) else None)


def sustained_crossing(curve, threshold=.8, duration=5000):
    t, y = curve[:, 0], curve[:, 1]
    start = None
    for ti, yi in zip(t, y):
        if yi < threshold:
            start = None
        else:
            if start is None:
                start = ti
            if ti - start >= duration:
                return int(start)
    return None


def curve_stats(a):
    t, y = a[:, 0], a[:, 1]
    tail = y[t >= 40000]
    post = y[t >= 5000]
    trough_index = int(np.argmax(np.maximum.accumulate(post) - post))
    slope = float(np.polyfit(t[t >= 40000] / 1000, tail, 1)[0]) if len(tail) > 1 else None
    return dict(endpoint_raw_step=int(t[-1]), endpoint=float(y[-1]),
                mean_auc=float(np.trapz(y, t) / t[-1]),
                values_at_steps={str(s): float(np.interp(s, t, y)) for s in (10000, 20000, 30000, 40000) if s <= t[-1]},
                max_drawdown_after_5k=float((np.maximum.accumulate(post) - post).max()),
                drawdown_trough_raw_step=int(t[t >= 5000][trough_index]),
                tail_curve_std=float(tail.std()) if len(tail) else None,
                tail_slope_per_1k=slope,
                excess_total_variation=float(np.abs(np.diff(post)).sum() - abs(post[-1] - post[0])),
                tail_excess_total_variation=float(np.abs(np.diff(tail)).sum() - abs(tail[-1] - tail[0])) if len(tail) else None,
                peak_to_tail_mean=float(post.max() - tail.mean()) if len(tail) else None)


def analyze_rows(rows):
    ends = np.cumsum([r['raw_simulation_steps'] for r in rows])
    arrays = {metric: transform(rows, metric) for metric in ('reward', 'success')}
    result = dict(complete_episodes=len(rows), last_complete_raw_step=int(ends[-1]),
                  training_seed_count=1, variance_scope='within one seed-0 training run; not across-seed uncertainty')
    for metric, key in (('reward', 'r'), ('success', 'is_success')):
        values = np.asarray([r[key] for r in rows], dtype=float)
        bins = [dict(raw_step_end=stop, **distribution(values[(ends > stop - 1000) & (ends <= stop)]))
                for stop in range(1000, 50001, 1000)]
        result[metric] = dict(raw_episodes=distribution(values),
                              tail_raw_episodes=distribution(values[ends > 40000]),
                              raw_1k_bins=bins, curve=curve_stats(arrays[metric]))
    success = result['success']['curve']
    success['sustained_80pct_first_raw_step'] = sustained_crossing(arrays['success'])
    success['late_plateau_heuristic'] = bool(
        success['tail_curve_std'] is not None and
        np.mean(arrays['success'][arrays['success'][:, 0] >= 40000, 1]) >= .8 and
        success['tail_curve_std'] <= .03 and abs(success['tail_slope_per_1k']) <= .005)
    success['performance_collapse_flag'] = success['max_drawdown_after_5k'] >= .20
    # This flag is performance drawdown, never a claim of numerical divergence.
    return result, arrays


def draw_curves(curves, labels, directory, name, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8.5))
    colors = ['#0072B2', '#777777', '#D55E00', '#009E73', '#CC79A7']
    for ax, metric in zip(axes, ('reward', 'success')):
        for i, (key, values) in enumerate(curves.items()):
            a = values[metric]
            ax.plot(a[:, 0], a[:, 1], label=labels[key], color=colors[i % len(colors)],
                    lw=2.3 if key == 'lr_half' else 1.6, ls='--' if key == 'lr_quarter' else '-')
            ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=colors[i % len(colors)], alpha=.09)
        ax.set(xlim=(0, 50000), xlabel='Raw simulation steps',
               ylabel='Average episode reward' if metric == 'reward' else 'Average success rate')
        if metric == 'success':
            ax.set_ylim(0, 1)
        ax.grid(alpha=.3)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title + ' | seed 0', fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .95), ncol=2, fontsize=9)
    fig.text(.5, .012, '20-episode zero-padded window; EMA 0.999 at every raw step.\nBands describe one run; they are not cross-seed confidence intervals.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .055, 1, .86])
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(directory / f'{name}.{extension}', dpi=180)
    plt.close(fig)
