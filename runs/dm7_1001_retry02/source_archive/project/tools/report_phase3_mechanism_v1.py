"""Versioned, source-backed episode20/EMA999 figures; never merge parameters."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tools.phase3_mechanism_v1.common import ROOT, OUT, SCENES, MECHANISM_SCENES, digest, read, sha, write
from tools.phase3_mechanism_v1.model import VARIANTS
from tools.report_phase2_curves_reference_style_v2 import transform, REFERENCE

DEST = OUT / 'figures_v1'
PRIOR = ROOT / 'results_phase2_training_curves_v2_episode20_ema999'
TITLES = dict(zip(SCENES, ('Left-turn', 'Cross', 'Roundabout-A', 'Roundabout-B', 'Roundabout-C', 'CARLA')))
CANDIDATES = (
    ('mst_slt', 'control', 'MST+SLT / default', '#666666', '--'),
    ('v4_8', 'lr_half', 'v4.8 / LR x 0.5', '#0072B2', '-'),
    ('v4_13', 'tau_half', 'v4.13 / tau x 0.5', '#009E73', '-.'),
)
FACTORIAL = (
    ('g1_s025', 'Full: isolate, lane x 0.25', '#009E73', '-'),
    ('g0_s025', 'No isolation, lane x 0.25', '#E69F00', '--'),
    ('g1_s1', 'Isolate, lane x 1', '#CC79A7', ':'),
    ('g0_s1', 'No isolation, lane x 1', '#0072B2', '-.'),
    ('g0_s1_lowall', 'No isolation, all support x 0.25', '#D55E00', (0, (5, 1, 1, 1, 1, 1))),
)


def monitor_snapshot(path):
    raw = path.read_bytes()
    # Ignore only a concurrently written final incomplete line, retaining its
    # bytes in the reproducible snapshot and recording how many were omitted.
    end = raw.rfind(b'\n') + 1
    stream = io.StringIO(raw[:end].decode('utf-8-sig'))
    rows, total = [], 0
    for row in csv.DictReader(line for line in stream if not line.startswith('#')):
        steps = int(row['raw_simulation_steps'])
        flag = row['is_success'].lower()
        if steps <= 0 or flag not in ('true', 'false', '1', '0'):
            raise ValueError(f'Invalid episode in {path}')
        reward = float(row['r'])
        if not np.isfinite(reward):
            raise ValueError('Nonfinite training reward')
        if total + steps > 50000:
            break
        rows.append({'r': reward, 'is_success': flag in ('true', '1'), 'raw_simulation_steps': steps})
        total += steps
    return raw, rows, {'complete_episodes': len(rows), 'last_complete_raw_step': total,
                       'ignored_incomplete_line_bytes': len(raw) - end}


def build(include_running=False):
    prior_sources, prior_values = read(PRIOR / 'source_manifest.json'), read(PRIOR / 'curve_values.json')
    data, sources, snapshots, validation = {}, {}, {}, []

    def add(key, path, complete, expected=None, reference_key=None):
        raw, rows, stats = monitor_snapshot(path)
        actual = hashlib.sha256(raw).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError(f'Frozen log changed: {key}')
        record = {'training_log': path.relative_to(ROOT).as_posix(), 'snapshot_sha256': actual,
                  'training_complete': complete, **stats}
        sources[key], snapshots[key] = record, raw
        if not rows:
            return
        curves = {metric: transform(rows, metric) for metric in ('reward', 'success')}
        for metric, values in curves.items():
            if not np.isfinite(values).all() or values[-1, 0] != stats['last_complete_raw_step']:
                raise ValueError('Curve does not match complete observed episodes')
            if metric == 'success' and (np.min(values[:, 1:]) < 0 or np.max(values[:, 1:]) > 1):
                raise ValueError('Success interval outside [0,1]')
            if reference_key:
                reference = np.asarray(prior_values[reference_key][metric])
                if values.shape != reference.shape:
                    raise ValueError('Prior curve shape changed')
                error = float(np.max(np.abs(values - reference)))
                if error > 1e-12:
                    raise ValueError('Prior reference transform changed')
                validation.append({'source': key, 'metric': metric, 'points': len(values), 'max_absolute_error': error})
        data[key] = curves

    for scene in SCENES:
        for method, candidate, *_ in CANDIDATES:
            key = f'{method}__{scene}__{candidate}'
            prior_key = key + '__seed0'
            source = prior_sources[prior_key]
            add(key, ROOT / source['training_log'], True, source['training_log_sha256'], prior_key)
    missing = []
    for scene in MECHANISM_SCENES:
        for variant in VARIANTS:
            if variant == 'g1_s025':
                continue
            key = f'{variant}__{scene}'
            directory = OUT / 'train' / key
            receipt = directory / 'training_complete.json'
            complete = receipt.is_file()
            if complete:
                result = read(receipt)
                if result['smoke'] or result['raw_steps'] != 50000 or result['variant'] != variant:
                    raise ValueError('Invalid completed scientific training')
                if sha(directory / 'final_model.zip') != result['checkpoint_sha256']:
                    raise ValueError('Final training model changed')
            path = directory / 'train_monitor.csv'
            if not path.is_file() or (not complete and not include_running):
                missing.append(key)
                continue
            add(key, path, complete)
            if complete:
                sources[key]['completion_receipt'] = receipt.relative_to(ROOT).as_posix()
                sources[key]['completion_receipt_sha256'] = sha(receipt)

    identity = {'sources': sources, 'missing_training': missing, 'include_running': include_running,
                'script_sha256': sha(Path(__file__)), 'reference_script_sha256': sha(REFERENCE),
                'transform_script_sha256': sha(ROOT / 'tools/report_phase2_curves_reference_style_v2.py'),
                'protocol_sha256': sha(OUT / 'manifest.json')}
    destination = DEST / 'snapshots' / digest(identity)[:16]
    receipt = destination / 'completion.json'
    if receipt.exists():
        for filename, expected in read(receipt)['artifacts'].items():
            if sha(destination / filename) != expected:
                raise ValueError('A completed figure snapshot changed')
        write(DEST / 'latest.json', {'snapshot': destination.relative_to(ROOT).as_posix()})
        return {'reused': True, 'snapshot': destination.relative_to(ROOT).as_posix()}
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'logs').mkdir()
    for key, raw in snapshots.items():
        (destination / 'logs' / f'{key}.csv').write_bytes(raw)
    write(destination / 'source_manifest.json', identity)
    write(destination / 'curve_values.json', {key: {metric: values.tolist() for metric, values in curves.items()} for key, curves in data.items()})
    write(destination / 'validation.json', {'prior_comparisons': validation, 'all_points_finite': True,
                                          'no_parameter_averaging': True, 'incomplete_episode_not_extrapolated': True})
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none', 'pdf.fonttype': 42})

    def series(group, scene):
        if group == 'candidates':
            return [(f'{method}__{scene}__{candidate}', label, color, style) for method, candidate, label, color, style in CANDIDATES]
        return [(f'v4_13__{scene}__tau_half' if variant == 'g1_s025' else f'{variant}__{scene}', label, color, style)
                for variant, label, color, style in FACTORIAL]

    def draw(ax, group, scene, metric, local_status=True):
        handles = []
        for key, label, color, style in series(group, scene):
            if key not in data:
                handles.append(Line2D([], [], color=color, ls=style, label=label + (' [pending]' if local_status else '')))
                continue
            values = data[key][metric]
            partial = not sources[key]['training_complete']
            display = label + (' [running]' if partial and local_status else '')
            ax.fill_between(values[:, 0], values[:, 2], values[:, 3], color=color, alpha=.10, linewidth=0)
            line, = ax.plot(values[:, 0], values[:, 1], color=color, ls=style, lw=2.1, label=display)
            if partial:
                ax.plot(values[-1, 0], values[-1, 1], marker='o', ms=5, color=color)
            handles.append(line)
        ax.set_xlim(0, 50000)
        ax.set_xticks(np.arange(0, 50001, 10000))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(3, 3))
        ax.set_xlabel('Raw simulation steps')
        ax.set_ylabel('Avg episode reward' if metric == 'reward' else 'Avg success rate')
        if metric == 'success':
            ax.set_ylim(0, 1)
        ax.grid(color='#BFBFBF', alpha=.6, lw=.7)
        ax.spines[['top', 'right']].set_visible(False)
        return handles

    figure_files = []

    def save(fig, name):
        for extension in ('png', 'pdf', 'svg'):
            filename = name + '.' + extension
            fig.savefig(destination / filename, dpi=180, bbox_inches='tight')
            figure_files.append(filename)
        plt.close(fig)

    note = 'Episode20 + per-raw-step EMA 0.999; bands describe one training run, not cross-seed uncertainty.'
    for group, scenes in [('candidates', SCENES), ('mechanisms', MECHANISM_SCENES)]:
        all_complete = all(key in data and sources[key]['training_complete'] for scene in scenes for key, *_ in series(group, scene))
        suffix = '' if all_complete else ' | IN PROGRESS'
        title = 'Candidate training' if group == 'candidates' else 'Mechanism controls'
        for scene in scenes:
            fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.8))
            handles = None
            for ax, metric in zip(axes, ('reward', 'success')):
                handles = draw(ax, group, scene, metric)
            fig.suptitle(title + ' | ' + TITLES[scene] + suffix, fontsize=15, y=.995)
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .958), ncol=2, fontsize=9.8, frameon=False)
            fig.text(.5, .01, note, ha='center', fontsize=8.7)
            fig.tight_layout(rect=(0, .045, 1, .855 if group == 'mechanisms' else .89))
            save(fig, group + '_' + scene)
        for metric in ('reward', 'success'):
            nrows = 2 if len(scenes) == 6 else 1
            fig, axes = plt.subplots(nrows, 3, figsize=(14.5, 7.9 if nrows == 2 else 5.2), squeeze=False, sharey=True)
            for ax, scene in zip(axes.flat, scenes):
                handles = draw(ax, group, scene, metric, local_status=False)
                count = sum(key in data for key, *_ in series(group, scene))
                running = sum(key in data and not sources[key]['training_complete'] for key, *_ in series(group, scene))
                status = '' if all_complete else f' ({count}/{len(series(group, scene))} shown; {running} running)'
                ax.set_title(TITLES[scene] + status, fontsize=11)
            fig.suptitle(title + ' | ' + metric + suffix, fontsize=15, y=.995)
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .948), ncol=3, fontsize=9.5, frameon=False)
            fig.text(.5, .01, note + (' Running endpoints: circles; absent series are pending.' if not all_complete else ''), ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .06, 1, .84 if nrows == 1 else .88))
            save(fig, group + '_' + metric)

    lines = ['# 第三阶段训练曲线快照', '',
             '奖励和成功率继续采用用户指定的 episode20＋每原始步 EMA999 画法。每条线是独立配置，不平均不同参数或场景。',
             '完整训练为 50k 原始步；曲线止于最后结束回合，未结束回合不填充。进行中的曲线有 running 标识，尚无数据的变体标为 pending。',
             '阴影是单次训练窗口的描述性波动，不是跨训练种子的置信区间。训练图不能代替闭环评估结果。', '',
             f'已核验 {len(validation)} 组复用数组，最大绝对误差 {max(v["max_absolute_error"] for v in validation):.3g}。',
             '', '| 曲线 | 训练完成 | 完整回合数 | 曲线终点原始步 |', '|---|---|---:|---:|']
    for key, source in sources.items():
        lines.append(f'| {key} | {source["training_complete"]} | {source["complete_episodes"]} | {source["last_complete_raw_step"]} |')
    if missing:
        lines += ['', '待纳入的新训练：' + '、'.join(missing) + '。']
    for group, scenes in [('candidates', SCENES), ('mechanisms', MECHANISM_SCENES)]:
        lines += ['', f'## {group}', '', f'![平均奖励]({group}_reward.png)', '', f'![成功率]({group}_success.png)', '']
        lines += [f'- [{TITLES[scene]} 单场景图]({group}_{scene}.png)' for scene in scenes]
    lines += ['', '来源和计算数组分别见 source_manifest.json、curve_values.json；logs/ 保存此次读取的原始字节快照。']
    (destination / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    artifacts = {p.relative_to(destination).as_posix(): sha(p) for p in destination.rglob('*') if p.is_file()}
    write(receipt, {'artifacts': artifacts, 'figures': len(figure_files) // 3, 'new_training_complete': sum(s['training_complete'] for k, s in sources.items() if k.startswith('g')), 'reference_array_comparisons': len(validation)})
    write(DEST / 'latest.json', {'snapshot': destination.relative_to(ROOT).as_posix()})
    (DEST / 'README.md').write_text('# 第三阶段训练图\n\n最新图表：[快照报告](snapshots/' + destination.name + '/README.md)。\n\n历史快照保留，各次结果可按源哈希追溯。\n', encoding='utf-8')
    return {'snapshot': destination.relative_to(ROOT).as_posix(), 'figures': len(figure_files) // 3, 'new_training_pending': missing}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-running', action='store_true')
    print(json.dumps(build(parser.parse_args().include_running), ensure_ascii=False))
