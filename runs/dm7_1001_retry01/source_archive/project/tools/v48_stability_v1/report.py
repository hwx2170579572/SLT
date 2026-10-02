from __future__ import annotations

import csv
import math

from .common import ROOT, OUT, SCENES, CANDIDATES, DEPLOYMENT, read, write, sha, source, cell, train_path, plan, relative
from .curves import read_monitor, analyze_rows, draw_curves
from .jobs import eval_path


def gates(candidate, baseline, rules):
    reasons = []
    c, b = candidate['training'], baseline['training']
    ce, be = candidate.get('evaluation'), baseline.get('evaluation')
    if not ce or not be:
        return dict(eligible=False, reasons=['paired actor-only evaluation incomplete'])
    for key, tolerance, direction in (
        ('success_rate', 'eval_success_tolerance', 1), ('collision_rate', 'eval_collision_tolerance', -1),
        ('timeout_rate', 'eval_timeout_tolerance', -1)):
        if direction * (ce['summary'][key] - be['summary'][key]) < -rules[tolerance] - 1e-12:
            reasons.append('deployment ' + key)
    for metric, tolerance in (('success', 'train_tail_success_tolerance'), ('reward', 'train_tail_reward_tolerance')):
        cv, bv = c[metric]['tail_raw_episodes']['mean'], b[metric]['tail_raw_episodes']['mean']
        if cv is None or bv is None or cv < bv - rules[tolerance] - 1e-12:
            reasons.append('tail raw ' + metric)
    cc, bc = c['success']['curve'], b['success']['curve']
    if cc['mean_auc'] < bc['mean_auc'] - rules['train_success_auc_tolerance']:
        reasons.append('success AUC')
    if cc['max_drawdown_after_5k'] > bc['max_drawdown_after_5k'] + rules['max_drawdown_tolerance']:
        reasons.append('maximum drawdown')
    if cc['excess_total_variation'] > bc['excess_total_variation'] * (1 + rules['max_oscillation_relative_tolerance']):
        reasons.append('oscillation')
    ratio = cc['tail_curve_std'] / max(bc['tail_curve_std'], 1e-12)
    if ratio > 1 - rules['min_tail_std_relative_improvement']:
        reasons.append('tail curve std improvement below 10%')
    return dict(eligible=not reasons, reasons=reasons, tail_std_ratio=ratio,
                deployment_success_delta=ce['summary']['success_rate'] - be['summary']['success_rate'])


def verify_pair(scene, candidate):
    """Compare actual SUMO seeds and route/network bytes for all paired episodes."""
    for index in range(100):
        left = read(eval_path(scene, 'lr_half') / f'e{index:03d}.json')
        right = read(eval_path(scene, candidate) / f'e{index:03d}.json')
        for key in ('seed', 'route_sha256', 'network_sha256'):
            if left['traffic'][key] != right['traffic'][key]:
                raise ValueError(f'Unpaired traffic: {scene}/{candidate}/{index}/{key}')


def build_report():
    directory = OUT / 'analysis'
    directory.mkdir(parents=True, exist_ok=True)
    metrics, curve_data, source_data, states = {}, {}, {}, {}
    for scene in SCENES:
        per_scene = {}
        for candidate in CANDIDATES:
            key = cell(scene, candidate)
            spec = source(scene, candidate)
            log = ROOT / spec['training_log'] if spec['reuse'] else train_path(scene, candidate) / 'train_monitor.csv'
            complete = spec['reuse'] or (train_path(scene, candidate) / 'training_complete.json').exists()
            status = train_path(scene, candidate) / 'status.json'
            states[key] = 'reused' if spec['reuse'] else read(status)['status'] if status.exists() else 'not_started'
            if not complete:
                continue
            expected = spec['training_log_sha256'] if spec['reuse'] else read(train_path(scene, candidate) / 'training_complete.json')['train_monitor_sha256']
            if sha(log) != expected:
                raise ValueError(f'Training log changed: {log}')
            stats, curves = analyze_rows(read_monitor(log))
            value = dict(training=stats, evaluation=None,
                         numerical_divergence='not observed in available aggregates; per-update historical data unavailable' if spec['reuse'] else 'none detected by new checks')
            result_path = eval_path(scene, candidate) / 'result.json'
            if result_path.exists():
                result = read(result_path)
                identity = result['identity']
                if identity['smoke'] or identity['deployment'] != DEPLOYMENT or identity['manifest_sha256'] != sha(OUT / 'manifest.json'):
                    raise ValueError('Evaluation protocol mismatch')
                for name, expected_episode_sha in result['episode_files'].items():
                    if sha(result_path.parent / name) != expected_episode_sha:
                        raise ValueError('Evaluation episode changed')
                value['evaluation'] = result
            metrics[key], per_scene[candidate] = value, curves
            source_data[key] = dict(training_log=relative(log), training_log_sha256=sha(log),
                                    evaluation=relative(result_path) if result_path.exists() else None)
            curve_data[key] = {m: a.tolist() for m, a in curves.items()}
        if per_scene:
            labels = {c: CANDIDATES[c]['label'] for c in per_scene}
            draw_curves(per_scene, labels, directory, scene + '_all', scene.upper() + ' | separate candidate curves')
            for candidate in per_scene:
                if candidate == 'lr_half':
                    continue
                selected = {c: per_scene[c] for c in ('lr_half', candidate)}
                draw_curves(selected, labels, directory, scene + '_' + candidate, scene.upper() + ' | ' + candidate + ' vs reference')
    comparisons, eligible = {}, []
    rules = plan()['gates']
    for candidate in CANDIDATES:
        if candidate == 'lr_half':
            continue
        entries = {}
        for scene in SCENES:
            c, b = metrics.get(cell(scene, candidate)), metrics.get(cell(scene, 'lr_half'))
            if c and b:
                if c['evaluation'] and b['evaluation']:
                    verify_pair(scene, candidate)
                entries[scene] = gates(c, b, rules)
            else:
                entries[scene] = dict(eligible=False, reasons=['training incomplete'])
        comparisons[candidate] = entries
        if all(entry['eligible'] for entry in entries.values()):
            eligible.append(candidate)
    eligible.sort(key=lambda c: max(v['tail_std_ratio'] for v in comparisons[c].values()))
    finished = len(metrics) == 10 and all(v['evaluation'] is not None for v in metrics.values())
    payload = dict(complete=finished, states=states, metrics=metrics, comparisons=comparisons,
                   eligible_in_worst_scene_tail_std_order=eligible,
                   provisional_recommendation=eligible[0] if finished and eligible else None,
                   no_multi_seed_claim=True, deployment=DEPLOYMENT)
    write(directory / 'metrics.json', payload)
    write(directory / 'curve_values.json', curve_data)
    write(directory / 'sources.json', source_data)
    columns = ['scene', 'candidate', 'metric', 'episode_mean', 'episode_variance', 'tail_episode_mean',
               'tail_episode_std', 'EMA_endpoint', 'EMA_tail_std', 'max_drawdown', 'excess_total_variation',
               'success_sustained80_step', 'eval_success', 'eval_collision', 'eval_timeout']
    with (directory / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for key, value in metrics.items():
            scene, candidate = key.split('__')
            for metric in ('success', 'reward'):
                t, e = value['training'][metric], value['evaluation']
                c = t['curve']
                writer.writerow(dict(scene=scene, candidate=candidate, metric=metric,
                    episode_mean=t['raw_episodes']['mean'], episode_variance=t['raw_episodes']['variance'],
                    tail_episode_mean=t['tail_raw_episodes']['mean'], tail_episode_std=t['tail_raw_episodes']['std'],
                    EMA_endpoint=c['endpoint'], EMA_tail_std=c['tail_curve_std'], max_drawdown=c['max_drawdown_after_5k'],
                    excess_total_variation=c['excess_total_variation'],
                    success_sustained80_step=c.get('sustained_80pct_first_raw_step'),
                    eval_success=e['summary']['success_rate'] if e else None,
                    eval_collision=e['summary']['collision_rate'] if e else None,
                    eval_timeout=e['summary']['timeout_rate'] if e else None))
    with (directory / 'driving_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'candidate', 'metric', 'mean', 'std', 'variance', 'valid_episodes', 'delta_mean_vs_lr_half'])
        for key, value in metrics.items():
            if not value['evaluation']:
                continue
            scene, candidate = key.split('__')
            reference = metrics[cell(scene, 'lr_half')]['evaluation']
            for name, v in value['evaluation']['driving'].items():
                if not isinstance(v, dict) or 'mean' not in v:
                    continue
                base = reference['driving'].get(name, {}).get('mean') if reference else None
                delta = v['mean'] - base if v['mean'] is not None and base is not None else None
                writer.writerow([scene, candidate, name, v['mean'], v['std'], v['variance'], v['n'], delta])
    lines = ['# v4.8/lr_half 稳定性有限筛选', '',
             f'状态：{"全部完成" if finished else "尚未全部完成；不作最终候选推荐"}。正式训练只使用 seed 0。',
             '部署统一为 exact-final 50k + deterministic actor；无检查点选择、无 critic/fusion 解码。',
             '方差和阴影只描述单次训练或固定模型的评估回合差异，不代表训练种子间稳定性。',
             '曲线逐场景、逐参数给出；20 回合补零窗口，逐原始步 EMA=0.999×旧值+0.001×当前值。',
             '曲线含当前回合结果的回填，属于回顾性画法；最后未完成回合不绘制。', '',
             '| 场景 | 参数 | 后10k原始成功率 | 成功率EMA终点 | 后10k曲线std | 5k后最大回撤 | actor评估成功率 |',
             '|---|---|---:|---:|---:|---:|---:|']
    def fmt(v):
        return '待完成/不可用' if v is None else f'{v:.4f}'
    for key, value in metrics.items():
        scene, candidate = key.split('__')
        t, e = value['training']['success'], value['evaluation']
        lines.append(f"|{scene}|{candidate}|{fmt(t['tail_raw_episodes']['mean'])}|{fmt(t['curve']['endpoint'])}|{fmt(t['curve']['tail_curve_std'])}|{fmt(t['curve']['max_drawdown_after_5k'])}|{fmt(e['summary']['success_rate'] if e else None)}|")
    lines += ['', '自动比较：先逐场景检查成功率、碰撞、超时和原始训练表现，再判断尾段波动、回撤和振荡。阈值是预先声明的开发筛选规则，不是显著性检验。',
              '性能回撤 ≥0.20 只记为表现退化；数值发散另由 NaN/Inf 检查判断。80% 持续5k的首次达标时间和尾段平台启发式均不能证明渐近收敛。',
              f'两场景均通过筛选的候选：{", ".join(eligible) if eligible else "暂无"}。',
              '即使筛选通过，也只获得 seed 0 下的待确认候选；当前不自动启动多种子实验。', '']
    for candidate, scenes in comparisons.items():
        lines.append('- ' + candidate + '：' + '；'.join(scene + ' / ' + ('通过' if v['eligible'] else ', '.join(v['reasons'])) for scene, v in scenes.items()))
    lines += ['', '完整指标：comparison.csv、driving_metrics.csv、metrics.json；可复现曲线点：curve_values.json。',
              '跟车 TTC 仅衡量当前前车的纵向接近，不能代表交叉冲突/行人风险。没有前车或没有接近时，最小 TTC 保留缺失。',
              'CARLA 使用项目的 SUMO 场景，交通模板共享，结果不是 CARLA 原生模拟器或未接触最终测试集结论。', '']
    for scene in SCENES:
        lines += [f'![{scene}]({scene}_all.png)', '']
    lines += ['任务状态：'] + [f'- {k}: {v}' for k, v in states.items()]
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return payload
