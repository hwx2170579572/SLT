"""Summarize the multi-seed matrix; missing cells stay explicit, never imputed."""
from __future__ import annotations

import csv

import numpy as np

from .common import OUT, ROOT, read, relative, sha, source_registry, write
from .config import METHODS, SCENES, TRAINING_SEEDS

METRICS = ('success_rate', 'collision_rate', 'timeout_rate', 'mean_return')
MODEL_KINDS = (('selected', 'selected'), ('exact_final', '50000'))
HEADLINE_METRICS = ('success_rate', 'collision_rate', 'timeout_rate')


def collect_rows():
    sources = source_registry()
    rows = []
    for path in sorted((OUT / 'jobs').glob('*.json')):
        receipt = read(path)
        artifact = ROOT / receipt['artifact']
        if sha(artifact) != receipt['artifact_sha256']:
            raise ValueError(f'Completed artifact changed: {artifact}')
        job = receipt['job']
        if job['kind'] != 'eval':
            continue
        result = read(artifact)
        if result['identity']['smoke']:
            raise ValueError('Engineering smoke entered scientific results')
        source = sources[job['source']]
        rows.append({'job': job['id'], 'cell': job['source'], 'method': source['method'],
                     'candidate': source['candidate'], 'scenario': source['scenario'],
                     'training_seed': source['training_seed'], 'requested_step': str(job['step']),
                     'actual_step': result['checkpoint']['raw_steps'], 'mode': job['mode'],
                     'decoder': job['decoder'], 'artifact': relative(artifact),
                     'origin': result['origin'], **{k: result['summary'][k] for k in METRICS}})
    return rows


def scored_row(rows, method, scenario, seed, requested_step):
    matches = [r for r in rows if r['method'] == method and r['scenario'] == scenario
               and r['training_seed'] == seed and r['mode'] == 'score'
               and r['requested_step'] == str(requested_step)]
    if len({r['artifact'] for r in matches}) > 1:
        raise ValueError(f'Conflicting scored models: {method}/{scenario}/seed{seed}/{requested_step}')
    return matches[0] if matches else None


def aggregate(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not values.size:
        return None
    return {'n': int(values.size), 'mean': float(values.mean()),
            'std': float(values.std(ddof=1)) if values.size > 1 else None}


def analyze():
    rows = collect_rows()
    sources = source_registry()
    directory = OUT / 'analysis'
    directory.mkdir(parents=True, exist_ok=True)

    per_seed, per_cell = [], []
    for method in METHODS:
        candidate = METHODS[method]['candidate']
        for seed in TRAINING_SEEDS:
            for kind, requested in MODEL_KINDS:
                scenarios = {scene: scored_row(rows, method, scene, seed, requested) for scene in SCENES}
                for scene, row in scenarios.items():
                    per_cell.append({'method': method, 'candidate': candidate, 'scenario': scene,
                                     'training_seed': seed, 'model': kind, 'complete': row is not None,
                                     **({k: (row[k] if row else None) for k in METRICS})})
                entry = {'method': method, 'candidate': candidate, 'training_seed': seed, 'model': kind,
                         'complete_scenes': sum(1 for row in scenarios.values() if row is not None),
                         'missing_scenes': [scene for scene, row in scenarios.items() if row is None]}
                for metric in METRICS:
                    entry[metric] = aggregate([row[metric] if row else None for row in scenarios.values()])
                per_seed.append(entry)

    headline = []
    for method in METHODS:
        for kind, _ in MODEL_KINDS:
            entries = [e for e in per_seed if e['method'] == method and e['model'] == kind]
            item = {'method': method, 'candidate': METHODS[method]['candidate'], 'model': kind,
                    'training_seeds': [e['training_seed'] for e in entries
                                       if all(e[m] is not None for m in HEADLINE_METRICS)]}
            for metric in HEADLINE_METRICS:
                item[metric] = aggregate([e[metric]['mean'] if e[metric] else None for e in entries])
            headline.append(item)

    payload = {'schema': 'phase4-multiseed-summary-v1',
               'manifest_sha256': sha(OUT / 'manifest.json'),
               'completed_evaluations': len(rows),
               'unique_evaluation_artifacts': len({r['artifact'] for r in rows}),
               'evaluation_rows': rows, 'per_cell': per_cell, 'per_seed': per_seed,
               'headline': headline, 'training_seeds': list(TRAINING_SEEDS),
               'selections': {p.stem: read(p) for p in (OUT / 'selection').glob('*.json')},
               'cells_with_three_seeds': sum(
                   1 for method in METHODS for scene in SCENES
                   if all(any(r['method'] == method and r['scenario'] == scene
                              and r['training_seed'] == seed and r['mode'] == 'score'
                              for r in rows) for seed in TRAINING_SEEDS)),
               'interpretation_status': 'Requires scientific interpretation; no automatic method promotion',
               'formal_test_accessed': False}
    write(directory / 'summary.json', payload)
    if rows:
        with (directory / 'evaluation_cells.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def pct(value):
        return '待完成' if value is None else f'{100 * value:.2f}%'

    def spread(entry):
        if entry is None or entry['mean'] is None:
            return '待完成'
        if entry['std'] is None:
            return f'{100 * entry["mean"]:.2f}%'
        return f'{100 * entry["mean"]:.2f}% ± {100 * entry["std"]:.2f}%'

    text = ['# 第四阶段：三方法 × 六场景 × 三训练种子确认', '',
            '三种方法使用统一训练器、统一 50k 预算、统一检查点筛选协议与统一评估种子；跨种子比较时只有模型不同。',
            f'已核查评估条目 {len(rows)} 个，独立评估文件 {payload["unique_evaluation_artifacts"]} 个。',
            'seed-0 的 v4.8 lr_half 与 v4.13 tau_half 直接复用已封存的 phase-2 / phase-3 证据（sha256 绑定），不重训。',
            f'三种子齐全的方法-场景单元：{payload["cells_with_three_seeds"]}/{len(METHODS) * len(SCENES)}。', '',
            '## 跨种子汇总（六场景等权，先场景内平均再在种子上取均值 ± 标准差）', '',
            '| 方法 | 参数 | 模型 | 完整种子数 | 成功率 | 碰撞率 | 超时率 |', '|---|---|---|---:|---:|---:|---:|']
    for item in headline:
        text.append('| ' + ' | '.join([item['method'], item['candidate'], item['model'],
                                       str(len(item['training_seeds'])), spread(item['success_rate']),
                                       spread(item['collision_rate']), spread(item['timeout_rate'])]) + ' |')
    text += ['', '## 逐种子结果', '',
             '| 方法 | 参数 | 种子 | 模型 | 场景完成数 | 缺失场景 | 成功率 | 碰撞率 | 超时率 |',
             '|---|---|---:|---|---:|---|---:|---:|---:|']
    for entry in per_seed:
        text.append('| ' + ' | '.join([entry['method'], entry['candidate'], f'seed{entry["training_seed"]}',
                                       entry['model'], str(entry['complete_scenes']),
                                       ', '.join(entry['missing_scenes']) or '—',
                                       pct(entry['success_rate']['mean']) if entry['success_rate'] else '待完成',
                                       pct(entry['collision_rate']['mean']) if entry['collision_rate'] else '待完成',
                                       pct(entry['timeout_rate']['mean']) if entry['timeout_rate'] else '待完成']) + ' |')
    text += ['', '## 复用的 seed-0 证据', '']
    for name in sorted(payload['selections']):
        source = sources.get(name)
        if source is None or not source.get('reused_evaluations'):
            continue
        text.append(f'- `{name}`：统一筛选选中第 {payload["selections"][name]["selected_step"]} 步检查点；'
                    f'训练与评估复用已封存证据（{len(source["reused_evaluations"])} 项 sha256 绑定）。')
    text += ['', '## 解释边界', '',
             '- 所有结果属于开发证据，`formal_test` 从未被访问。',
             '- 三个训练种子共用同一套评估种子与同一套交通实现；跨种子差异只反映训练随机性。',
             '- 六场景不完整的方法不计算总体均值；缺失单元显式保留为“待完成”，不作插补。',
             '- 复用的 seed-0 证据在读取时逐项校验 sha256，任何漂移都会直接报错而不是静默重算。', '',
             '机器可核查数值见 [summary.json](summary.json)，逐评估单元见 [evaluation_cells.csv](evaluation_cells.csv)。']
    (directory / 'REPORT.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    return {'completed_evaluations': len(rows), 'unique_artifacts': payload['unique_evaluation_artifacts'],
            'cells_with_three_seeds': payload['cells_with_three_seeds'],
            'report': relative(directory / 'REPORT.md')}
