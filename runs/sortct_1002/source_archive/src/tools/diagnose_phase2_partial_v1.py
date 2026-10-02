"""Describe validated screen evidence without selecting from an incomplete search."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.phase1_checkpoint_diagnostics import read, write, sha
from tools.analyze_phase2_reuse_v3 import main as validate, OUT, CONTRACT
from tools.analyze_phase2_screen_v2 import candidate_guards, mean_metric


def describe(contract, sources):
    data = {}
    for key, source in sources.items():
        path = ROOT / source['path']
        if sha(path) != source['sha256']:
            raise ValueError('Evidence changed: ' + key)
        data[key] = read(path)
    rows = []
    for method in contract['methods']:
        for candidate in contract['candidates']:
            key = candidate['id']
            if method == 'mst_slt' and key != 'control':
                continue
            ids = [f'{method}__{s}__{key}__seed0' for s in contract['scenarios']]
            missing = [k for k in ids if k not in data]
            scene_rows = []
            for scene, cell in zip(contract['scenarios'], ids):
                if cell not in data:
                    continue
                d = data[cell]
                control = data[f'{method}__{scene}__control__seed0']
                scene_rows.append(dict(scenario=scene, metrics=d['summary'],
                    success_delta=d['summary']['success_rate']-control['summary']['success_rate'],
                    collision_delta=d['summary']['collision_rate']-control['summary']['collision_rate']))
            row = dict(method=method, candidate=key, missing=missing, scenes=scene_rows,
                       metrics=None, guards=None)
            if not missing:
                cells = [data[k] for k in ids]
                row['metrics'] = {f: mean_metric(cells, f) for f in ('success_rate', 'collision_rate', 'timeout_rate')}
                if key != 'control':
                    controls = [data[f'{method}__{s}__control__seed0'] for s in contract['scenarios']]
                    row['guards'] = candidate_guards(controls, cells, contract['promotion_guards_relative_to_own_control'])
            rows.append(row)
    return rows


def main():
    validate()
    state = read(OUT/'analysis/completion.json')
    if state['invalid']:
        raise ValueError('Invalid evidence must be resolved before diagnosis')
    evidence = read(OUT/'analysis/evidence_sources.json')
    contract = read(CONTRACT)
    rows = describe(contract, evidence['sources'])
    dest = OUT/'diagnosis_v1'
    dest.mkdir(exist_ok=True)
    write(dest/'diagnosis.json', dict(completion=state, rows=rows, selected_for_confirmation=None,
        development_only=True, causal_attribution_allowed=False,
        contract_sha256=sha(CONTRACT), sources=evidence))
    lines = ['# 第二阶段：有限调参诊断', '',
        f"已验证 {state['resolved']}/{state['expected']} 单元。训练种子仅为 0；每单元 100 个配对评估回合。",
        '历史控制代码等价性未完全证实，以下差异仅作开发诊断，不能证明参数或结构的因果收益。',
        '缺失场景不插值，不用部分场景平均替代六场景指标；本报告不生成候选选择。', '',
        '| 方法 | 候选 | 完整场景 | 成功率 % | 碰撞率 % | 超时率 % | 既定约束 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        m = r['metrics']
        vals = [f"{100*m[k]:.2f}" for k in ('success_rate','collision_rate','timeout_rate')] if m else ['缺失']*3
        guard = r['guards']
        result = ('通过' if guard['passed'] else '未通过：'+', '.join(guard['reasons'])) if guard else ('控制' if m else '待完整')
        lines.append('| '+' | '.join([r['method'],r['candidate'],str(len(r['scenes']))+'/6',*vals,result])+' |')
    lines += ['', '## 逐场景变化（相对各自历史控制）', '', '| 方法 | 候选 | 场景 | 成功率变化 pp | 碰撞率变化 pp |', '| --- | --- | --- | --- | --- |']
    for r in rows:
        if r['candidate'] == 'control':
            continue
        for s in r['scenes']:
            lines.append(f"| {r['method']} | {r['candidate']} | {s['scenario']} | {100*s['success_delta']:+.2f} | {100*s['collision_delta']:+.2f} |")
    lines += ['', '## 缺失单元', ''] + ['- '+x for x in state['missing']]
    lines += ['', '完整候选的约束失败可以定位退步场景，但不能排除尚未完成的候选。缺失处理须单独记录，不能静默改动既定筛选协议。',
        '100 个评估回合不等于 100 个训练种子；后续仍需要多训练种子确认、独立选择协议、机制消融及效率和鲁棒性证据。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines[:21]))


if __name__ == '__main__':
    main()
