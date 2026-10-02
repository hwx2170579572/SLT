"""Summarize only verified completed artifacts, retaining every missing cell."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .common import ROOT, OUT, SCENES, MECHANISM_SCENES, read, relative, sha, source_registry, write
from .model import VARIANTS


def risk_summary(result_path):
    result_path = ROOT / result_path
    predictions, targets, episode_seeds, per_episode, excluded = [], [], [], [], []
    for p in sorted(result_path.parent.glob('episode_*.json')):
        record = read(p)
        bank = record['bank']
        bank_path = ROOT / bank['path']
        if sha(bank_path) != bank['sha256']:
            raise ValueError('Risk source bank hash mismatch')
        if bank['risk_outcome_censored']:
            excluded.append(record['record']['seed'])
            continue
        with np.load(bank_path, allow_pickle=False) as data:
            heads = data['_risk_heads']
            pred = np.column_stack([heads, heads.max(axis=1)])
            target = data['_collision_return']
            predictions.append(pred)
            targets.append(target)
            episode_seeds.extend(data['_episode_seed'].tolist())
            per_episode.append(((pred - target[:,None]) ** 2).mean(axis=0))
    if not predictions:
        return {'complete_episodes':0,'censored_episode_seeds':excluded,'mse':None}
    predictions, targets = np.concatenate(predictions), np.concatenate(targets)
    columns = ['head_0','head_1','deployment_max']
    stats = {}
    for index,name in enumerate(columns):
        values = predictions[:,index]
        bins = []
        for b in range(10):
            mask = (values >= b/10) & ((values < (b+1)/10) if b < 9 else values <= 1.)
            if not mask.any():
                continue
            bins.append({'prediction_bin':[b/10,(b+1)/10], 'state_count':int(mask.sum()),
                         'distinct_episodes':len(set(np.asarray(episode_seeds)[mask].tolist())),
                         'mean_prediction':float(values[mask].mean()),
                         'mean_discounted_collision_return':float(targets[mask].mean())})
        stats[name] = {'state_weighted_mse':float(((values-targets)**2).mean()),
                       'episode_weighted_mse':float(np.asarray(per_episode)[:,index].mean()),
                       'mean_prediction':float(values.mean()),'mean_observed_return':float(targets.mean()),
                       'reliability_bins':bins}
    return {'complete_episodes':len(per_episode),'censored_episode_seeds':excluded,
            'states':len(targets),'statistics':stats,
            'target':'discounted collision return per stored decision transition',
            'conditioning':'complete non-timeout episodes only; no population calibration claim',
            'independent_training_seeds':1,'no_safety_guarantee':True}


def analyze():
    sources = source_registry()
    job_results, eval_rows, gradients, risks = [], [], [], {}
    for path in sorted((OUT/'jobs').glob('*.json')):
        receipt = read(path)
        artifact = ROOT / receipt['artifact']
        if sha(artifact) != receipt['artifact_sha256']:
            raise ValueError(f'Completed artifact changed: {artifact}')
        job = receipt['job']
        result = read(artifact)
        job_results.append(receipt)
        if job['kind'] == 'eval':
            if result['identity']['smoke']:
                raise ValueError('Engineering smoke entered scientific results')
            source = sources[job['source']]
            row = {'job':job['id'],'source':job['source'],'method':source['method'],
                   'candidate':source['candidate'],'scenario':source['scenario'],
                   'requested_step':job['step'],'actual_step':result['checkpoint']['raw_steps'],
                   'mode':job['mode'],'decoder':job['decoder'],'artifact':relative(artifact),
                   'origin':result['origin'],**result['summary']}
            eval_rows.append(row)
            if job['mode'] == 'mechanism':
                risks[source['scenario']] = risk_summary(receipt['artifact'])
        elif job['kind'] == 'gradient':
            gradients.append({'source':job['source'],'step':job['step'],'artifact':relative(artifact),
                              'aggregates':result['aggregates'],
                              'actual_isolation':result['actual_isolation']})
    selections = {p.stem:read(p) for p in (OUT/'selection').glob('*.json')}
    macros = []
    configs = [('mst_slt','control'),('v4_8','control'),('v4_13','control'),
               ('v4_8','lr_half'),('v4_13','tau_half')]
    for method,candidate in configs:
        for mode,step in [('legacy',50000),('score',50000),('score','selected'),
                          ('robust_lower','selected'),('robust_higher','selected')]:
            rows = [r for r in eval_rows if (r['method'],r['candidate'],r['mode'],r['requested_step'],r['decoder'])
                    == (method,candidate,mode,step,'native')]
            available = {r['scenario']:r for r in rows}
            if mode.startswith('robust') and candidate == 'control' and method != 'mst_slt':
                continue
            missing = [s for s in SCENES if s not in available]
            metrics = None if missing else {k:float(np.mean([available[s][k] for s in SCENES]))
                                           for k in ('success_rate','collision_rate','timeout_rate','mean_return')}
            macros.append({'method':method,'candidate':candidate,'mode':mode,'step':step,
                           'complete_scenes':len(available),'missing_scenes':missing,'metrics':metrics})
    payload = {'manifest_sha256':sha(OUT/'manifest.json'),'completed_jobs':len(job_results),
               'unique_evaluation_artifacts':len({r['artifact'] for r in eval_rows}),
               'eval_rows':eval_rows,'macro_results':macros,'selections':selections,
               'gradients':gradients,'risk_diagnostics':risks,
               'interpretation_status':'Requires scientific interpretation; no automatic method promotion',
               'multi_training_seed_confirmation_deferred':True,'formal_test_accessed':False}
    directory = OUT/'analysis'; directory.mkdir(exist_ok=True)
    write(directory/'summary.json',payload)
    if eval_rows:
        with (directory/'evaluation_cells.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=list(eval_rows[0]))
            writer.writeheader(); writer.writerows(eval_rows)

    def pct(value):
        return f'{100*value:.2f}%'

    text = ['# 第三阶段机制诊断结果','',
            '本轮暂缓多训练种子确认；效率不作为否决条件，超时仍计为失败。所有结果属于开发证据。',
            f'已核查完成任务 {len(job_results)} 个；独立评估结果文件 {payload["unique_evaluation_artifacts"]} 个。',
            '相同所选/最后模型通过缓存与互斥锁复用；任务条目数不等于新增实验次数。','',
            '## 三方法与参数配置','',
            '| 方法 | 参数 | 评估交通 | 模型 | 场景完成数 | 成功率 | 碰撞率 | 超时率 | 平均回报 |',
            '|---|---|---|---|---:|---:|---:|---:|---:|']
    for row in macros:
        m = row['metrics']
        vals = [pct(m[k]) for k in ('success_rate','collision_rate','timeout_rate')] + [f'{m["mean_return"]:.4f}'] if m else ['待完成']*4
        text.append('| '+' | '.join([row['method'],row['candidate'],row['mode'],str(row['step']),f'{row["complete_scenes"]}/6',*vals])+' |')
    text += ['', '## 固定最后模型的机制消融','',
             '四个因子组合及简单整体调权对照保持 50k、seed 0、τ=0.0025、学习率 1e-4、增强与部署规则一致。完整方法复用已有模型。','',
             '| 场景 | 变体 | 交通 | 成功率 | 碰撞率 | 超时率 |','|---|---|---|---:|---:|---:|']
    for scene in MECHANISM_SCENES:
        for variant in VARIANTS:
            source = f'v4_13__{scene}__tau_half' if variant == 'g1_s025' else f'{variant}__{scene}'
            for mode in ('legacy','score'):
                found = next((r for r in eval_rows if r['source']==source and r['mode']==mode and r['requested_step']==50000 and r['decoder']=='native'),None)
                metrics = [pct(found[k]) for k in ('success_rate','collision_rate','timeout_rate')] if found else ['待完成']*3
                text.append('| '+' | '.join([scene,variant,mode,*metrics])+' |')
    text += ['', '## 解码与训练交通诊断','',
             '| 场景 | 方法/参数 | 交通 | 解码 | 成功率 | 碰撞率 | 超时率 |','|---|---|---|---|---:|---:|---:|']
    for row in eval_rows:
        if row['mode'] == 'legacy_train' or (row['mode']=='legacy' and row['decoder']!='native'):
            text.append('| '+' | '.join([row['scenario'],row['method']+'/'+row['candidate'],row['mode'],row['decoder'],
                                         *[pct(row[k]) for k in ('success_rate','collision_rate','timeout_rate')]])+' |')
    text += ['', '## 解释边界','',
             '- 新验证/评分交通按分别固定的种子生成到达时序；CARLA 保留同一个路线模板，不能解释为未见地图泛化。',
             '- 梯度对照使用同一真实状态/动作诊断库。它不是既有训练过程的原始回放；隔离后的零梯度只能证明路径实现。',
             '- 碰撞网络与折扣碰撞回报比较，时间单位为回放决策转移。超时轨迹单列为删失数据，不把它们当作已知零未来风险。',
             '- 不完整六场景不计算总体均值；不同参数的训练曲线分别绘制；单种子波动不作为多种子置信区间。',
             '- 多训练种子与基线新增调参按用户要求暂缓，当前不能给出最终跨种子优胜结论。','',
             '机器可核查数值见 [summary.json](summary.json)，逐评估单元见 [evaluation_cells.csv](evaluation_cells.csv)。']
    (directory/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return {'completed_jobs':len(job_results),'report':relative(directory/'REPORT.md')}
