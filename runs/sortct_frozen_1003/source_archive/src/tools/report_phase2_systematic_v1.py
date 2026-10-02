"""Complete result tables and traceable training curves from existing evidence."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.phase1_checkpoint_diagnostics import read,write,sha
from tools.analyze_phase2_reuse_v3 import main as validate,OUT,CONTRACT

DEST=ROOT/'results_phase2_systematic_report_v1'
SCENES=['left_turn','cross','roundabout_easy','roundabout_medium','roundabout','carla']
TITLES=['Left Turn','Cross','Roundabout-A','Roundabout-B','Roundabout-C','CARLA']
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
STYLES=['-','--','-.',':','-']


def monitor(path):
    with path.open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(line for line in f if not line.startswith('#')))
    steps=np.cumsum([int(r['raw_simulation_steps']) for r in rows])
    rewards=np.array([float(r['r']) for r in rows])
    flags=[r['is_success'].lower() for r in rows]
    if any(x not in ('true','false','1','0') for x in flags):raise ValueError('Invalid success flag')
    successes=np.array([float(x in ('true','1')) for x in flags])
    keep=steps<=50000;steps,rewards,successes=steps[keep],rewards[keep],successes[keep]
    if len(steps)<20 or not np.all(np.diff(steps)>0):raise ValueError('Insufficient or invalid training log')
    smooth=lambda x:np.convolve(x,np.ones(20)/20,mode='valid')
    return dict(steps=steps[19:],reward=smooth(rewards),success=smooth(successes)*100,
                episodes=len(steps),last_complete_raw_step=int(steps[-1]))


def save(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(DEST/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def grid(curves,method,metric):
    variants=['control','lr_half','lr_quarter','tau_half','tau_double']
    series=[('mst_slt','control'),('v4_8','control'),('v4_13','control')] if method=='defaults' else [(method,c) for c in variants]
    labels=['MST+SLT','v4.8','v4.13'] if method=='defaults' else ['Default','LR x 0.5','LR x 0.25','Tau x 0.5','Tau x 2']
    fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    for i,(scene,title,ax) in enumerate(zip(SCENES,TITLES,axs.flat)):
        for j,((m,c),label) in enumerate(zip(series,labels)):
            d=curves[f'{m}__{scene}__{c}__seed0']
            ax.plot(d['steps']/1000,d[metric],color=COLORS[j],ls=STYLES[j],lw=1.5,label=label)
        ax.set_title(f'({chr(97+i)}) {title}',loc='left',fontsize=11)
        ax.set_xlim(0,50);ax.grid(alpha=.2)
        if metric=='success':ax.set_ylim(-3,103)
        if i>=3:ax.set_xlabel('Raw training steps (thousands)')
        if i%3==0:ax.set_ylabel('Training success rate (%)' if metric=='success' else 'Mean episode reward')
    fig.suptitle(('Default configurations' if method=='defaults' else method.replace('_','.'))+' | '+('Training success' if metric=='success' else 'Training reward'),fontsize=16,y=.995)
    fig.legend(*axs.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.952),ncol=len(labels),frameon=False)
    fig.text(.5,.005,'Training seed 0 | trailing 20 completed episodes | no cross-seed confidence interval | historical default logs reused',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.90]);save(fig,f'{method}_{metric}')


def main():
    DEST.mkdir(exist_ok=True)
    write(DEST/'visual_contract.json',dict(question='How do observed training reward and success evolve, and which frozen candidates pass all evaluation guards?',
        panels='Six scenarios per figure; default comparison plus every candidate for v4.8 and v4.13',
        window='Trailing 20 complete training episodes; first 19 omitted; incomplete final episode omitted',
        x='Cumulative raw simulation steps capped at 50000',uncertainty='One training seed; no inferential band',
        palette='Okabe-Ito colors with redundant line styles',selection='Show all candidates; no selective curve omission'))
    validate();state=read(OUT/'analysis/completion.json')
    if not state['complete']:raise ValueError('Complete screen required')
    sources=read(OUT/'analysis/evidence_sources.json')['sources']
    controls={r['id']:r for r in read(OUT/'execution_manifest.json')['historical_controls']}
    curves={};trace={};cells={};flat=[]
    for key,s in sources.items():
        p=ROOT/s['path']
        if sha(p)!=s['sha256']:raise ValueError('Changed evidence')
        d=read(p);cells[key]=d
        log=(ROOT/controls[key]['checkpoint']).parent/'train_monitor.csv' if key in controls else p.parent/'train_monitor.csv'
        curves[key]=monitor(log)
        trace[key]=dict(result=s,training_log=str(log.relative_to(ROOT)),training_log_sha256=sha(log),
                        complete_episodes=curves[key]['episodes'],last_complete_raw_step=curves[key]['last_complete_raw_step'])
        method,scene,candidate,seed=key.split('__')
        flat.append(dict(method=method,scenario=scene,candidate=candidate,seed=0,**d['summary']))
    write(DEST/'source_manifest.json',trace)
    write(DEST/'training_curve_values.json',{k:{n:v.tolist() if isinstance(v,np.ndarray) else v for n,v in d.items()} for k,d in curves.items()})
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    for method in ('defaults','v4_8','v4_13'):
        for metric in ('reward','success'):grid(curves,method,metric)
    decision=read(OUT/'analysis/screen_decision.json')
    lines=['# 当前系统性实验汇总','',
        '## 覆盖与结论','',
        '完整筛选 66/66：MST+SLT 默认配置 6 单元，v4.8 与 v4.13 各 5 配置 × 6 场景。每单元训练 seed 0、50k 原始步、100 个配对确定性评估回合（评估 seeds 10000–10099）。18 个历史默认结果引用复用，48 个变体结果有效；共 6600 个逻辑评估回合。四个中断单元经用户授权重跑，原数据已归档。',
        '所有 8 个调参候选均未通过事先固定的全部晋级约束，因此两方法暂保留默认参数。平均成功率提高不等于通过安全、逐场景与效率约束。',
        'Full+BalancedSlots 与 TemporalGraph 不纳入本轮汇总。当前为开发证据，尚未完成独立验证选择、多训练种子确认及投稿主实验。','',
        '## 六场景等权平均：最终模型评估','',
        '| 方法 | 配置 | 平均奖励 | 成功率 % | 碰撞率 % | 超时率 % | 晋级 |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for r in decision['rows']:
        m,c=r['method'],r['candidate'];data=[cells[f'{m}__{s}__{c}__seed0'] for s in SCENES]
        reward=np.mean([x['summary']['mean_return'] for x in data]);g=r['guards']
        lines.append(f"| {m} | {c} | {reward:.4f} | {100*r['metrics']['success_rate']:.2f} | {100*r['metrics']['collision_rate']:.2f} | {100*r['metrics']['timeout_rate']:.2f} | {'默认保留' if g is None else ('通过' if g['passed'] else '未通过')} |")
    lines+=['','## 各场景成功率（%）','','| 方法 / 配置 | '+' | '.join(TITLES)+' |','| --- | '+' | '.join(['---:']*6)+' |']
    for r in decision['rows']:
        m,c=r['method'],r['candidate'];values=[100*cells[f'{m}__{s}__{c}__seed0']['summary']['success_rate'] for s in SCENES]
        lines.append('| '+m+' / '+c+' | '+' | '.join(f'{v:.0f}' for v in values)+' |')
    lines+=['','## 晋级约束与诊断','','约束：六场景平均成功率严格提高，平均碰撞及超时不增加；每场景成功率最多下降 3 个百分点、碰撞率最多上升 3 个百分点；共同成功至少 20 回合，配对完成时间增幅不超过 10%。比较对象为各自默认控制。','']
    for r in decision['rows']:
        if r['guards']:
            reasons=[x for x in r['guards']['reasons']]
            for i,s in enumerate(TITLES):reasons=[x.replace(f'scene_{i}_',s+': ') for x in reasons]
            lines.append('- '+r['method']+' / '+r['candidate']+'：'+', '.join(reasons))
    lines+=['','v4.13 tau_half 平均成功率最高（94.17%），但 CARLA 共同成功时间增加 15.93%；v4.8 lr_half 为 93.50%，但 Roundabout-A 成功率退步 6 个百分点，并有完成时间代价。现有终局日志不足以证明等待、低速或风险约束是原因，不能直接据此改结构。',
        '', '## 训练曲线与读图方法','',
        '纵轴分别是最近 20 个完整训练回合的平均 episode reward 与成功比例；横轴由 raw_simulation_steps 累积得到，最多 50k。起始不足 20 回合不画，未结束回合不计入，末端可能略早于 50k。训练包含探索动作，不等于确定性最终评估；奖励尺度保持原值，未归一化。窗口是描述性平滑，不代表多训练种子平均。无跨种子置信区间。',
        '三方法默认曲线来自各自历史控制的原始日志；两方法调参图完整包含默认和四个候选。历史训练代码等价性未完全证实，曲线差异不能单独支持结构因果收益或训练稳定性结论。','']
    for m in ('defaults','v4_8','v4_13'):
        lines += [f'### {m}','','![平均奖励]('+m+'_reward.png)','','![训练成功率]('+m+'_success.png)','']
    lines+=['## 66 单元完整指标','','| 方法 | 场景 | 配置 | 平均奖励 | 成功率 % | 碰撞率 % | 超时率 % |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in flat:
        lines.append(f"| {r['method']} | {r['scenario']} | {r['candidate']} | {r['mean_return']:.4f} | {100*r['success_rate']:.0f} | {100*r['collision_rate']:.0f} | {100*r['timeout_rate']:.0f} |")
    lines+=['','## 后续工作与边界','','已生成 36 个开发确认候选单元（三方法 × 六场景 × 训练 seeds 1/2），尚未启动。扫描 790 份参数记录发现 15 个潜在复用运行；12 个基线已有匹配 50k 检查点，3 个 v4.8 有匹配最终模型，参数及预算检查通过，环境/源码与评估协议复用审计仍需完成。其余 21 个暂未找到匹配记录。',
        '当前不能宣称达到投稿要求：还缺多训练种子稳定性、公平独立模型选择、机制消融、效率分解和额外交通鲁棒性证据。当前决定是结束这一轮有限搜索、保留默认配置，并用后续机制证据决定是否做单点结构迭代。','',
        'source_manifest.json 保存每条曲线和评估的路径及哈希；training_curve_values.json 保存图中全部平滑坐标；all_cells.csv 为完整数值表。图同时提供 PNG、PDF、SVG。']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fields=['method','scenario','candidate','seed','mean_return','success_rate','collision_rate','timeout_rate','episodes']
    with (DEST/'all_cells.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(flat)
    print('Created report, 66-cell table, 6 figures in PNG/PDF/SVG, and source hashes')


if __name__=='__main__':main()
