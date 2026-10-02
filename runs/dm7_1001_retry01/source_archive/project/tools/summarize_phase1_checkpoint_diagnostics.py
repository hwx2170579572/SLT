"""Summarize only real completed phase-1 evaluations; mark partial output explicitly."""
from pathlib import Path
import csv
import json
import numpy as np
from phase1_checkpoint_diagnostics import ROOT, OUT, read, write, validate_report, sha

METHODS=['mst_slt','temporal_graph','full_balanced','v4_8','v4_13']
LABELS=['MST+SLT','TemporalGraph','Full+BalancedSlots','v4.8','v4.13']
SCENES=['left_turn','cross','roundabout_easy','roundabout_medium','roundabout','carla']
NAMES=['Left Turn','Cross','Roundabout-A','Roundabout-B','Roundabout-C','CARLA']
METRICS=['success_rate','collision_rate','off_route_rate','timeout_rate','mean_return']
def csvout(p,rows):
    if not rows:return
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    plan=read(OUT/'plan.json'); dest=OUT/'summary';dest.mkdir(exist_ok=True)
    resolved={};missing=[];episode_rows=[];table=[]
    for j in plan['jobs']:
        p=OUT/'jobs'/j['id']/'result.json'
        if not p.exists():missing.append(j['id']);continue
        d=read(p);validate_report(d,100,10000)
        assert d['checkpoint_sha256']==j['checkpoint_sha256']
        assert sha(ROOT/j['checkpoint'])==j['checkpoint_sha256']
        resolved[j['id']]=d
        episode_rows.extend(dict(method=j['method'],scenario=j['scenario'],kind=j['kind'],**r) for r in d['episode_records'])
    for a in plan['aliases']:
        if a['target'] in resolved:
            resolved[a['job']]=dict(resolved[a['target']],kind=a['kind'],checkpoint=a['checkpoint'],checkpoint_sha256=a['checkpoint_sha256'],alias_of=a['target'])
    for jobid,d in resolved.items():
        table.append(dict(method=d['method'],scenario=d['scenario'],kind=d['kind'],raw_steps=d['raw_steps'],decoder=d['decoder'],origin=d['origin'],**{k:d['summary'][k] for k in METRICS},mean_success_completion_time_seconds=d['mean_success_completion_time_seconds'],checkpoint=d['checkpoint']))
    comparison=[]
    for m in METHODS:
        for s in SCENES:
            jobid=f'{m}__{s}__exact_final'
            if jobid not in resolved:continue
            d=resolved[jobid];old=read(ROOT/d['original_evaluation'])
            comparison.append(dict(method=m,scenario=s,original_checkpoint=old.get('selected_checkpoint_kind','exact_final'),decoder=d['decoder'],**{f'original_{k}':old['summary'][k] for k in METRICS},**{f'final_{k}':d['summary'][k] for k in METRICS},success_delta_final_minus_original=d['summary']['success_rate']-old['summary']['success_rate'],original_success_time=old.get('mean_success_completion_time_seconds'),final_success_time=d['mean_success_completion_time_seconds']))
    macro=[]
    for m in METHODS:
        rr=[r for r in comparison if r['method']==m]
        if len(rr)==6:
            macro.append(dict(method=m,**{f'{v}_{k}':float(np.mean([r[f'{v}_{k}'] for r in rr])) for v in ['original','final'] for k in METRICS}))
    csvout(dest/'checkpoint_metrics.csv',table);csvout(dest/'episode_records.csv',episode_rows)
    csvout(dest/'original_vs_exact_final.csv',comparison);csvout(dest/'macro_original_vs_exact_final.csv',macro)
    state=dict(complete=not missing,exact_final_complete=len(comparison)==30,completed_unique_models=len(resolved)-len([a for a in plan['aliases'] if a['job'] in resolved]),expected_unique_models=len(plan['jobs']),missing=missing)
    write(dest/'completion.json',state)
    lines=['# 第一阶段诊断报告','',f"状态：{'完整' if not missing else '检查点曲线未完成，当前仅报告已完成部分'}。",f"统一最后模型对照：{len(comparison)}/30；不同模型评估：{state['completed_unique_models']}/{len(plan['jobs'])}。",'', '仅使用原 evaluation 分区的开发诊断；训练 seed 0，不代表跨种子稳定性。','固定各单元原部署解码器，历史曲线不重新校准，不作为在线模型选择或样本效率证据。','', '| 方法 | 原流程成功率 | 最后模型成功率 | 原流程碰撞率 | 最后模型碰撞率 |','|---|---:|---:|---:|---:|']
    for r in macro:
        lines.append(f"| {LABELS[METHODS.index(r['method'])]} | {r['original_success_rate']:.2%} | {r['final_success_rate']:.2%} | {r['original_collision_rate']:.2%} | {r['final_collision_rate']:.2%} |")
    lines+=['','## 选择影响（最后模型减原流程）','']
    for r in comparison:
        if r['original_checkpoint']!='exact_final':lines.append(f"- {r['method']} / {r['scenario']}: 成功率差 {100*r['success_delta_final_minus_original']:+.1f} 个百分点；原检查点 {r['original_checkpoint']}。")
    if len(macro)==5:
        baseline=next(r for r in macro if r['method']=='mst_slt')
        lines+=['','## 公平最后模型对照','']
        for m in ['v4_8','v4_13']:
            r=next(r for r in macro if r['method']==m);delta=100*(r['final_success_rate']-baseline['final_success_rate'])
            lines.append(f"- {m} 相比 MST+SLT，最后模型六场景等权成功率差 {delta:+.2f} 个百分点。此为单训练种子描述性结果。")
    if not missing:
        lines+=['','## 检查点变化（50k 减 10k；不进行最优检查点选择）','', '| 场景 | 方法 | 10k 成功率 | 50k 成功率 | 变化（百分点） |','|---|---|---:|---:|---:|']
        for s in SCENES:
            for m in ['mst_slt','v4_8','v4_13']:
                a=resolved[f'{m}__{s}__raw_10000']['summary']['success_rate'];b=resolved[f'{m}__{s}__raw_50000']['summary']['success_rate']
                lines.append(f'| {NAMES[SCENES.index(s)]} | {LABELS[METHODS.index(m)]} | {a:.0%} | {b:.0%} | {100*(b-a):+.1f} |')
    lines+=['','## 曲线解释边界','', '- 原始训练日志与固定模型的评估曲线口径不同，不强求排名一致。','- 五个离散检查点直接连线，不插入虚构结果，不进行 EMA 平滑，不按评估结果选择最佳检查点。','- 成功完成时间仅条件于成功回合，不能独立解释整体效率。','- 后续调参或结构迭代仍需依据失败分析和多训练种子确认。']
    (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    if missing:
        print(f'Partial: missing {len(missing)} models');return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#666666','#E69F00','#009E73','#0072B2','#CC79A7'];styles=['--',':','-.','-',(0,(5,1,1,1))]
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    def draw_axis(ax,s,metric,label):
        for i,m in enumerate(METHODS):
            rr=sorted([r for r in table if r['method']==m and r['scenario']==s and r['kind'].startswith('raw_')],key=lambda r:r['raw_steps'])
            assert len(rr)==5
            ax.plot([r['raw_steps'] for r in rr],[r[metric] for r in rr],color=colors[i],ls=styles[i],lw=2.5 if m=='v4_8' else 1.6,marker='o',ms=4,label=LABELS[i])
        ax.set_title('Scenario: '+NAMES[SCENES.index(s)]);ax.set_xlabel('Raw training steps');ax.set_ylabel(label)
        ax.set_xticks([10000,20000,30000,40000,50000]);ax.tick_params(labelsize=8);ax.grid(alpha=.35);ax.set_facecolor('#FCFCFC')
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylim((0,1.02) if metric=='success_rate' else (-1.05,1.05))
    for metric,label in [('success_rate','Evaluation success rate'),('mean_return','Average evaluation return')]:
        fig,axs=plt.subplots(2,3,figsize=(15,8.2));fig.subplots_adjust(top=.9,bottom=.12,hspace=.38,wspace=.24)
        for ax,s in zip(axs.flat,SCENES):
            draw_axis(ax,s,metric,label)
        handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False)
        fig.text(.5,.025,'100 paired episodes per checkpoint; training seed 0; fixed original deployment decoder\nDevelopment diagnostics only; no smoothing or cross-training-seed confidence bands.',ha='center',fontsize=9)
        for ext in ['png','svg','pdf']:fig.savefig(dest/f'checkpoint_{metric}.{ext}',dpi=240,bbox_inches='tight')
        plt.close(fig)
    for s in SCENES:
        fig,axs=plt.subplots(1,2,figsize=(12,4.8))
        draw_axis(axs[0],s,'mean_return','Average evaluation return');draw_axis(axs[1],s,'success_rate','Evaluation success rate')
        handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False)
        fig.text(.5,.025,'100 paired episodes per checkpoint; seed 0; fixed original decoder; development diagnostics',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.055,1,.9))
        for ext in ['png','svg','pdf']:fig.savefig(dest/f'{s}_reward_success.{ext}',dpi=240,bbox_inches='tight')
        plt.close(fig)
    print('Complete:',dest)
if __name__=='__main__':main()
