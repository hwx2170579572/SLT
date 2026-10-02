"""One failed configuration and one scenario per figure; never combine parameters."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.phase1_checkpoint_diagnostics import read,write,sha


def main():
    source=ROOT/'results_phase2_training_curves_v2_episode20_ema999'
    dest=ROOT/'results_phase2_failed_parameters_by_scenario_v1'
    dest.mkdir(exist_ok=True)
    data=read(source/'curve_values.json');sources=read(source/'source_manifest.json')
    decision=read(ROOT/'results_phase2_reuse_v3/analysis/screen_decision.json')
    failed=[r for r in decision['rows'] if r['guards'] is not None and not r['guards']['passed']]
    scenes=['left_turn','cross','roundabout_easy','roundabout_medium','roundabout','carla']
    titles=['Left Turn','Cross','Roundabout-A','Roundabout-B','Roundabout-C','CARLA']
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    lines=['# 未通过参数：逐场景独立曲线','','每个链接对应一个方法、一个参数配置和一个场景；上图为训练平均奖励，下图为训练成功率。各图不叠加其他参数，也不跨参数平均。',
           '仅沿用已确认的画法：20回合补零窗口、逐原始步 EMA（旧值权重0.999）；阴影是该单次训练的描述性窗口波动，不是参数间或训练种子间的区间。',
           '“未通过”指该配置未通过六场景整体晋级规则，不代表它在每个场景都失败。全部八个未通过候选的六场景均完整展示。','']
    ledger=[]
    for scene,title in zip(scenes,titles):
        folder=dest/scene;folder.mkdir(exist_ok=True)
        lines.extend([f'## {title}','','| 方法 | 参数配置 | 独立 PNG | PDF |','| --- | --- | --- | --- |'])
        for r in failed:
            method,candidate=r['method'],r['candidate']
            key=f'{method}__{scene}__{candidate}__seed0'
            trace=sources[key]
            if sha(ROOT/trace['training_log'])!=trace['training_log_sha256']:raise ValueError('Training log changed')
            fig,axs=plt.subplots(2,1,figsize=(9,8.5),sharex=True)
            for ax,metric in zip(axs,('reward','success')):
                a=np.asarray(data[key][metric],dtype=float)
                ax.plot(a[:,0],a[:,1],color='#0072B2',lw=2.3)
                ax.fill_between(a[:,0],a[:,2],a[:,3],color='#0072B2',alpha=.16,linewidth=0)
                ax.set_ylabel('Avg episode reward' if metric=='reward' else 'Avg success rate')
                ax.set_title('Training reward' if metric=='reward' else 'Training success',loc='left',fontsize=11)
                ax.set_xlim(0,50000);ax.set_xticks(np.arange(0,50001,10000));ax.grid(alpha=.4)
                if metric=='success':ax.set_ylim(0,1)
                ax.spines[['top','right']].set_visible(False)
            axs[1].set_xlabel('Raw training steps')
            fig.suptitle(f'{title} | {method.replace("_", ".")} | {candidate} | seed 0',fontsize=15)
            fig.text(.5,.012,'Single configuration only | zero-padded episode20 + EMA 0.999 per raw step\nBand: descriptive within-run variation; no averaging across configurations',ha='center',fontsize=9)
            fig.tight_layout(rect=[0,.065,1,.95])
            name=f'{method}__{candidate}'
            for ext in ('png','pdf','svg'):fig.savefig(folder/f'{name}.{ext}',dpi=160,bbox_inches='tight')
            plt.close(fig)
            lines.append(f'| {method} | {candidate} | [查看]({scene}/{name}.png) | [下载]({scene}/{name}.pdf) |')
            ledger.append(dict(cell=key,source=trace,figure=f'{scene}/{name}.png',parameters_combined=1))
    assert len(ledger)==48 and len({r['cell'] for r in ledger})==48
    (dest/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(dest/'manifest.json',dict(figures=ledger,source_curve_sha256=sha(source/'curve_values.json'),
        decision_sha256=sha(ROOT/'results_phase2_reuse_v3/analysis/screen_decision.json'),
        figures_count=48,method='Each curve copied directly from its own previously verified cell; no cross-parameter calculation'))
    print('48 independent cell figures exported in PNG/PDF/SVG, indexed by six scenarios')


if __name__=='__main__':main()
