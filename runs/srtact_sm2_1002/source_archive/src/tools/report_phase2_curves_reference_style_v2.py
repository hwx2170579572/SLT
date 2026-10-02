"""Replot current 66 logs with the user's original episode20/EMA999 transform."""
import csv
import runpy
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.phase1_checkpoint_diagnostics import read,write,sha
from tools.report_phase2_systematic_v1 import SCENES,TITLES
DEST=ROOT/'results_phase2_training_curves_v2_episode20_ema999'
REFERENCE=ROOT/'visual-composer/plot_training_curves_v5_episode20_ema999.py'
OLD=runpy.run_path(str(REFERENCE),run_name='reference_only')


def transform(rows,metric):
    if metric=='success':
        values,_=OLD['build_time_step_curve'](rows)
        return np.array([[r['time_step'],r['ema_success_rate'],r['ema_band_lower'],r['ema_band_upper']] for r in values])
    mean,std,_=OLD['padded_episode_window'](np.array([r['r'] for r in rows]),20)
    lengths=np.array([r['raw_simulation_steps'] for r in rows])
    signal=np.repeat(mean,lengths);noise=np.repeat(std/np.sqrt(20),lengths)
    out=np.zeros((len(signal)+1,2))
    for i,(m,se) in enumerate(zip(signal,noise)):
        out[i+1]=.999*out[i]+.001*np.array([m,se])
    t=list(range(0,len(signal)+1,100))
    if t[-1]!=len(signal):t.append(len(signal))
    return np.array([[x,out[x,0],out[x,0]-out[x,1],out[x,0]+out[x,1]] for x in t])


def main():
    DEST.mkdir(exist_ok=True)
    sources=read(ROOT/'results_phase2_systematic_report_v1/source_manifest.json')
    data={}
    for key,s in sources.items():
        p=ROOT/s['training_log']
        if sha(p)!=s['training_log_sha256']:raise ValueError('Log changed')
        with p.open(encoding='utf-8-sig') as f:raw=list(csv.DictReader(x for x in f if not x.startswith('#')))
        rows=[];total=0
        for r in raw:
            total+=int(r['raw_simulation_steps'])
            if total>50000:break
            if r['is_success'].lower() not in ('true','false','1','0'):raise ValueError('Invalid success')
            rows.append(dict(r=float(r['r']),is_success=r['is_success'].lower() in ('true','1'),raw_simulation_steps=int(r['raw_simulation_steps'])))
        data[key]={m:transform(rows,m) for m in ('reward','success')}
    oldpath=ROOT/'outputs/iv2e3m100x100_training_curves_v5_episode20_ema999/training_success_curve_episode20_ema999_data.csv'
    matched=0;max_error=0.
    with oldpath.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            key=f"{r['method']}__{r['scenario']}__control__seed0"
            if key not in data:continue
            a=data[key]['success'];i=np.searchsorted(a[:,0],int(r['time_step']))
            assert i<len(a) and a[i,0]==int(r['time_step'])
            err=max(abs(a[i,j+1]-float(r[n])) for j,n in enumerate(('ema_success_rate','ema_band_lower','ema_band_upper')))
            max_error=max(max_error,err);matched+=1
    assert max_error<1e-12
    write(DEST/'validation.json',dict(reference_points_verified=matched,max_absolute_error=max_error,
        reference_script_sha256=sha(REFERENCE),reference_csv_sha256=sha(oldpath),source_curves=66))
    write(DEST/'source_manifest.json',sources)
    write(DEST/'curve_values.json',{k:{m:a.tolist() for m,a in v.items()} for k,v in data.items()})
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    groups={'defaults': [('mst_slt','control','MST+SLT'),('v4_8','control','v4.8'),('v4_13','control','v4.13')]}
    for m in ('v4_8','v4_13'):groups[m]=[(m,c,label) for c,label in [('control','Default'),('lr_half','LR x 0.5'),('lr_quarter','LR x 0.25'),('tau_half','Tau x 0.5'),('tau_double','Tau x 2')]]
    colors=['#666666','#0072B2','#E69F00','#009E73','#CC79A7'];styles=['--','-',':','-.','-']
    def draw(ax,group,scene,metric):
        for j,(m,c,label) in enumerate(groups[group]):
            a=data[f'{m}__{scene}__{c}__seed0'][metric]
            ax.fill_between(a[:,0],a[:,2],a[:,3],color=colors[j],alpha=.12,linewidth=0)
            ax.plot(a[:,0],a[:,1],color=colors[j],ls=styles[j],lw=2.5 if j==1 else 1.7,label=label)
        ax.set_xlim(0,50000);ax.set_xticks(np.arange(0,50001,10000))
        if metric=='success':ax.set_ylim(0,1)
        ax.grid(color='#BFBFBF',alpha=.7,lw=.7);ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('Raw time steps');ax.set_ylabel('Avg Success rate' if metric=='success' else 'Avg episode reward')
    def save(fig,name):
        for ext in ('png','svg','pdf'):fig.savefig(DEST/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    note='20-episode zero-padded window | EMA old-value weight 0.999 per raw step | shaded bands: descriptive, not cross-seed CI'
    for group in groups:
        for metric in ('reward','success'):
            fig,axs=plt.subplots(2,3,figsize=(14,7.8),sharey=True)
            for ax,scene,title in zip(axs.flat,SCENES,TITLES):draw(ax,group,scene,metric);ax.set_title('Scenario: '+title)
            fig.legend(*axs.flat[0].get_legend_handles_labels(),loc='upper center',ncol=5,bbox_to_anchor=(.5,.965),frameon=False)
            fig.suptitle(group.replace('_','.')+' | '+metric,fontsize=16)
            fig.text(.5,.01,note,ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.90]);save(fig,group+'_'+metric)
        for scene,title in zip(SCENES,TITLES):
            fig,axs=plt.subplots(2,1,figsize=(9,9))
            for ax,metric in zip(axs,('reward','success')):draw(ax,group,scene,metric);ax.legend(loc='best',ncol=3,fontsize=9)
            fig.suptitle(group.replace('_','.')+' | Scenario: '+title,fontsize=16)
            fig.text(.5,.008,'Zero-padded episode20 + per-raw-step EMA 0.999; bands are descriptive.',ha='center',fontsize=9)
            fig.tight_layout(rect=[0,.035,1,.96]);save(fig,group+'_'+scene)
    lines=['# 按参考 episode20 + EMA999 重绘','',
        '本版本保留原始训练数据，仅改统计变换和呈现。66 条曲线完整覆盖默认配置及所有调参候选。',
        f'与参考 CSV 中同源的 MST+SLT/v4.8、CARLA/Cross/Roundabout-C 六条成功率曲线逐点比对：{matched} 个点，均值及上下边界最大绝对误差 {max_error:.3g}。',
        '', '## 上一版差异的原因','',
        '1. 上一版不足20回合不画；参考图固定除以20、缺失槽位补零，因此参考图前段从零逐渐上升。',
        '2. 上一版只在回合结束点连线；参考图将包含当前回合结果的窗口值回填到该回合每个原始步，之后逐步平滑。这是回顾性画法，不是实时可用指标。',
        '3. 上一版无二次EMA；参考图 y[t]=0.999*y[t-1]+0.001*x[t]，初值0，每100步取一个显示点。因此参考图更平滑，峰谷更弱且有滞后。',
        '4. 参考阴影为20槽位总体标准差/sqrt(20)，再以同样EMA平滑；成功率边界截到[0,1]。它不是多训练种子的置信区间。',
        '5. 原图是v4.8、MST+SLT、TemporalGraph三个方法与三个场景；本次按当前任务范围显示MST+SLT、v4.8、v4.13及六场景，调参图另列，不能把不同方法曲线当作同一曲线比较。',
        '6. 原图下半部分是v4.8相对两个基线逐点最大值的差值，不是另一条训练成功率。本次每场景图上半为奖励、下半为成功率，不沿用已排除方法的差值定义。',
        '', '奖励扩展沿用相同补零、总体标准差/sqrt(20)和EMA，保留奖励原尺度，不按[0,1]裁剪。参考图本身只有成功率，因此奖励属于一致规则的新增绘制。',
        '以上变化不改变最终评估表或晋级决定。原始日志和前一版本均保留。','']
    for group in groups:
        lines.extend([f'## {group}','','![奖励]('+group+'_reward.png)','','![成功率]('+group+'_success.png)',''])
        for s,t in zip(SCENES,TITLES):lines.append(f'- [{t} 单场景 PNG]({group}_{s}.png)')
    (DEST/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'24 figures in PNG/SVG/PDF; {matched} reference points verified; maximum error {max_error}')


if __name__=='__main__':main()
