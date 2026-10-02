"""Match the original v5 individual/composite visual grammar for both metrics."""
from pathlib import Path
import csv
import json
import runpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/final_5m6s_training_curves_episode20_ema999'
OUT=ROOT/'outputs/final_5m6s_reference_v5_style'
V5=runpy.run_path(str(Path(__file__).with_name('plot_training_curves_v5_episode20_ema999.py')),run_name='style_only')
METHODS=['mst_slt','temporal_graph','full_balanced','v4_13','v4_8']
LABELS={'mst_slt':'MST+SLT','temporal_graph':'TemporalGraph','full_balanced':'Full+BalancedSlots','v4_13':'v4.13','v4_8':'v4.8'}
STYLES=dict(V5['METHOD_STYLES'])
STYLES.update(full_balanced=dict(color='#009E73',linestyle='-.',linewidth=1.45,band_alpha=.08),v4_13=dict(color='#CC79A7',linestyle=(0,(5,1,1,1)),linewidth=1.9,band_alpha=.12))
SCENES=['left_turn','cross','roundabout_easy','roundabout_medium','roundabout','carla']
NAMES=['Left Turn','Cross','Roundabout Easy','Roundabout Medium','Roundabout','CARLA']

def read(path):
    with path.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'visual-contract.md').write_text('''# Visual contract
Match original v5: per-scenario 6.4 x 7.8 inch figure, upper five-method mean curves and descriptive bands, lower signed v4.8 margin against max(MST+SLT, TemporalGraph), yellow panel label, internal legends, 10000-step ticks, original font/grid/background/line weights. Export each metric independently plus six-scene composites.
Success uses unchanged v5 samples. Reward uses the same zero-padded 20-episode mean and population SD/sqrt(20), expanded over observed raw steps and EMA-smoothed with retention .999 and initial zero. Reward bands are not clipped. Bands are descriptive window variability, not cross-seed confidence intervals.
Source: existing verified final 30-cell logs via prior manifest. No training/evaluation execution. Curve time support remains each method's actual observed length; signed comparisons use the original three-method common sampled support.
''',encoding='utf-8')
    manifest=json.loads((SRC/'manifest.json').read_text())
    assert all(V5['sha256'](ROOT/a['monitor'])==a['monitor_sha256'] for a in manifest['sources'])
    data=read(SRC/'training_curve_data.csv')
    curves={(s,m):[r for r in data if r['scenario']==s and r['method']==m] for s in SCENES for m in METHODS}
    for rr in curves.values():
        for r in rr:
            for key in ['time_step','ema_success_rate','ema_band_lower','ema_band_upper','ema_average_reward']:
                r[key]=float(r[key])
    reference=read(ROOT/'outputs/iv2e3m100x100_training_curves_v5_episode20_ema999/training_success_curve_episode20_ema999_data.csv')
    lookup={(r['scenario'],r['method'],int(r['time_step'])):r for r in data}
    max_error=max(abs(float(r[k])-float(lookup[r['scenario'],r['method'],int(r['time_step'])][k])) for r in reference for k in ['ema_success_rate','ema_band_lower','ema_band_upper'])
    assert max_error < 1e-12
    episodes=read(SRC/'training_episode_data.csv')
    for s in SCENES:
        for m in METHODS:
            ep=[r for r in episodes if r['scenario']==s and r['method']==m]
            values=np.array([float(r['episode_reward']) for r in ep])
            _,sd,_=V5['padded_episode_window'](values,20)
            signal=np.repeat(sd/np.sqrt(20),[int(r['raw_simulation_steps']) for r in ep])
            band=np.zeros(len(signal)+1)
            for t,x in enumerate(signal): band[t+1]=.999*band[t]+.001*x
            for r in curves[s,m]:
                se=band[int(r['time_step'])]
                r['reward_band_lower']=r['ema_average_reward']-se
                r['reward_band_upper']=r['ema_average_reward']+se
    def top(ax,s,metric):
        for m in METHODS:
            rr=curves[s,m]; st=STYLES[m]
            key,lo,hi=('ema_success_rate','ema_band_lower','ema_band_upper') if metric=='success' else ('ema_average_reward','reward_band_lower','reward_band_upper')
            x=[r['time_step'] for r in rr]
            ax.fill_between(x,[r[lo] for r in rr],[r[hi] for r in rr],color=st['color'],alpha=st['band_alpha'],linewidth=0,zorder=1)
            ax.plot(x,[r[key] for r in rr],color=st['color'],ls=st['linestyle'],lw=st['linewidth'],label=LABELS[m],zorder=3)
        ax.set_title('Scenario: '+NAMES[SCENES.index(s)],fontsize=10.5,pad=7)
        ax.set_xlim(0,50000); ax.set_xticks(range(0,50001,10000))
        if metric=='success': ax.set_ylim(0,1); ax.set_yticks(np.arange(0,1.01,.2))
        else: ax.set_ylim(-1.2,1.2); ax.set_yticks(np.arange(-1,1.01,.5))
        ax.set_xlabel('Time steps',fontsize=9.2)
        ax.set_ylabel('Avg Success rate' if metric=='success' else 'Avg Reward',fontsize=9.2)
        ax.tick_params(labelsize=8.1)
        ax.grid(True,color='#BFBFBF',linewidth=.65,alpha=.8)
        ax.spines[['top','right']].set_visible(False); ax.set_facecolor('#FCFCFC')
        handles,labels=ax.get_legend_handles_labels()
        order=[labels.index(LABELS[m]) for m in ['v4_8','mst_slt','temporal_graph','full_balanced','v4_13']]
        ax.legend([handles[i] for i in order],[labels[i] for i in order],loc='upper left',fontsize=8,frameon=True,framealpha=.84,borderpad=.4,handlelength=2.2)
    summaries=[]
    def bottom(ax,s,metric):
        key='ema_success_rate' if metric=='success' else 'ema_average_reward'
        maps={m:{r['time_step']:r[key] for r in curves[s,m]} for m in ['v4_8','mst_slt','temporal_graph']}
        common=sorted(set.intersection(*(set(v) for v in maps.values())))
        delta=[maps['v4_8'][t]-max(maps['mst_slt'][t],maps['temporal_graph'][t]) for t in common]
        rows=[dict(time_step=t,v4_8_margin_vs_best_baseline=d) for t,d in zip(common,delta)]
        summary=dict(mean_margin_vs_best_baseline=float(np.mean(delta)),positive_margin_fraction=float(np.mean(np.array(delta)>0)))
        V5['draw_margin_axis'](ax,rows,summary,50000,.6 if metric=='success' else 1.2)
        if metric=='reward': ax.set_ylabel('Reward margin',fontsize=9.2); ax.set_yticks([-1,-.5,0,.5,1])
        return dict(scenario=s,metric=metric,**summary)
    def panel(ax,s):
        ax.text(-.12,1.10,chr(97+SCENES.index(s))+')',transform=ax.transAxes,fontsize=13,fontweight='bold',va='top',ha='left',bbox=dict(facecolor='#FFF200',edgecolor='none',pad=1.4))
    def save(fig,name):
        for ext in ['png','svg','pdf']: fig.savefig(OUT/(name+'.'+ext),dpi=300,facecolor='white',bbox_inches='tight')
        plt.close(fig)
    for metric in ['success','reward']:
        for s in SCENES:
            fig,axs=plt.subplots(2,1,figsize=(6.4,7.8),sharex=True,gridspec_kw=dict(height_ratios=[1.15,.85],hspace=.26))
            fig.subplots_adjust(left=.14,right=.97,top=.88,bottom=.16)
            top(axs[0],s,metric); summaries.append(bottom(axs[1],s,metric)); panel(axs[0],s)
            fig.text(.5,.055,'20-episode zero-padded '+metric+' · EMA α=0.999 per raw time step\nBands: descriptive window SE; seed 0. Margin: v4.8 − max(MST+SLT, TemporalGraph).',ha='center',fontsize=8,color='#555555')
            save(fig,s+'_'+metric+'_v5_style')
        fig,axs=plt.subplots(4,3,figsize=(15,15.5),gridspec_kw=dict(height_ratios=[1.15,.85,1.15,.85],hspace=.48,wspace=.25))
        fig.subplots_adjust(left=.065,right=.995,top=.95,bottom=.075)
        for i,s in enumerate(SCENES):
            row=2*(i//3); col=i%3
            top(axs[row,col],s,metric); bottom(axs[row+1,col],s,metric); panel(axs[row,col],s)
        fig.text(.5,.025,'20-episode zero-padded '+metric+' · EMA α=0.999 per raw time step · seed 0\nBands: descriptive window SE. Bottom: v4.8 − max(MST+SLT, TemporalGraph).',ha='center',fontsize=8.25,color='#555555')
        save(fig,'all_scenarios_'+metric+'_v5_style')
    V5['write_csv'](OUT/'curve_data.csv',[r for rr in curves.values() for r in rr])
    manifest.update(reference_success_max_absolute_error=max_error,style_reference='plot_training_curves_v5_episode20_ema999.py',reward_band='EMA(population SD of zero-padded 20 episode rewards / sqrt(20)); no clipping',plot_script_sha256=V5['sha256'](Path(__file__)),margin_summaries=summaries)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Reference success and bands max absolute error:',max_error)
    print('Output:',OUT)

if __name__=='__main__': main()
