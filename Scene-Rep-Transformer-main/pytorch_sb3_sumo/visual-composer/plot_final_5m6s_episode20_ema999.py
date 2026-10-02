"""Read-only replot of the final 30 cells; reuses the v5 numerical transform."""
from pathlib import Path
import csv
import json
import runpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/final_5m6s_training_curves_episode20_ema999'
SOURCE = ROOT / 'results_iv2_5m6s100e_v1/comparison/summary/method_scenario.csv'
PROTOCOL = ROOT / 'experiments/independent_v2_five_methods_six_scenarios_100ep_v1/protocol.json'
V5 = Path(__file__).with_name('plot_training_curves_v5_episode20_ema999.py')
v5 = runpy.run_path(str(V5), run_name='numerical_transform_only')
METHODS = ['mst_slt', 'temporal_graph', 'full_balanced', 'v4_8', 'v4_13']
LABELS = ['MST+SLT', 'TemporalGraph', 'Full+BalancedSlots', 'v4.8', 'v4.13']
COLORS = ['#666666', '#E69F00', '#009E73', '#0072B2', '#CC79A7']
STYLES = ['--', ':', '-.', '-', (0, (5, 1, 1, 1))]
SCENES = ['left_turn', 'cross', 'roundabout_easy', 'roundabout_medium', 'roundabout', 'carla']
NAMES = ['Left Turn', 'Cross', 'Roundabout Easy', 'Roundabout Medium', 'Roundabout', 'CARLA']

def dump_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'visual-contract.md').write_text('''# Visual contract
Artifact: final five-method, six-scenario training comparison; PNG, SVG, PDF.
Question: how do observed training success trajectories compare at equal raw simulation steps?
Source: final method_scenario.csv resolves each training log, including 12 adopted parent cells.
Statistics: exact v5 padded 20-episode mean, held over each observed episode duration; EMA retention 0.999 per raw step, initial zero. This is retrospective episode-to-step expansion.
Panel map: six success panels (2 x 3); companion six signed-margin panels; combined view places margins below each group of three success panels.
Margins: v4.8 and v4.13 minus max(MST+SLT, TemporalGraph), on shared sampled support only. Full+BalancedSlots remains in the success panels.
Uncertainty: no cross-seed confidence intervals; each cell is training seed 0. Curves are training data, not the 3000 evaluation episodes.
Traceability: source paths and hashes, full training episodes, sampled curve CSV and margin CSV retained. Display sampling is 100 raw steps, after per-step EMA.
Style: stable Okabe-Ito/neutral colors with distinct line styles; full-width or appendix use.
''', encoding='utf-8')
    cells = list(csv.DictReader(SOURCE.open(encoding='utf-8-sig')))
    assert len(cells) == 30
    assert {(c['method'], c['scenario']) for c in cells} == {(m,s) for m in METHODS for s in SCENES}
    assert sum(c['source'] == 'adopted_parent_v2' for c in cells) == 12
    assert all(c['status'] == 'complete' and int(c['evaluation_episodes']) == 100 for c in cells)
    curves, audits, all_rows, episodes = {}, [], [], []
    for c in cells:
        m, s = c['method'], c['scenario']
        run = Path(c['run_directory'])
        monitor = run / 'train_monitor.csv'
        evaluation = run / 'paper_evaluation_detailed.json'
        assert v5['sha256'](evaluation) == c['detailed_evaluation_sha256']
        with monitor.open(encoding='utf-8-sig') as f:
            raw = list(csv.DictReader(line for line in f if not line.startswith('#')))
        rows = []
        for i, r in enumerate(raw):
            assert r['is_success'].lower() in ('true', 'false', '0', '1')
            n = float(r['raw_simulation_steps'])
            assert n.is_integer() and n > 0
            rows.append({'is_success': r['is_success'].lower() in ('true', '1'), 'raw_simulation_steps': int(n)})
            reward = float(r['r'])
            assert np.isfinite(reward)
            rows[-1]['episode_reward'] = reward
            episodes.append(dict(method=m, scenario=s, episode=i+1, **rows[-1]))
        curve, stats = v5['build_time_step_curve'](rows)
        rewards = np.array([r['episode_reward'] for r in rows])
        reward_means = v5['padded_episode_window'](rewards, 20)[0]
        expanded = np.repeat(reward_means, [r['raw_simulation_steps'] for r in rows])
        reward_ema = np.zeros(len(expanded)+1)
        for t,x in enumerate(expanded):
            reward_ema[t+1] = .999*reward_ema[t] + .001*x
        reward_state = 0.0
        for x,r in zip(reward_means,rows):
            reward_state = x + (reward_state-x)*.999**r['raw_simulation_steps']
        assert abs(reward_state-reward_ema[-1]) < 1e-10
        for point in curve:
            point['ema_average_reward'] = float(reward_ema[point['time_step']])
            point['episode_window_average_reward'] = float(reward_means[point['episode_index']-1]) if point['episode_index'] else 0.0
        # Independent boundary calculation checks the final state of per-step recurrence.
        state = 0.0
        for i, r in enumerate(rows):
            x = sum(z['is_success'] for z in rows[max(0,i-19):i+1]) / 20
            state = x + (state-x) * 0.999 ** r['raw_simulation_steps']
        assert abs(state - curve[-1]['ema_success_rate']) < 1e-11
        assert curve[0]['ema_success_rate'] == 0
        curves[m,s] = curve
        all_rows.extend(dict(method=m, scenario=s, **r) for r in curve)
        audits.append(dict(method=m, scenario=s, source=c['source'], monitor=monitor.relative_to(ROOT).as_posix(), monitor_sha256=v5['sha256'](monitor), evaluation_sha256=c['detailed_evaluation_sha256'], model_sha256=v5['sha256'](run/'final_model.zip'), **stats))
        print(m, s, len(rows), stats['total_raw_simulation_steps'], flush=True)
    margin_rows, summary = [], []
    for s in SCENES:
        maps = {m:{r['time_step']:r['ema_success_rate'] for r in curves[m,s]} for m in ['mst_slt','temporal_graph','v4_8','v4_13']}
        for m in ['v4_8','v4_13']:
            times = sorted(set(maps[m]) & set(maps['mst_slt']) & set(maps['temporal_graph']))
            delta = [maps[m][t]-max(maps['mst_slt'][t],maps['temporal_graph'][t]) for t in times]
            margin_rows.extend(dict(scenario=s, method=m, time_step=t, margin=d) for t,d in zip(times,delta))
            summary.append(dict(scenario=s,method=m,common_end=times[-1],mean_margin=float(np.mean(delta)),positive_fraction=float(np.mean(np.array(delta)>0)),final_margin=delta[-1]))
    limit = max(0.1, np.ceil(max(abs(r['margin']) for r in margin_rows)*10)/10)
    def draw(ax, s, margin=False):
        if margin:
            for m in ['v4_8','v4_13']:
                rr = [r for r in margin_rows if r['scenario']==s and r['method']==m]
                j=METHODS.index(m)
                ax.plot([r['time_step'] for r in rr], [r['margin'] for r in rr], color=COLORS[j], ls=STYLES[j], lw=1.7)
            ax.axhline(0,color='#333333',ls='--',lw=.8)
            ax.set_ylim(-limit,limit)
            ax.set_ylabel('Success-rate difference')
        else:
            for j,m in enumerate(METHODS):
                rr=curves[m,s]
                ax.plot([r['time_step'] for r in rr],[r['ema_success_rate'] for r in rr],color=COLORS[j],ls=STYLES[j],lw=1.6,label=LABELS[j])
            ax.set_ylim(0,1)
            ax.set_ylabel('Training success rate')
        ax.set_xlim(0,max(curves[m,s][-1]['time_step'] for m in METHODS))
        ax.set_title(NAMES[SCENES.index(s)] + (' | signed margin' if margin else ' | high traffic'))
        ax.set_xlabel('Raw simulation steps')
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        ax.grid(alpha=.25)
        ax.spines[['top','right']].set_visible(False)
    def save(fig, name):
        for ext in ['png','svg','pdf']:
            fig.savefig(OUT/f'{name}.{ext}',dpi=190,bbox_inches='tight')
        plt.close(fig)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    handles = [plt.Line2D([],[],color=COLORS[j],ls=STYLES[j],lw=2,label=LABELS[j]) for j in range(5)]
    for combined in [False, True]:
        fig,axs=plt.subplots(4 if combined else 2,3,figsize=(16,14 if combined else 8))
        for i,s in enumerate(SCENES):
            row=(i//3)*(2 if combined else 1)
            draw(axs[row,i%3],s)
            if combined: draw(axs[row+1,i%3],s,True)
        fig.legend(handles=handles,loc='upper center',ncol=5,bbox_to_anchor=(.5,.99),frameon=False)
        fig.text(.5,.015,'20-episode window (zero-padded) | EMA retention 0.999 per raw step | training seed 0' + ('\nMargins: each version minus max(MST+SLT, TemporalGraph); shared support only' if combined else ''),ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.06,1,.95),h_pad=2)
        save(fig,'final_5m6s_curves_with_margins' if combined else 'final_5m6s_training_curves')
    for s in SCENES:
        fig,axs=plt.subplots(2,1,figsize=(8,7.5))
        draw(axs[0],s)
        draw(axs[1],s,True)
        fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False)
        fig.text(.5,.012,'Margin = version - max(MST+SLT, TemporalGraph) | seed 0',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.91))
        save(fig,s+'_training_curves')
    # Final requested deliverables: per-scenario five-way reward and success comparison.
    def draw_metric(ax,s,metric):
        key = 'ema_average_reward' if metric=='reward' else 'ema_success_rate'
        for j,m in enumerate(METHODS):
            rr=curves[m,s]
            ax.plot([r['time_step'] for r in rr],[r[key] for r in rr],color=COLORS[j],ls=STYLES[j],lw=1.8)
        ax.set_title(NAMES[SCENES.index(s)]+' | high traffic')
        ax.set_ylabel('Average episode reward' if metric=='reward' else 'Success rate')
        ax.set_xlabel('Raw simulation steps')
        ax.set_xlim(0,max(curves[m,s][-1]['time_step'] for m in METHODS))
        if metric=='success': ax.set_ylim(0,1)
        else:
            values=[r[key] for m in METHODS for r in curves[m,s]]
            lo,hi=min(values),max(values)
            pad=max(.05,(hi-lo)*.08)
            ax.set_ylim(lo-pad,hi+pad)
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        ax.grid(alpha=.25)
        ax.spines[['top','right']].set_visible(False)
    for s in SCENES:
        fig,axs=plt.subplots(1,2,figsize=(12,4.7))
        for ax,metric in zip(axs,['reward','success']): draw_metric(ax,s,metric)
        fig.legend(handles=handles,loc='upper center',ncol=5,frameon=False)
        fig.text(.5,.018,'20-episode window (zero-padded) | EMA retention 0.999 per raw step | training seed 0',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.055,1,.9))
        save(fig,s+'_reward_success')
        for metric in ['reward','success']:
            fig,ax=plt.subplots(figsize=(8,5))
            draw_metric(ax,s,metric)
            fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False)
            fig.tight_layout(rect=(0,0,1,.88))
            save(fig,s+'_'+metric)
    for metric in ['reward','success']:
        fig,axs=plt.subplots(2,3,figsize=(16,8))
        for ax,s in zip(axs.flat,SCENES): draw_metric(ax,s,metric)
        fig.legend(handles=handles,loc='upper center',ncol=5,frameon=False)
        fig.text(.5,.015,'20-episode window (zero-padded) | EMA retention 0.999 per raw step | training seed 0',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.045,1,.94),h_pad=2)
        save(fig,'final_5m6s_'+metric)
    dump_csv('training_curve_data.csv',all_rows)
    dump_csv('training_episode_data.csv',episodes)
    dump_csv('signed_margin_data.csv',margin_rows)
    dump_csv('signed_margin_summary.csv',summary)
    manifest=dict(protocol=PROTOCOL.relative_to(ROOT).as_posix(),protocol_sha256=v5['sha256'](PROTOCOL),source_summary_sha256=v5['sha256'](SOURCE),transform_script_sha256=v5['sha256'](V5),plot_script_sha256=v5['sha256'](Path(__file__)),retrained=False,workers_started=0,evaluation_episodes=3000,reused_cells=12,fresh_existing_cells=18,baseline_definition=['mst_slt','temporal_graph'],sources=audits,margin_summary=summary,validation={'complete_matrix':True,'evaluation_hashes_match':True,'independent_ema_boundary_check':True,'prefix_zero_padding':True},uncertainty='Single training seed; no cross-seed confidence bands. EMA bands in CSV inherited from v5 are descriptive only and not rendered.')
    # Ensure plotting has not changed any input logs.
    assert all(v5['sha256'](ROOT/a['monitor'])==a['monitor_sha256'] for a in audits)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with (OUT/'visual-contract.md').open('a',encoding='utf-8') as f:
        f.write('\nFinal user steering: per-scenario paired panels show average episode reward (raw monitor r, not per-step reward) and success rate for all five methods. Reward uses the same padded 20-episode window and raw-step EMA, without clipping or normalization. Separate metric figures and six-scene overviews are also exported. Independent boundary recurrence validates both metrics.\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
