"""Reproducible read-only diagnosis; excludes Full+BalancedSlots by user request."""
from pathlib import Path
import csv
import json
import statistics as st
import zipfile
from collections import Counter
from phase1_checkpoint_diagnostics import read, sha, validate_report

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results_phase1_checkpoint_diagnostics_v1'
OUT = ROOT / 'results_phase2_diagnosis_20260908'
METHODS = ['mst_slt', 'temporal_graph', 'v4_8', 'v4_13']
SCENES = ['left_turn', 'cross', 'roundabout_easy', 'roundabout_medium', 'roundabout', 'carla']

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def csvout(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def outcome(r):
    keys = [k for k in ['success', 'collision', 'off_route', 'timeout'] if r[k]]
    # Terminal flags can overlap at the time limit; retain their joint state.
    return '+'.join(keys) if keys else 'unclassified'

def paired(a, b):
    aa, bb = a['episode_records'], b['episode_records']
    assert [(r['seed'], r['traffic_variant']) for r in aa] == [(r['seed'], r['traffic_variant']) for r in bb]
    counts = Counter(outcome(x) + '->' + outcome(y) for x, y in zip(aa, bb))
    times = [y['completion_time_seconds'] - x['completion_time_seconds'] for x, y in zip(aa, bb) if x['success'] and y['success']]
    return {'transitions': dict(counts), 'both_success_count': len(times),
            'both_success_mean_time_delta_seconds': st.mean(times) if times else None}

def main():
    OUT.mkdir(exist_ok=True)
    p = read(SOURCE / 'plan.json'); resolved = {}; hashes = {}; configs = []; windows = []; joint_flags = []
    for j in p['jobs']:
        if j['method'] not in METHODS: continue
        path = SOURCE / 'jobs' / j['id'] / 'result.json'; d = read(path)
        validate_report(d, 100, 10000)
        assert d['checkpoint_sha256'] == j['checkpoint_sha256'] == sha(ROOT / j['checkpoint'])
        assert d['decoder'] == j['decoder']
        for r in d['episode_records']:
            state = outcome(r)
            if '+' in state or state == 'unclassified':
                joint_flags.append(dict(job=j['id'], seed=r['seed'], terminal_state=state))
        resolved[j['id']] = d; hashes[str(path.relative_to(ROOT))] = sha(path)
    for a in p['aliases']:
        if a['method'] in METHODS:
            assert sha(ROOT / a['checkpoint']) == a['checkpoint_sha256']
            resolved[a['job']] = resolved[a['target']]
    assert len(resolved) == 4 * 6 * 6
    metrics = []
    for m in METHODS:
        for s in SCENES:
            for kind in ['exact_final'] + [f'raw_{k}' for k in range(10000, 50001, 10000)]:
                d = resolved[f'{m}__{s}__{kind}']
                metrics.append(dict(method=m, scenario=s, checkpoint=kind, **{k: d['summary'][k] for k in ['success_rate','collision_rate','timeout_rate']}, success_time_seconds=d['mean_success_completion_time_seconds']))
    csvout('metrics.csv', metrics)
    save('terminal_flag_overlap.json', joint_flags)
    comparisons = {}; drift = {}
    for s in SCENES:
        for m in METHODS[1:]:
            comparisons[f'{m}__{s}'] = paired(resolved[f'mst_slt__{s}__exact_final'], resolved[f'{m}__{s}__exact_final'])
        drift[s] = paired(resolved[f'v4_13__{s}__raw_10000'], resolved[f'v4_13__{s}__exact_final'])
    save('paired_transitions.json', dict(vs_mst_final=comparisons, v413_10k_to_50k=drift))
    for cell in p['sources']:
        if cell['method'] not in METHODS: continue
        run = ROOT / cell['run']; args = read(run / 'arguments.json'); diag = read(run / 'training_diagnostics.json')
        with zipfile.ZipFile(run / 'final_model.zip') as z: data = json.loads(z.read('data'))
        configs.append(dict(method=cell['method'], scenario=cell['scenario'], **{k: data.get(k) for k in ['learning_rate','representation_learning_rate','tau','target_update_interval','gradient_steps','_n_updates','_raw_steps_seen']}, declared_traffic_protocol=args['requested_raw_steps']['traffic_protocol'], training_diagnostic_updates=diag['learner_updates']))
        for name in ['arguments.json','training_diagnostics.json','train_monitor.csv']:
            hashes[str((run/name).relative_to(ROOT))] = sha(run/name)
        with (run/'train_monitor.csv').open() as f:
            reader = csv.DictReader(line for line in f if not line.startswith('#')); bins = {}; raw = 0
            for row in reader:
                raw += int(row['raw_simulation_steps']); end = min(50000, ((raw-1)//10000+1)*10000)
                bins.setdefault(end, []).append(row)
        for end, rr in bins.items():
            windows.append(dict(method=cell['method'], scenario=cell['scenario'], raw_window_end=end, completed_episodes=len(rr), **{k:sum(r[k].lower()=='true' for r in rr)/len(rr) for k in ['is_success','collision','max_time']}))
    assert len(configs) == 24
    assert all(r['_raw_steps_seen']==50000 and r['_n_updates']==45001 and r['training_diagnostic_updates']==45001 for r in configs)
    csvout('effective_configs.csv', configs); csvout('training_episode_windows.csv', windows)
    # TensorBoard steps are decision steps; join to the logged raw clock rather than multiplying by three.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tbrows = []
    cell = next(s for s in p['sources'] if s['method']=='v4_13' and s['scenario']=='cross')
    tbdir = Path(read(ROOT/cell['run']/'arguments.json')['runtime_paths']['tensorboard_log'])
    tags = ['train/critic_loss','risk/policy_expected_collision_cost','risk/collision_probability_absolute_error','risk/collision_cost_label','risk/reward_twin_disagreement','support/replay_joint_action_nll_recomputed','hybrid/lane_entropy','diagnostic/route_compatible_attention_mass','diagnostic/topology_fallback_rate']
    for event in tbdir.rglob('events.out.tfevents.*'):
        hashes[str(event.relative_to(ROOT))] = sha(event)
        acc = EventAccumulator(str(event), size_guidance={'scalars':0}); acc.Reload()
        clock = {v.step: v.value for v in acc.Scalars('time/raw_simulation_steps')}
        for tag in tags:
            if tag not in acc.Tags()['scalars']: continue
            grouped = {}
            for v in acc.Scalars(tag):
                assert v.step in clock
                end = min(50000, int((clock[v.step]-1)//10000+1)*10000)
                grouped.setdefault(end, []).append(v.value)
            for end, values in grouped.items():
                tbrows.append(dict(tag=tag, raw_window_end=end, logged_points=len(values), mean=st.mean(values), minimum=min(values), maximum=max(values)))
    assert tbrows
    csvout('cross_v413_tensorboard_windows.csv', tbrows)
    save('source_hashes.json', hashes)
    macro = []
    for m in METHODS:
        rr = [r for r in metrics if r['method']==m and r['checkpoint']=='exact_final']
        macro.append(dict(method=m, **{k:st.mean(r[k] for r in rr) for k in ['success_rate','collision_rate','timeout_rate']}))
    save('diagnosis.json', dict(decision='limited_optimization_tuning_before_structure', methods=METHODS, excluded_methods=['full_balanced'], exclusion_reason='Explicit user instruction on 2026-09-08', unique_evaluations=120, logical_evaluations=144, training_seeds=[0], macro=macro, hypotheses=['Cross checkpoint drift with collision failures; optimization hypothesis, not proven cause','Checkpoint selection materially affects prior aggregate advantage','Roundabout-C completion-time cost remains after conditioning on paired successes'], no_new_training=True, no_new_evaluation=True, formal_evidence=False))
    print(json.dumps(dict(macro=macro, cross_drift=drift['cross'], roundabout_c=comparisons['v4_13__roundabout'], checked_cells=len(configs), source_files=len(hashes)), ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
