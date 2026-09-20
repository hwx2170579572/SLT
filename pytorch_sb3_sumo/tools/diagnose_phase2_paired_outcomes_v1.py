"""Paired terminal outcomes and successful completion times; no behavioral inference."""
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.phase1_checkpoint_diagnostics import read,write,sha
from tools.analyze_phase2_reuse_v3 import main as validate,OUT


def paired(a,b):
    keys=lambda rows:[(r['seed'],r['traffic_variant']) for r in rows]
    if keys(a)!=keys(b) or len(set(keys(a)))!=len(a):
        raise ValueError('Nonunique or mismatched seed/traffic pairs')
    shared=[(x,y) for x,y in zip(a,b) if x['success'] and y['success']]
    diffs=[y['completion_time_seconds']-x['completion_time_seconds'] for x,y in shared]
    base=statistics.mean(x['completion_time_seconds'] for x,y in shared) if shared else None
    variant=statistics.mean(y['completion_time_seconds'] for x,y in shared) if shared else None
    return dict(episodes=len(a),both_success=len(shared),
        lost_success=sum(x['success'] and not y['success'] for x,y in zip(a,b)),
        gained_success=sum(not x['success'] and y['success'] for x,y in zip(a,b)),
        both_not_success=sum(not x['success'] and not y['success'] for x,y in zip(a,b)),
        success_to_collision=sum(x['success'] and y['collision'] for x,y in zip(a,b)),
        collision_to_success=sum(x['collision'] and y['success'] for x,y in zip(a,b)),
        paired_base_seconds=base,paired_variant_seconds=variant,
        paired_mean_delta_seconds=statistics.mean(diffs) if diffs else None,
        paired_median_delta_seconds=statistics.median(diffs) if diffs else None,
        paired_relative_increase=variant/base-1 if base and base>0 else None,
        slower=sum(d>1e-9 for d in diffs),faster=sum(d < -1e-9 for d in diffs),
        equal=sum(abs(d)<=1e-9 for d in diffs))


def main():
    validate()
    evidence=read(OUT/'analysis/evidence_sources.json')
    data={}
    for key,s in evidence['sources'].items():
        path=ROOT/s['path']
        if sha(path)!=s['sha256']:raise ValueError('Source changed')
        data[key]=read(path)['episode_records']
    rows=[]
    for key,b in data.items():
        method,scene,candidate,seed=key.split('__')
        if candidate=='control':continue
        a=data[f'{method}__{scene}__control__{seed}']
        row=dict(id=key,comparison=paired(a,b),traffic_groups={})
        for traffic in sorted({r['traffic_variant'] for r in a}):
            row['traffic_groups'][traffic]=paired([r for r in a if r['traffic_variant']==traffic],
                                                 [r for r in b if r['traffic_variant']==traffic])
        rows.append(row)
    dest=OUT/'diagnosis_v1'
    write(dest/'paired_outcomes.json',dict(rows=rows,evidence=evidence,development_only=True,
        limitations=['Terminal records contain no speed, waiting, action or interaction trajectories.',
                     'Paired time statistics condition on both policies succeeding.',
                     'Traffic strata and one training seed are descriptive, not independent training replications.',
                     'Collision transition counts may overlap other terminal flags.']))
    for key in ('v4_13__carla__tau_half__seed0','v4_13__cross__tau_half__seed0'):
        row=next(r for r in rows if r['id']==key)
        print(key,row)


if __name__=='__main__':main()
