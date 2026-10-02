"""Outcome-independent aggregation and frozen guard checks for the v2 screen."""
import json
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.control_phase2_screen_v2 import jobs,validate_finished,CONTRACT,OUT
from tools.phase1_checkpoint_diagnostics import read,write,sha


def mean_metric(cells,field):
    return statistics.mean(d['summary'][field] for d in cells)


def candidate_guards(control,candidate,limits):
    if len(control)!=6 or len(candidate)!=6:raise ValueError('All six scenes required')
    reasons=[]; per_scene=[]
    for field,maximum in [('collision_rate',limits['mean_collision_delta_max']),('timeout_rate',limits['mean_timeout_delta_max'])]:
        if mean_metric(candidate,field)-mean_metric(control,field)>maximum+1e-12:reasons.append('mean_'+field)
    if mean_metric(candidate,'success_rate')<=mean_metric(control,'success_rate')+1e-12:reasons.append('mean_success_not_improved')
    for i,(a,b) in enumerate(zip(control,candidate)):
        ar,br=a['episode_records'],b['episode_records']
        if [(r['seed'],r['traffic_variant']) for r in ar]!=[(r['seed'],r['traffic_variant']) for r in br]:
            raise ValueError('Traffic/seed pairing mismatch')
        delta=b['summary']['success_rate']-a['summary']['success_rate']
        collision=b['summary']['collision_rate']-a['summary']['collision_rate']
        shared=[(x['completion_time_seconds'],y['completion_time_seconds']) for x,y in zip(ar,br) if x['success'] and y['success']]
        time_ratio=None
        if len(shared)<limits['paired_success_min_episodes']:reasons.append(f'scene_{i}_insufficient_paired_success')
        else:
            base_time=statistics.mean(x for x,y in shared)
            if base_time<=0:raise ValueError('Nonpositive successful completion time')
            time_ratio=statistics.mean(y for x,y in shared)/base_time-1
            if time_ratio>limits['paired_success_time_relative_increase_max']+1e-12:reasons.append(f'scene_{i}_efficiency')
        if delta<limits['per_scenario_success_delta_min']-1e-12:reasons.append(f'scene_{i}_success')
        if collision>limits['per_scenario_collision_delta_max']+1e-12:reasons.append(f'scene_{i}_collision')
        per_scene.append(dict(scene_index=i,success_delta=delta,collision_delta=collision,paired_success_count=len(shared),paired_time_relative_increase=time_ratio))
    return dict(passed=not reasons,reasons=reasons,per_scene=per_scene)


def main():
    c=read(CONTRACT); expected=jobs(c); resolved={}; missing=[]; invalid=[]
    for j in expected:
        directory=OUT/'screen'/j['id']
        if not (directory/'evaluation.json').exists():missing.append(j['id']);continue
        try:
            validate_finished(directory)
            a=read(directory/'arguments.json')
            if a['contract_sha256']!=sha(CONTRACT):raise ValueError('Contract identity mismatch')
            if any(a[k]!=j[k] for k in ('method','scenario','seed')) or a['candidate']['id']!=j['candidate']:raise ValueError('Cell identity mismatch')
            resolved[j['id']]=read(directory/'evaluation.json')
        except Exception as exc:invalid.append(dict(id=j['id'],error=str(exc)))
    dest=OUT/'analysis';dest.mkdir(exist_ok=True)
    state=dict(complete=not missing and not invalid,completed=len(resolved),expected=len(expected),missing=missing,invalid=invalid)
    write(dest/'completion.json',state)
    if not state['complete']:
        print(f"Incomplete: {len(resolved)}/{len(expected)}; selection withheld")
        return
    rows=[];selected={}
    def cells(m,k):return [resolved[f'{m}__{s}__{k}__seed0'] for s in c['scenarios']]
    for method in c['methods']:
        control=cells(method,'control');eligible=[]
        for index,variant in enumerate(c['candidates']):
            key=variant['id']
            if method==c['baseline_method'] and key!='control':continue
            data=cells(method,key)
            metrics={f:mean_metric(data,f) for f in ['success_rate','collision_rate','timeout_rate']}
            guard=candidate_guards(control,data,c['promotion_guards_relative_to_own_control']) if key!='control' else None
            rows.append(dict(method=method,candidate=key,metrics=metrics,guards=guard))
            if guard and guard['passed']:eligible.append((-metrics['success_rate'],metrics['collision_rate'],metrics['timeout_rate'],index,key))
        if method in c['optimization_methods']:selected[method]=min(eligible)[-1] if eligible else 'control'
    write(dest/'screen_decision.json',dict(rows=rows,selected_for_confirmation=selected,scenes=c['scenarios'],
        training_seeds=[0],development_only=True,formal_claim_allowed=False,contract_sha256=sha(CONTRACT)))
    print(json.dumps(selected))


if __name__=='__main__':main()
