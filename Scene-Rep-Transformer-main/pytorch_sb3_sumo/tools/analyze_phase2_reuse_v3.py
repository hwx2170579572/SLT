"""Analyze referenced historical controls plus new variants without copying evidence."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.phase1_checkpoint_diagnostics import read,write,sha,validate_report
from tools.control_phase2_screen_v2 import jobs,validate_finished,CONTRACT
from tools.analyze_phase2_screen_v2 import mean_metric,candidate_guards
OUT=ROOT/'results_phase2_reuse_v3'
RUNTIME=ROOT/'results_phase2_runtime_v2'


def check_manifest(manifest,contract):
    expected=jobs(contract)
    for field,control in [('historical_controls',True),('jobs',False)]:
        ids=[r['id'] for r in manifest[field]]
        target={j['id'] for j in expected if (j['candidate']=='control')==control}
        if len(ids)!=len(set(ids)) or set(ids)!=target:
            raise ValueError(f'Incomplete or duplicate {field} coverage')


def historical_result(reference,contract):
    path=ROOT/reference['result'];model=ROOT/reference['checkpoint']
    if sha(path)!=reference['result_sha256'] or sha(model)!=reference['checkpoint_sha256']:
        raise ValueError('Historical artifact hash mismatch')
    d=read(path);validate_report(d,100,10000)
    if any(d[k]!=reference[k] for k in ('method','scenario','checkpoint_sha256','decoder')):
        raise ValueError('Historical reference identity mismatch')
    if d['kind']!='exact_final' or d['raw_steps']!=50000:
        raise ValueError('Historical reference must be exact final')
    if d['decoder']!=contract['decoder_per_cell'][d['method']+'__'+d['scenario']]:
        raise ValueError('Historical deployment condition mismatch')
    return d


def ready_to_select(expected,resolved,invalid):
    return not invalid and set(resolved)==set(expected)


def main():
    contract=read(CONTRACT);manifest_path=OUT/'execution_manifest.json';manifest=read(manifest_path)
    check_manifest(manifest,contract)
    expected={j['id']:j for j in jobs(contract)};resolved={};invalid=[];sources={}
    for r in manifest['historical_controls']:
        try:
            resolved[r['id']]=historical_result(r,contract)
            sources[r['id']]=dict(origin='historical_reference',path=r['result'],sha256=r['result_sha256'],
                training_code_equivalence=r['historical_training_code_equivalence'])
        except Exception as exc:invalid.append(dict(id=r['id'],error=str(exc)))
    for j in manifest['jobs']:
        directory=RUNTIME/'screen'/j['id']
        if not (directory/'evaluation.json').exists():continue
        try:
            validate_finished(directory);a=read(directory/'arguments.json')
            if a['contract_sha256']!=sha(CONTRACT):raise ValueError('Candidate contract mismatch')
            if any(a[k]!=j[k] for k in ('method','scenario','seed')):raise ValueError('Candidate identity mismatch')
            desired=next(c for c in contract['candidates'] if c['id']==j['candidate'])
            if a['candidate']!=desired:raise ValueError('Candidate settings mismatch')
            result=directory/'evaluation.json';d=read(result)
            decoder=contract['decoder_per_cell'][j['method']+'__'+j['scenario']]
            if d['decoder']!=decoder or a['decoder']!=decoder:raise ValueError('Candidate decoder mismatch')
            resolved[j['id']]=d;sources[j['id']]=dict(origin='new_variant',path=str(result.relative_to(ROOT)),sha256=sha(result))
        except Exception as exc:invalid.append(dict(id=j['id'],error=str(exc)))
    complete=ready_to_select(expected,resolved,invalid);dest=OUT/'analysis';dest.mkdir(exist_ok=True)
    write(dest/'completion.json',dict(complete=complete,expected=len(expected),resolved=len(resolved),
        historical_references=sum(s['origin']=='historical_reference' for s in sources.values()),
        new_variants=sum(s['origin']=='new_variant' for s in sources.values()),
        missing=sorted(set(expected)-set(resolved)),invalid=invalid))
    write(dest/'evidence_sources.json',dict(contract_sha256=sha(CONTRACT),manifest_sha256=sha(manifest_path),sources=sources))
    if not complete:
        print(f'Validated {len(resolved)}/66 logical cells; selection withheld; original evidence not copied');return
    def cells(m,c):return [resolved[f'{m}__{s}__{c}__seed0'] for s in contract['scenarios']]
    rows=[];selection={};baseline=cells('mst_slt','control')
    for method in contract['methods']:
        control=cells(method,'control');eligible=[]
        for index,c in enumerate(contract['candidates']):
            key=c['id']
            if method=='mst_slt' and key!='control':continue
            data=cells(method,key);metrics={f:mean_metric(data,f) for f in ('success_rate','collision_rate','timeout_rate')}
            guard=candidate_guards(control,data,contract['promotion_guards_relative_to_own_control']) if key!='control' else None
            rows.append(dict(method=method,candidate=key,metrics=metrics,guards=guard,
                success_delta_vs_historical_mst=metrics['success_rate']-mean_metric(baseline,'success_rate')))
            if guard and guard['passed']:eligible.append((-metrics['success_rate'],metrics['collision_rate'],metrics['timeout_rate'],index,key))
        if method in contract['optimization_methods']:selection[method]=min(eligible)[-1] if eligible else 'control'
    write(dest/'screen_decision.json',dict(rows=rows,selected_for_confirmation=selection,
        contract_sha256=sha(CONTRACT),manifest_sha256=sha(manifest_path),evidence_sources_sha256=sha(dest/'evidence_sources.json'),
        training_seeds=[0],development_only=True,causal_tuning_attribution_allowed=False,
        limitation='Historical training-code equivalence not fully established; differences are descriptive. Confirmation must audit/reuse matching runs before scheduling anything new.'))
    print(selection)


if __name__=='__main__':main()
