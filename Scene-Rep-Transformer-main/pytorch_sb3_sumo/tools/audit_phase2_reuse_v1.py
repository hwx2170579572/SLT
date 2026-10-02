"""Inventory historical controls and search for existing tuning runs; no simulations."""
import sys
import zipfile
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.phase1_checkpoint_diagnostics import read,write,sha,validate_report
from tools.control_phase2_screen_v2 import jobs,CONTRACT


def relative(path):
    value=str(path)
    if value.startswith('\\\\?\\'):value=value[4:]
    return str(Path(value).relative_to(ROOT))


def main():
    c=read(CONTRACT);plan=read(ROOT/'results_phase1_checkpoint_diagnostics_v1/plan.json')
    controls=[];algorithms={}
    for j in plan['jobs']:
        if j['method'] not in c['methods'] or j['kind']!='exact_final':continue
        result=ROOT/'results_phase1_checkpoint_diagnostics_v1/jobs'/j['id']/'result.json'
        d=read(result);validate_report(d,100,10000)
        if sha(ROOT/j['checkpoint'])!=j['checkpoint_sha256'] or d['checkpoint_sha256']!=j['checkpoint_sha256']:raise ValueError('Historical model hash mismatch')
        if d['decoder']!=c['decoder_per_cell'][j['method']+'__'+j['scenario']]:raise ValueError('Decoder mismatch')
        args=read(ROOT/j['run']/'arguments.json')['requested_raw_steps']
        algorithms[args['algo']]=j['method']
        with zipfile.ZipFile(ROOT/j['checkpoint']) as z:data=json.loads(z.read('data'))
        if data['_raw_steps_seen']!=50000 or data['learning_rate']!=1e-4 or data['tau']!=.005 or args['seed']!=0:raise ValueError('Historical control settings mismatch')
        controls.append(dict(id=f"{j['method']}__{j['scenario']}__control__seed0",method=j['method'],scenario=j['scenario'],
            action='reference_existing_development_result_do_not_retrain',result=str(result.relative_to(ROOT)),result_sha256=sha(result),
            checkpoint=j['checkpoint'],checkpoint_sha256=j['checkpoint_sha256'],decoder=j['decoder'],
            historical_training_code_equivalence='not fully established; no matched causal attribution from this reference alone'))
    assert len(controls)==18
    # Search only research-result roots; avoid unrelated temporary/test folders.
    roots=[p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith(('results_topo_v4_8','results_topo_v4_13','results_hd','results_iv2','results_phase2_runtime'))]
    scanned=0;potential=[];errors=[]
    novel=[j for j in jobs(c) if j['candidate']!='control']
    candidates={x['id']:x for x in c['candidates']}
    for base in roots:
        argument_files=[]
        scan_base=('\\\\?\\'+str(base)) if os.name=='nt' else str(base)
        for directory,children,files in os.walk(scan_base,onerror=lambda e:errors.append(dict(path=str(e.filename),error=str(e)))):
            children[:]=[n for n in children if not n.startswith(('overlays','tensorboard','tb_')) and n not in ('ov','tb','checkpoints','__pycache__')]
            if 'arguments.json' in files:argument_files.append(Path(directory)/'arguments.json')
        for file in argument_files:
            scanned+=1
            try:
                a=read(file);requested=a.get('requested_raw_steps',{})
                method=algorithms.get(requested.get('algo'),a.get('method'))
                seed=requested.get('seed',a.get('seed'));scene=requested.get('scenario',a.get('scenario'))
                model=file.parent/'final_model.zip'
                if method not in c['optimization_methods'] or seed!=0 or not model.exists():continue
                with zipfile.ZipFile(model) as z:data=json.loads(z.read('data'))
                if data.get('_raw_steps_seen')!=50000:continue
                for j in novel:
                    v=candidates[j['candidate']]
                    if j['method']==method and j['scenario']==scene and data.get('learning_rate')==v['learning_rate'] and data.get('tau')==v['tau']:
                        potential.append(dict(job=j['id'],model=relative(model),arguments=relative(file),action='audit_protocol_and_reuse_training_before_any_new_run'))
            except (OSError,ValueError,KeyError,zipfile.BadZipFile) as exc:errors.append(dict(path=relative(file),error=str(exc)))
    covered={p['job'] for p in potential}
    payload=dict(schema='phase2-reuse-inventory/v1',user_rule='Do not repeat experiments already run',controls=controls,
        proposed_missing_candidates=[j for j in novel if j['id'] not in covered],potential_existing_candidates=potential,
        scanned_argument_files=scanned,scan_errors=errors,scan_roots=[str(p.relative_to(ROOT)) for p in roots],
        no_new_training_started=True,original_queue_pause='results_phase2_runtime_v2/pause_new_jobs.json',
        note='Existing historical controls remain separately attributed; old training-code uncertainty does not authorize automatic retraining')
    dest=ROOT/'results_phase2_reuse_v3';dest.mkdir(exist_ok=True);write(dest/'inventory.json',payload)
    print(json.dumps(dict(historical_controls=len(controls),scanned=scanned,potential_existing_variants=len(potential),missing_variant_jobs=len(payload['proposed_missing_candidates']),scan_errors=len(errors))))


if __name__=='__main__':main()
