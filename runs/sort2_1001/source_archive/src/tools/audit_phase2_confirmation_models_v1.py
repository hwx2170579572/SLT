"""Compare historical training metadata and fixed-budget checkpoints without loading pickle."""
import json
import sys
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.phase1_checkpoint_diagnostics import read,write,sha

TRAIN_FIELDS=('algo','batch_size','learning_rate','discount','learning_starts','buffer_size',
              'neighbors','history_steps','path_length','action_repeat','ego_control_profile',
              'traffic_protocol','episode_limit_profile')


def differences(old,reference):
    return {k:dict(existing=old.get(k),reference=reference.get(k)) for k in TRAIN_FIELDS
            if k not in old or k not in reference or old[k]!=reference[k]}


def metadata_errors(model,job):
    expected=dict(_raw_steps_seen=50000,_n_updates=45001,seed=job['seed'],
        learning_rate=job['candidate']['learning_rate'],tau=job['candidate']['tau'],
        batch_size=32,buffer_size=20000,gamma=.99,gradient_steps=3)
    return {k:dict(existing=model.get(k),expected=v) for k,v in expected.items() if model.get(k)!=v}


def main():
    dest=ROOT/'results_phase2_confirmation_v2'
    discovery=read(dest/'reuse_discovery_v1.json');proposal=read(dest/'proposed_manifest.json')
    if discovery['scan_errors'] or discovery['proposal_sha256']!=sha(dest/'proposed_manifest.json'):
        raise ValueError('Discovery stale or incomplete')
    jobs={j['id']:j for j in proposal['proposed_jobs']}
    sources={(s['method'],s['scenario']):s for s in read(ROOT/'results_phase1_checkpoint_diagnostics_v1/plan.json')['sources']}
    rows=[]
    for match in discovery['potential_matches']:
        job=jobs[match['job']];args=ROOT/match['arguments']
        if sha(args)!=match['arguments_sha256']:raise ValueError('Arguments changed')
        a=read(args)['requested_raw_steps'];source=sources[(job['method'],job['scenario'])]
        refpath=ROOT/source['run']/'arguments.json';ref=read(refpath)['requested_raw_steps']
        models=[args.parent/'final_model.zip']
        models.extend(sorted((args.parent/'checkpoints').glob('*50000_steps.zip')))
        candidates=[]
        for path in models:
            if not path.exists():continue
            with zipfile.ZipFile(path) as z:meta=json.loads(z.read('data'))
            errors=metadata_errors(meta,job)
            candidates.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),errors=errors,
                raw_steps=meta.get('_raw_steps_seen'),learner_updates=meta.get('_n_updates')))
        diff=differences(a,ref)
        evaluations=[]
        for path in sorted(args.parent.glob('*evaluation*.json')):
            e=read(path)
            evaluations.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),
                seed_start=e.get('evaluation_seed_start'),split=e.get('evaluation_split'),
                decoder=e.get('selected_deployment_decoder'),model_sha256=e.get('model_sha256'),
                episodes=len(e.get('episode_records',[]))))
        rows.append(dict(job=job['id'],arguments=str(args.relative_to(ROOT)),arguments_sha256=sha(args),
            reference_arguments=str(refpath.relative_to(ROOT)),reference_arguments_sha256=sha(refpath),
            training_argument_differences=diff,checkpoints=candidates,evaluations=evaluations,
            desired_decoder=job['decoder'],metadata_compatible=not diff and any(not c['errors'] for c in candidates),
            decision='Do not retrain; finish environment/source equivalence audit and evaluate compatible fixed-budget model if existing evaluation does not match'))
    write(dest/'reuse_model_audit_v1.json',dict(proposal_sha256=sha(dest/'proposed_manifest.json'),rows=rows,
        metadata_compatible=sum(r['metadata_compatible'] for r in rows),launch_allowed=False,
        limitation='Metadata compatibility does not prove historical source or physical traffic identity; final evaluation compatibility still needs validation'))
    print(json.dumps([dict(job=r['job'],compatible=r['metadata_compatible'],argument_differences=r['training_argument_differences'],
        matching_models=[c['path'] for c in r['checkpoints'] if not c['errors']]) for r in rows],indent=2))


if __name__=='__main__':main()
