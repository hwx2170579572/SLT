"""Locate existing seed 1/2 runs before confirmation; discovery never authorizes training."""
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.phase1_checkpoint_diagnostics import read,write,sha
from tools.audit_phase2_reuse_v1 import relative


def main():
    proposal=ROOT/'results_phase2_confirmation_v2/proposed_manifest.json'
    p=read(proposal)
    plan=read(ROOT/'results_phase1_checkpoint_diagnostics_v1/plan.json')
    aliases={}
    for source in plan['sources']:
        if source['method'] not in ('mst_slt','v4_8','v4_13'):continue
        a=read(ROOT/source['run']/'arguments.json')['requested_raw_steps']
        aliases[a['algo']]=source['method']
    roots=sorted(x for x in ROOT.iterdir() if x.is_dir() and x.name.startswith('results'))
    errors=[];matches=[];scanned=0;seed_runs=[]
    for root in roots:
        scan=('\\\\?\\'+str(root)) if os.name=='nt' else str(root)
        for directory,children,files in os.walk(scan,onerror=lambda e:errors.append(dict(path=str(e.filename),error=str(e)))):
            children[:]=[n for n in children if not n.startswith(('pytest','tmp','overlays','tensorboard','tb_'))
                         and n not in ('ov','tb','checkpoints','__pycache__','.git')]
            if 'arguments.json' not in files:continue
            path=Path(directory)/'arguments.json';scanned+=1
            try:
                a=read(path);r=a.get('requested_raw_steps',{})
                seed=r.get('seed',a.get('seed'))
                if seed not in p['training_seeds']:continue
                method=aliases.get(r.get('algo'),a.get('method'))
                scene=r.get('scenario',a.get('scenario'))
                item=dict(arguments=relative(path),arguments_sha256=sha(path),method=method,scenario=scene,
                          seed=seed,final_model_exists=(path.parent/'final_model.zip').exists())
                seed_runs.append(item)
                for j in p['proposed_jobs']:
                    # Do not discard incomplete or differently configured runs before protocol review.
                    if (method,scene,seed)==(j['method'],j['scenario'],j['seed']):
                        matches.append(dict(job=j['id'],**item,status='requires_protocol_and_model_audit'))
            except (OSError,ValueError,TypeError) as exc:
                errors.append(dict(path=relative(path),error=str(exc)))
    covered={m['job'] for m in matches}
    write(proposal.parent/'reuse_discovery_v1.json',dict(proposal_sha256=sha(proposal),
        scan_roots=[str(r.relative_to(ROOT)) for r in roots],scanned_arguments=scanned,scan_errors=errors,
        seed_1_2_runs=seed_runs,potential_matches=matches,
        unmatched_jobs=[j['id'] for j in p['proposed_jobs'] if j['id'] not in covered],
        launch_allowed=False,rule='Discovery only; audit matching artifacts and resolve scan errors before scheduling'))
    print(json.dumps(dict(scanned=scanned,seed_runs=len(seed_runs),potential_matches=len(matches),
                         unmatched=len(p['proposed_jobs'])-len(covered),errors=len(errors))))


if __name__=='__main__':main()
