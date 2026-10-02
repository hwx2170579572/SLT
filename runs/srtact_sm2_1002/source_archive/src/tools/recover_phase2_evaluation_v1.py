"""Recover evaluation only from a completed training receipt; never retrain."""
import argparse
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.phase1_checkpoint_diagnostics import read,write,sha,validate_report,summarize
from tools.phase2_model_factory_v1 import verify_optimizer_settings


def validate_block(d,*,count,offset,checkpoint_hash,decoder,smoke,reference):
    validate_report(d,count,10000+offset)
    if d.get('checkpoint_sha256')!=checkpoint_hash or d.get('decoder')!=decoder or d.get('smoke')!=smoke:
        raise ValueError('Block model/decoder/scope mismatch')
    if [r['traffic_variant'] for r in d['episode_records']] != [r['traffic_variant'] for r in reference[offset:offset+count]]:
        raise ValueError('Block traffic schedule mismatch')


def inspect(directory):
    a=read(directory/'arguments.json'); complete=read(directory/'training_complete.json')
    checkpoint=directory/'final_model.zip'
    if sha(checkpoint)!=complete['checkpoint_sha256']:raise ValueError('Training model hash mismatch')
    if complete['raw_steps']!=a['raw_budget'] or complete['learner_updates']!=a['raw_budget']-a['warmup']+1:
        raise ValueError('Training budget receipt mismatch')
    contract=ROOT/'results_phase2_diagnosis_20260908/tuning_contract_v2.json'
    if a['contract_sha256']!=sha(contract):raise ValueError('Contract changed')
    c=read(contract)
    if a['method'] not in c['methods'] or a['candidate'] not in c['candidates']:raise ValueError('Unknown method/candidate')
    if a['method']==c['baseline_method'] and a['candidate']['id'] not in c['baseline_candidates']:raise ValueError('Baseline tuning forbidden')
    if not a['smoke']:
        frozen=read(ROOT/'results_phase2_runtime_v2/preflight_v2.json')
        for p,h in frozen['source_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('Frozen input changed: '+p)
    plan=read(ROOT/'results_phase1_checkpoint_diagnostics_v1/plan.json')
    source=next(s for s in plan['sources'] if s['method']==a['method'] and s['scenario']==a['scenario'])
    if a['decoder']!=source['decoder']:raise ValueError('Deployment decoder changed')
    reference=read(ROOT/source['run']/'paper_evaluation_detailed.json')['episode_records']
    count=1 if a['smoke'] else 100; blocks={}; missing=[]
    for offset in range(0,count,20):
        p=directory/f'evaluation_block_{offset:03d}.json'
        if not p.exists():missing.append(offset);continue
        d=read(p)
        validate_block(d,count=min(20,count-offset),offset=offset,checkpoint_hash=complete['checkpoint_sha256'],decoder=a['decoder'],smoke=a['smoke'],reference=reference)
        blocks[offset]=d
    return a,complete,reference,blocks,missing


def main(args):
    import psutil
    directory=Path(args.directory).resolve()
    allowed=(ROOT/'results_phase2_runtime_v2').resolve()
    if not directory.is_relative_to(allowed):raise ValueError('Recovery restricted to v2 runtime directory')
    a,complete,reference,blocks,missing=inspect(directory)
    if not args.execute:
        print(dict(mode='read_only',valid_blocks=len(blocks),missing_offsets=missing,training_verified=True));return
    process_file=allowed/'logs'/(directory.name+'.process.json')
    if process_file.exists():
        p=read(process_file)
        if psutil.pid_exists(p['pid']) and abs(psutil.Process(p['pid']).create_time()-p['create_time'])<1:
            raise RuntimeError('Original worker is alive; recovery refused')
    if read(directory/'status.json').get('status')=='completed':
        raise ValueError('Already completed; use read-only inspection')
    lock=directory/'evaluation_recovery.lock'
    with lock.open('x') as f:f.write(str(__import__('os').getpid()))
    env=None; started=time.time()
    try:
        # Preserve the pre-recovery status, including failure details, before any mutation.
        receipt=directory/f'recovery_{time.time_ns()}.json'
        write(receipt,dict(original_status=read(directory/'status.json'),reused_offsets=list(blocks),missing_offsets=missing,tool_sha256=sha(Path(__file__))))
        import torch
        torch.set_num_threads(1)
        from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory
        from algos.sb3_torch import evaluate_model_detailed
        from tools.paper_evaluation_contract import validate_model_environment_spaces
        plan=read(ROOT/'results_phase1_checkpoint_diagnostics_v1/plan.json')
        protocol=read(ROOT/plan['protocol'])
        factory=_make_environment_factory(adapter='base' if a['method']=='mst_slt' else a['method'],
            density=protocol['scenarios'][a['scenario']],overlay_root=allowed/'recovery_ov'/sha(receipt)[:12])
        env=factory(argparse.Namespace(**a['inherited_environment_arguments']),evaluation=True)
        checkpoint=directory/'final_model.zip'
        if a['method']=='mst_slt':
            from algos.sb3_torch.sac import SceneRepresentationSAC
            model=SceneRepresentationSAC.load(checkpoint,env=env,device=args.device)
        elif a['method']=='v4_8':
            from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45
            from tools.action_diagnostics_v4_6 import load_model_for_deployment
            model=load_model_for_deployment(ConfidentActorFusionSACV45,checkpoint,decoder=a['decoder'],env=env,device=args.device)
        else:
            from tools.action_diagnostics_v4_13_model import load_model_for_deployment_v4_13,GradientIsolatedTemperedJointSupportSACV413
            model=load_model_for_deployment_v4_13(GradientIsolatedTemperedJointSupportSACV413,checkpoint,decoder=a['decoder'],env=env,device=args.device)
        verify_optimizer_settings(model,learning_rate=a['candidate']['learning_rate'],tau=a['candidate']['tau'])
        if model._raw_steps_seen!=complete['raw_steps'] or model._n_updates!=complete['learner_updates']:raise ValueError('Loaded budget mismatch')
        validate_model_environment_spaces(model,env)
        total=1 if a['smoke'] else 100
        for offset in missing:
            d=evaluate_model_detailed(model,env,episodes=min(20,total-offset),seed=10000+offset,
                deterministic=True,sumo_step_seconds=.1,policy_action_hold=1).to_dict()
            d.update(checkpoint_sha256=complete['checkpoint_sha256'],decoder=a['decoder'],smoke=a['smoke'])
            validate_block(d,count=min(20,total-offset),offset=offset,checkpoint_hash=complete['checkpoint_sha256'],decoder=a['decoder'],smoke=a['smoke'],reference=reference)
            write(directory/f'evaluation_block_{offset:03d}.json',d);blocks[offset]=d
        records=[]
        for offset in sorted(blocks):
            for r in blocks[offset]['episode_records']:records.append(dict(r,episode=r['seed']-10000))
        d=dict(**summarize(records),checkpoint_sha256=complete['checkpoint_sha256'],decoder=a['decoder'],smoke=a['smoke'])
        validate_report(d,total,10000);write(directory/'evaluation.json',d)
        write(directory/'status.json',dict(status='completed',smoke=a['smoke'],scientific_result=not a['smoke'],
            raw_steps=complete['raw_steps'],learner_updates=complete['learner_updates'],evaluation_episodes=total,
            recovery_receipt=receipt.name,recovery_wall_seconds=time.time()-started))
    finally:
        if env is not None:env.close()
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--execute',action='store_true');p.add_argument('--device',default='cuda');main(p.parse_args())
