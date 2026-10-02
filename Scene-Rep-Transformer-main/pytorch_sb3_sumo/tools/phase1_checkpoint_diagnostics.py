"""Additional, evaluation-only phase-1 diagnostics. Never trains or selects models."""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results_phase1_checkpoint_diagnostics_v1'
SOURCE = ROOT / 'results_iv2_5m6s100e_v1/comparison/summary/method_scenario.csv'
PROTOCOL = ROOT / 'experiments/independent_v2_five_methods_six_scenarios_100ep_v1/protocol.json'
STEPS = [10000,20000,30000,40000,50000]

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x,indent=2),encoding='utf-8'); tmp.replace(p)
def rel(p): return Path(p).resolve().relative_to(ROOT).as_posix()
def archive_signature(p):
    # Compare actual serialized model contents, ignoring ZIP timestamps only.
    with zipfile.ZipFile(p) as z:
        return {n:hashlib.sha256(z.read(n)).hexdigest() for n in sorted(z.namelist()) if n=='data' or n.endswith('.pth')}
def cells():
    with SOURCE.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def prepare():
    jobs=[]; aliases=[]; sources=[]
    cc=cells(); assert len(cc)==30
    for c in cc:
        run=Path(c['run_directory']); original=run/'paper_evaluation_detailed.json'; d=read(original)
        assert sha(original)==c['detailed_evaluation_sha256']
        assert d['evaluation_seed_start']==10000 and d['summary']['episodes']==100
        assert [r['seed'] for r in d['episode_records']]==list(range(10000,10100))
        assert d['evaluation_provenance']['validated'] is True
        evaluated_model=run/Path(d['model']).name
        if d.get('model_sha256'):
            assert sha(evaluated_model)==d['model_sha256']
        else:
            receipts=[run/'i5m6s100_job_receipt.json',run/'high_density_job_receipt.json']
            receipt=next(p for p in receipts if p.is_file())
            assert read(receipt)['artifact_sha256'][evaluated_model.name]==sha(evaluated_model)
        decoder=d.get('selected_deployment_decoder','native_deterministic')
        paths={'exact_final':run/'final_model.zip'}
        for step in STEPS:
            match=list((run/'checkpoints').glob(f'*_raw_{step}_steps.zip')); assert len(match)==1
            paths[f'raw_{step}']=match[0]
        prior_signature=archive_signature(run/Path(d['model']).name)
        canonical={}
        for kind,path in paths.items():
            sig=archive_signature(path); signature=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest()
            jobid=f"{c['method']}__{c['scenario']}__{kind}"
            if signature in canonical:
                aliases.append(dict(job=jobid,target=canonical[signature],kind=kind,method=c['method'],scenario=c['scenario'],checkpoint=rel(path),checkpoint_sha256=sha(path)))
                continue
            canonical[signature]=jobid
            reuse=(sig==prior_signature or (kind=='exact_final' and d.get('selected_checkpoint_kind')=='exact_final' and d.get('selected_source_checkpoint_sha256')==sha(path)))
            jobs.append(dict(id=jobid,method=c['method'],scenario=c['scenario'],kind=kind,raw_steps=50000 if kind=='exact_final' else int(kind[4:]),checkpoint=rel(path),checkpoint_sha256=sha(path),archive_content_signature=signature,run=rel(run),decoder=decoder,original_evaluation=rel(original),original_sha256=sha(original),reuse_original=reuse,source=c['source']))
        sources.append(dict(method=c['method'],scenario=c['scenario'],run=rel(run),original_evaluation_sha256=sha(original),arguments_sha256=sha(run/'arguments.json'),monitor_sha256=sha(run/'train_monitor.csv'),original_checkpoint=d.get('selected_checkpoint_kind','exact_final'),decoder=decoder))
    payload=dict(schema='phase1-development-diagnostics/v1',created_at=time.time(),source_summary=rel(SOURCE),source_summary_sha256=sha(SOURCE),protocol=rel(PROTOCOL),protocol_sha256=sha(PROTOCOL),training=False,model_selection=False,formal_test_accessed=False,scope='Development diagnostics on existing evaluation partition; not untouched final-test evidence.',evaluation_seed_start=10000,episodes=100,block_episodes=20,workers=2,decoder_policy='Keep original sealed deployment decoder fixed per cell at every checkpoint; this is retrospective decoder-conditioned diagnostics, not an online selection curve.',jobs=jobs,aliases=aliases,sources=sources)
    p=OUT/'plan.json'
    if p.exists():
        old=read(p)
        assert old['jobs']==jobs and old['aliases']==aliases and old['protocol_sha256']==payload['protocol_sha256'], 'Existing phase1 plan differs'
        return old
    write(p,payload)
    (OUT/'PLAN.md').write_text('''# 第一阶段：已有模型检查点诊断

新增实验，保留原模型、配置及结果；不创建额外快照，不训练、不调参、不重新选择模型。

- 五方法、六场景、训练 seed 0。
- exact-final 对照与 10k/20k/30k/40k/50k 原始步检查点评估。
- 每模型 100 回合，种子 10000–10099；每 20 回合落盘，可续跑。
- 两个并行评估 worker，使用原高车流参数、物理 evaluation 分区与确定性部署。
- v4.8 的解码器按原选择记录固定，v4.13 同样保持原 target_critic；不在各检查点上重新选择解码器。
- 因解码器来自原完整训练后的选择，此曲线属于回顾性、固定解码器诊断，不能解释为当时可部署策略或样本效率证明。
- 原结果只在模型内容相同或 exact-final 源哈希绑定成立时复用；同内容周期模型与 final 共用评估。
- 结果为开发诊断，禁止用作未参与开发的最终测试证据。
- 输出：逐回合数据、检查点表、原流程与最后模型对照、两种指标曲线及阶段诊断报告。

重点：Cross 早期模型优势、Roundabout-A/C 退步、Roundabout-B 失败与 CARLA 强基线；全部六场景保留。
''',encoding='utf-8')
    return payload

def validate_report(d,n,seed):
    records=d['episode_records']; assert len(records)==n
    assert [r['seed'] for r in records]==list(range(seed,seed+n))
    assert d['summary']['episodes']==n
    assert all(r['raw_steps']>0 and r['traffic_variant'] for r in records)
    for key,field in [('success_rate','success'),('collision_rate','collision'),('off_route_rate','off_route'),('timeout_rate','timeout')]:
        assert abs(d['summary'][key]-sum(r[field] for r in records)/n)<1e-10
    assert abs(d['summary']['mean_return']-sum(r['episode_return'] for r in records)/n)<1e-9

def summarize(records):
    import numpy as np
    n=len(records); success=[r['completion_time_seconds'] for r in records if r['success']]
    return dict(summary=dict(episodes=n,mean_return=float(np.mean([r['episode_return'] for r in records])),std_return=float(np.std([r['episode_return'] for r in records])),mean_decision_steps=float(np.mean([r['decision_steps'] for r in records])),mean_raw_steps=float(np.mean([r['raw_steps'] for r in records])),**{key:sum(r[field] for r in records)/n for key,field in [('success_rate','success'),('collision_rate','collision'),('off_route_rate','off_route'),('timeout_rate','timeout')]}),mean_success_completion_time_seconds=float(np.mean(success)) if success else None,episode_records=records)

def make_runtime(job,device):
    import torch
    torch.set_num_threads(1)
    from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory
    requested=read(ROOT/job['run']/'arguments.json'); requested=requested.get('requested_raw_steps',requested)
    args=argparse.Namespace(**requested); args.gui=False; args.evaluation_split='validation'
    protocol=read(PROTOCOL); adapter='base' if job['method'] not in ('v4_8','v4_13') else job['method']
    # Short job namespace avoids Windows MAX_PATH on roundabout_medium manifests.
    overlay_id=hashlib.sha256(job['id'].encode()).hexdigest()[:12]
    factory=_make_environment_factory(adapter=adapter,density=protocol['scenarios'][job['scenario']],overlay_root=OUT/'ov'/overlay_id)
    env=factory(args,evaluation=True)
    try:
        if adapter=='base':
            from algos.sb3_torch.sac import SceneRepresentationSAC
            model=SceneRepresentationSAC.load(ROOT/job['checkpoint'],env=env,device=device)
        elif adapter=='v4_8':
            from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45
            from tools.action_diagnostics_v4_6 import load_model_for_deployment
            model=load_model_for_deployment(ConfidentActorFusionSACV45,ROOT/job['checkpoint'],decoder=job['decoder'],env=env,device=device)
        else:
            from tools.action_diagnostics_v4_13_model import load_model_for_deployment_v4_13,GradientIsolatedTemperedJointSupportSACV413
            model=load_model_for_deployment_v4_13(GradientIsolatedTemperedJointSupportSACV413,ROOT/job['checkpoint'],decoder=job['decoder'],env=env,device=device)
        from tools.paper_evaluation_contract import validate_model_environment_spaces
        validate_model_environment_spaces(model,env)
        actual=int(model._raw_steps_seen); assert actual==job['raw_steps'],(actual,job['raw_steps'])
        return model,env
    except BaseException:
        env.close(); raise

def worker(jobid,device='cuda',smoke=False):
    plan=read(OUT/'plan.json'); job=next(j for j in plan['jobs'] if j['id']==jobid)
    assert sha(ROOT/job['checkpoint'])==job['checkpoint_sha256']
    assert sha(ROOT/job['original_evaluation'])==job['original_sha256']
    directory=OUT/('smoke' if smoke else 'jobs')/jobid; directory.mkdir(parents=True,exist_ok=True)
    dest=directory/'result.json'
    if dest.exists():
        d=read(dest); assert d['checkpoint_sha256']==job['checkpoint_sha256']; validate_report(d,1 if smoke else 100,10000); return
    if job['reuse_original'] and not smoke:
        d=read(ROOT/job['original_evaluation']); validate_report(d,100,10000)
        write(dest,dict(**job,**summarize(d['episode_records']),origin='reused_original',actual_raw_steps=job['raw_steps']))
        return
    model,env=make_runtime(job,device)
    try:
        from algos.sb3_torch import evaluate_model_detailed
        all_records=[]
        for offset in ([0] if smoke else range(0,100,20)):
            block=directory/f'block_{offset:03d}.json'; n=1 if smoke else 20
            if block.exists():
                d=read(block); assert d['checkpoint_sha256']==job['checkpoint_sha256']
            else:
                started=time.time()
                d=evaluate_model_detailed(model,env,episodes=n,seed=10000+offset,deterministic=True,sumo_step_seconds=.1,policy_action_hold=1).to_dict()
                d.update(checkpoint_sha256=job['checkpoint_sha256'],decoder=job['decoder'],wall_seconds=time.time()-started)
                validate_report(d,n,10000+offset)
                write(block,d)
            validate_report(d,n,10000+offset)
            for r in d['episode_records']: r['episode']=r['seed']-10000
            original=read(ROOT/job['original_evaluation'])['episode_records']
            assert [r['traffic_variant'] for r in d['episode_records']]==[r['traffic_variant'] for r in original[offset:offset+n]], 'Traffic schedule mismatch'
            all_records.extend(d['episode_records'])
            write(directory/'progress.json',dict(completed_episodes=len(all_records),updated_at=time.time()))
        if smoke and job['reuse_original']:
            old=read(ROOT/job['original_evaluation'])['episode_records'][0]; now=all_records[0]
            for field in ['success','collision','off_route','timeout','raw_steps','episode_return','traffic_variant']:
                assert now[field]==old[field],(field,now[field],old[field])
        write(dest,dict(**job,**summarize(all_records),origin='fresh_rollout',actual_raw_steps=int(model._raw_steps_seen)))
    finally: env.close()

def protected_worker(jobid,device='cuda',smoke=False):
    if smoke:return worker(jobid,device,True)
    import psutil
    directory=OUT/'jobs'/jobid;directory.mkdir(parents=True,exist_ok=True)
    lock=directory/'worker.lock'
    while True:
        if (directory/'result.json').exists():return worker(jobid,device,False)
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            os.write(fd,str(os.getpid()).encode());os.close(fd);break
        except FileExistsError:
            try:
                pid=int(lock.read_text())
                if not psutil.pid_exists(pid):lock.unlink(missing_ok=True);continue
            except ValueError:pass
            time.sleep(2)
    try:return worker(jobid,device,False)
    finally:lock.unlink(missing_ok=True)

def run(workers=6,supplement=False,pool_name='supplement',retry_failed=False):
    plan=prepare(); OUT.mkdir(exist_ok=True)
    lock=OUT/(f'{pool_name}.lock' if supplement else 'controller.lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        jobs=sorted(plan['jobs'],key=lambda j:(j['kind']!='exact_final',not j['reuse_original'],j['scenario'],j['method'],j['raw_steps']))
        if retry_failed:
            failed=set(read(OUT/'supplement_status.json')['failed'])
            jobs=[j for j in jobs if j['id'] in failed]
        import psutil
        inherited={}
        for proc in psutil.process_iter(['pid','name','cmdline']):
            try:
                cmd=proc.info['cmdline'] or []
                if proc.pid!=os.getpid() and proc.info['name'].lower().startswith('python') and 'worker' in cmd and '--job' in cmd and any(str(x).endswith('phase1_checkpoint_diagnostics.py') for x in cmd):
                    inherited[cmd[cmd.index('--job')+1]]=proc.pid
            except psutil.Error:pass
        if supplement:
            inherited={};jobs.reverse()
        else:jobs.sort(key=lambda j:j['id'] not in inherited)
        status=dict(status='running',pid=os.getpid(),workers=workers,started_at=time.time(),completed=[],failed=[],inherited_workers=inherited)
        def dispatch(j):
            if j['id'] in inherited:
                try: psutil.Process(inherited[j['id']]).wait()
                except psutil.NoSuchProcess: pass
                existing=OUT/'jobs'/j['id']/'result.json'
                if existing.exists():return j['id'],0
            log=OUT/'logs'/f"{j['id']}.log"; log.parent.mkdir(exist_ok=True)
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            with log.open('a',encoding='utf-8') as f:
                p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--job',j['id']],stdout=f,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            return j['id'],p.returncode
        status_path=OUT/(f'{pool_name}_status.json' if supplement else 'status.json')
        write(status_path,status)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(dispatch,j) for j in jobs]
            for future in concurrent.futures.as_completed(futures):
                jobid,rc=future.result();status['completed' if rc==0 else 'failed'].append(jobid)
                status['updated_at']=time.time(); write(status_path,status)
        status['status']='summarizing' if not status['failed'] else 'needs_attention';write(status_path,status)
        if supplement:
            status['status']='completed' if not status['failed'] else 'needs_attention';write(status_path,status);return
        summary_script=Path(__file__).with_name('summarize_phase1_checkpoint_diagnostics.py')
        with (OUT/'summary.log').open('a',encoding='utf-8') as f:
            result=subprocess.run([sys.executable,str(summary_script)],stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        status['summary_returncode']=result.returncode
        status['status']='completed' if not status['failed'] and result.returncode==0 else 'needs_attention'
        status['finished_at']=time.time();write(status_path,status)
    finally: lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','smoke','run']);p.add_argument('--job');p.add_argument('--device',default='cuda');p.add_argument('--workers',type=int,default=6);p.add_argument('--supplement',action='store_true');p.add_argument('--pool',choices=['supplement','repair'],default='supplement');p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    if a.mode=='prepare':
        x=prepare();print(json.dumps(dict(jobs=len(x['jobs']),aliases=len(x['aliases']),reusable=sum(j['reuse_original'] for j in x['jobs']))))
    elif a.mode=='run':
        if a.workers<1:raise ValueError('workers must be positive')
        run(a.workers,a.supplement,a.pool,a.retry_failed)
    else:protected_worker(a.job,a.device,a.mode=='smoke')
