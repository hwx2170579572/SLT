"""Seal current source and input identities after real three-method smoke checks."""
import sys
import platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.phase1_checkpoint_diagnostics import read,write,sha


def main():
    import torch
    import stable_baselines3
    out=ROOT/'results_phase2_runtime_v2'
    target=out/'preflight_v2.json'
    if target.exists():raise FileExistsError('Preflight is immutable; create a new version for changes')
    contract=ROOT/'results_phase2_diagnosis_20260908/tuning_contract_v2.json'
    c=read(contract); evidence={}
    for file in sorted((out/'smoke').glob('*/arguments.json')):
        a=read(file);s=read(file.parent/'status.json')
        if a.get('runner_sha256')!=sha(ROOT/'tools/run_phase2_cell_v2.py'):continue
        if a.get('model_factory_sha256')!=sha(ROOT/'tools/phase2_model_factory_v1.py'):continue
        if s['status']=='completed' and s['raw_steps']==72 and s['learner_updates']==25 and s['policy_tensor_roundtrip_equal']:
            evidence[a['method']]={'path':str(file.parent.relative_to(ROOT)), 'status_sha256':sha(file.parent/'status.json')}
    if set(evidence)!=set(c['methods']):raise ValueError(f'Current-runner smoke coverage incomplete: {list(evidence)}')
    paths=set()
    for directory in ['algos','configs','envs','tools']:
        paths.update(p for p in (ROOT/directory).rglob('*.py') if '__pycache__' not in p.parts)
    for p in (ROOT/'envs/sumo').rglob('*'):
        if p.is_file() and p.suffix in ('.xml','.json','.yaml'):paths.add(p)
    plan=ROOT/'results_phase1_checkpoint_diagnostics_v1/plan.json'; paths.add(plan)
    p=read(plan);paths.add(ROOT/p['protocol'])
    for s in p['sources']:
        if s['method'] in c['methods']:
            paths.add(ROOT/s['run']/'arguments.json');paths.add(ROOT/s['run']/'paper_evaluation_detailed.json')
    write(target,dict(passed=True,contract_sha256=sha(contract),smoke_evidence=evidence,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},
        runtime={'python':platform.python_version(),'torch':torch.__version__,'sb3':stable_baselines3.__version__},
        control_reuse=False,training_recovery='Never continue model-only checkpoints without replay/RNG; preserve interrupted output for explicit recovery',
        terminal_semantics='Preserve independent success and max_time flags; no reclassification',
        evidence_role='development screen; not untouched final test'))
    print(f'Sealed {len(paths)} source/input files and three current-runner smoke receipts')


if __name__=='__main__':main()
