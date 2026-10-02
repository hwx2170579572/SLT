"""Generate an immutable confirmation manifest only after the whole screen passes validation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.phase1_checkpoint_diagnostics import read,write,sha
from tools.control_phase2_screen_v2 import CONTRACT,OUT,jobs,validate_finished
from tools.analyze_phase2_screen_v2 import main as analyze_screen


def confirmation_jobs(contract,selection):
    if set(selection)!=set(contract['optimization_methods']):
        raise ValueError('Selection must cover exactly the two optimization methods')
    candidates={c['id']:c for c in contract['candidates']}
    if any(v not in candidates for v in selection.values()):raise ValueError('Unknown selected candidate')
    result=[]
    for method in contract['methods']:
        variants=['control']
        if method in selection and selection[method]!='control':variants.append(selection[method])
        for variant in variants:
            for scene in contract['scenarios']:
                for seed in contract['confirmation_training_seeds']:
                    result.append(dict(id=f'{method}__{scene}__{variant}__seed{seed}',method=method,
                        scenario=scene,candidate=candidates[variant],seed=seed,
                        raw_steps=contract['raw_steps_per_run'],decoder=contract['decoder_per_cell'][f'{method}__{scene}'],
                        evaluation=contract['evaluation']))
    if len(result)>contract['confirmation_max_cells']:raise ValueError('Confirmation budget exceeded')
    if len({r['id'] for r in result})!=len(result):raise ValueError('Duplicate confirmation job')
    return result


def require_complete(completion):
    if not completion.get('complete') or completion.get('completed')!=66 or completion.get('expected')!=66 or completion.get('missing') or completion.get('invalid'):
        raise ValueError('All 66 valid screen cells required; no early confirmation selection')


def main():
    target=ROOT/'results_phase2_confirmation_v1/manifest.json'
    if target.exists():raise FileExistsError('Manifest is immutable; do not replace selected candidates')
    analyze_screen()
    require_complete(read(OUT/'analysis/completion.json'))
    contract=read(CONTRACT);decision_path=OUT/'analysis/screen_decision.json';decision=read(decision_path)
    if decision['contract_sha256']!=sha(CONTRACT):raise ValueError('Screen decision uses a different contract')
    evidence={}
    for j in jobs(contract):
        directory=OUT/'screen'/j['id'];validate_finished(directory)
        for name in ['arguments.json','evaluation.json','final_model.zip','training_complete.json']:
            p=directory/name;evidence[str(p.relative_to(ROOT))]=sha(p)
    selected=decision['selected_for_confirmation']
    tasks=confirmation_jobs(contract,selected)
    write(target,dict(schema='phase2-confirmation-manifest/v1',status='prepared_not_started',
        contract_sha256=sha(CONTRACT),screen_decision_sha256=sha(decision_path),screen_evidence_sha256=evidence,
        selected_candidates=selected,jobs=tasks,training_seeds=contract['confirmation_training_seeds'],
        expected_cells=len(tasks),maximum_raw_steps=sum(j['raw_steps'] for j in tasks),
        rule='Report every seed; no candidate re-ranking using confirmation results; seed 0 remains development screening',
        unchanged='Architecture, loss, inherited deployment decoder and exact-final selection remain fixed',
        final_test=False,baseline_tuning_budget='Default only; unequal screen tuning budget must be disclosed'))
    print(f'Prepared {len(tasks)} confirmation cells; no training started')


if __name__=='__main__':main()
