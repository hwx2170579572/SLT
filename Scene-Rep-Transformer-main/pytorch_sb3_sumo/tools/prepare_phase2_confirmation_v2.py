"""Prepare development confirmation from the reference-based screen; never dispatch."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.phase1_checkpoint_diagnostics import read, write, sha
from tools.analyze_phase2_reuse_v3 import main as analyze_screen, OUT, CONTRACT
from tools.control_phase2_screen_v2 import jobs
from tools.prepare_phase2_confirmation_v1 import confirmation_jobs


def require_complete(completion):
    if (completion.get('complete') is not True or completion.get('resolved') != 66
            or completion.get('expected') != 66 or completion.get('historical_references') != 18
            or completion.get('new_variants') != 48 or completion.get('missing')
            or completion.get('invalid')):
        raise ValueError('All 66 validated reference-based screen cells required')


def validate_provenance(contract, decision, evidence, contract_hash, manifest_hash, evidence_hash):
    for value in (decision, evidence):
        if value.get('contract_sha256') != contract_hash or value.get('manifest_sha256') != manifest_hash:
            raise ValueError('Contract or execution manifest mismatch')
    if decision.get('evidence_sources_sha256') != evidence_hash:
        raise ValueError('Decision provenance mismatch')
    expected = {j['id']:j for j in jobs(contract)}
    if set(evidence['sources']) != set(expected):
        raise ValueError('Incomplete evidence coverage')
    for key, job in expected.items():
        source = evidence['sources'][key]
        origin = 'historical_reference' if job['candidate']=='control' else 'new_variant'
        if source['origin'] != origin:
            raise ValueError('Evidence origin mismatch')
        if sha(ROOT/source['path']) != source['sha256']:
            raise ValueError('Evidence changed: '+key)


def main():
    target = ROOT/'results_phase2_confirmation_v2/proposed_manifest.json'
    if target.exists():
        raise FileExistsError('Proposed manifest already exists; preserve selected candidates')
    analyze_screen()  # Recompute the decision from fully validated original artifacts.
    require_complete(read(OUT/'analysis/completion.json'))
    contract = read(CONTRACT)
    decision_path = OUT/'analysis/screen_decision.json'
    evidence_path = OUT/'analysis/evidence_sources.json'
    decision, evidence = read(decision_path), read(evidence_path)
    validate_provenance(contract, decision, evidence, sha(CONTRACT),
                        sha(OUT/'execution_manifest.json'), sha(evidence_path))
    selected = decision['selected_for_confirmation']
    tasks = confirmation_jobs(contract, selected)
    write(target, dict(schema='phase2-confirmation-proposal/v2', status='reuse_audit_required',
        launch_allowed=False, contract_sha256=sha(CONTRACT), screen_decision_sha256=sha(decision_path),
        screen_evidence=evidence, selected_candidates=selected, proposed_jobs=tasks,
        expected_cells=len(tasks), training_seeds=contract['confirmation_training_seeds'],
        maximum_raw_steps_before_reuse=sum(j['raw_steps'] for j in tasks),
        development_only=True, final_test=False,
        selection_protocol='Inherited decoder; exact final; seed 0 screening, seeds 1 and 2 confirmation; no re-ranking',
        required_before_dispatch='Audit existing seed 1/2 training and evaluation artifacts for exact matching reuse; then prepare a missing-only execution manifest',
        limitation='Historical control training-code equivalence is incomplete; this does not implement independent validation selection or final paper testing'))
    print(f'Prepared {len(tasks)} proposed cells; reuse audit required; training not started')


if __name__ == '__main__':
    main()
