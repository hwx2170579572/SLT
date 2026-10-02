import pytest
from tools import prepare_phase2_confirmation_v2 as p


@pytest.mark.parametrize('change', [dict(resolved=65), dict(historical_references=17),
    dict(new_variants=47), dict(complete=False), dict(missing=['a']), dict(invalid=['a'])])
def test_incomplete_reuse_cannot_unlock(change):
    state=dict(complete=True,resolved=66,expected=66,historical_references=18,new_variants=48,missing=[],invalid=[])
    state.update(change)
    with pytest.raises(ValueError, match='66 validated'):
        p.require_complete(state)


def test_valid_reference_completion():
    p.require_complete(dict(complete=True,resolved=66,expected=66,historical_references=18,new_variants=48,missing=[],invalid=[]))


@pytest.mark.parametrize('mutation', ['decision_hash','coverage','origin','artifact'])
def test_provenance_rejects_stale_or_mislabelled_evidence(monkeypatch, mutation):
    monkeypatch.setattr(p, 'jobs', lambda c:[dict(id='one',candidate='control')])
    monkeypatch.setattr(p, 'sha', lambda path:'ok')
    evidence=dict(contract_sha256='c',manifest_sha256='m',sources={'one':dict(origin='historical_reference',path='unused',sha256='ok')})
    decision=dict(contract_sha256='c',manifest_sha256='m',evidence_sources_sha256='e')
    if mutation=='decision_hash': decision['evidence_sources_sha256']='stale'
    if mutation=='coverage': evidence['sources']={}
    if mutation=='origin': evidence['sources']['one']['origin']='new_variant'
    if mutation=='artifact': evidence['sources']['one']['sha256']='changed'
    with pytest.raises(ValueError):
        p.validate_provenance({},decision,evidence,'c','m','e')
