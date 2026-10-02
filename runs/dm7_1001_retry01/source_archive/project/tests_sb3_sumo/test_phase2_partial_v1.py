import pytest
from tools import diagnose_phase2_partial_v1 as d


def test_incomplete_candidate_has_no_macro_or_guard(monkeypatch):
    scenes = ['a', 'b', 'c', 'd', 'e', 'f']
    contract = dict(methods=['v4_8'], candidates=[{'id':'control'}, {'id':'trial'}], scenarios=scenes)
    sources = {f'v4_8__{s}__control__seed0': {'path':s, 'sha256':'ok'} for s in scenes}
    sources['v4_8__a__trial__seed0'] = {'path':'trial', 'sha256':'ok'}
    monkeypatch.setattr(d, 'sha', lambda p:'ok')
    monkeypatch.setattr(d, 'read', lambda p:dict(summary=dict(success_rate=.9, collision_rate=.1, timeout_rate=0)))
    monkeypatch.setattr(d, 'candidate_guards', lambda *a:pytest.fail('Incomplete candidate evaluated'))
    rows = d.describe(contract, sources)
    assert rows[0]['metrics']['success_rate'] == pytest.approx(.9)
    assert len(rows[1]['missing']) == 5
    assert rows[1]['metrics'] is None and rows[1]['guards'] is None
    assert len(rows[1]['scenes']) == 1


def test_changed_source_rejected(monkeypatch):
    monkeypatch.setattr(d, 'sha', lambda p:'changed')
    with pytest.raises(ValueError, match='Evidence changed'):
        d.describe({}, {'cell':{'path':'unused', 'sha256':'original'}})
