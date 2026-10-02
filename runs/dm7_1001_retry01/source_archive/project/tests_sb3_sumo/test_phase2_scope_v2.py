import json
import pytest
from tools.control_phase2_screen_v2 import jobs, CONTRACT, run_job


def test_three_method_scope_and_fixed_baseline():
    c=json.loads(CONTRACT.read_text())
    rows=jobs(c)
    assert len(rows)==66 and len({r['id'] for r in rows})==66
    assert {r['method'] for r in rows}=={'mst_slt','v4_8','v4_13'}
    baseline=[r for r in rows if r['method']=='mst_slt']
    assert len(baseline)==6
    assert {r['candidate'] for r in baseline}=={'control'}
    for method in ('v4_8','v4_13'):
        for candidate in c['candidates']:
            assert len([r for r in rows if r['method']==method and r['candidate']==candidate['id']])==6
    assert len(c['decoder_per_cell'])==18


def test_pause_does_not_start_process(tmp_path,monkeypatch):
    import tools.control_phase2_screen_v2 as control
    monkeypatch.setattr(control,'OUT',tmp_path)
    (tmp_path/'pause_new_jobs.json').write_text('{}')
    monkeypatch.setattr(control.subprocess,'Popen',lambda *a,**kw:pytest.fail('Paused queue started subprocess'))
    assert run_job({'id':'never_started'},'cpu')['status']=='paused_before_start'
