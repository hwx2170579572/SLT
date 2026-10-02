import json
import pytest
from tools.control_phase2_screen_v1 import jobs, validate_finished, CONTRACT


def test_frozen_matrix_has_all_candidates_and_no_excluded_method():
    c=json.loads(CONTRACT.read_text())
    rows=jobs(c)
    assert len(rows)==120 and len({r['id'] for r in rows})==120
    assert {r['method'] for r in rows}=={'mst_slt','temporal_graph','v4_8','v4_13'}
    for method in c['methods']:
        for candidate in c['candidates']:
            assert len([r for r in rows if r['method']==method and r['candidate']==candidate['id']])==6


def test_smoke_cannot_count_as_completed_screen(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'status':'completed','smoke':True}))
    with pytest.raises(ValueError,match='completed screen'):
        validate_finished(tmp_path)


def test_failed_training_cannot_count_as_completed_screen(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'status':'failed','smoke':False}))
    with pytest.raises(ValueError,match='completed screen'):
        validate_finished(tmp_path)
