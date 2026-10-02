import pytest
from tools.control_phase2_missing_v3 import missing_jobs,read,OUT


def test_real_inventory_has_no_repeated_controls():
    inv=read(OUT/'inventory.json');jobs=missing_jobs(inv)
    assert len(inv['controls'])==18 and len(jobs)==48
    assert all(j['candidate']!='control' for j in jobs)


def test_unknown_coverage_blocks_execution():
    with pytest.raises(ValueError,match='errors'):
        missing_jobs({'scan_errors':['unreadable'],'proposed_missing_candidates':[]})


def test_injected_control_is_rejected():
    with pytest.raises(ValueError,match='Repeated controls'):
        missing_jobs({'scan_errors':[],'proposed_missing_candidates':[{'candidate':'control','method':'mst_slt'}]})
