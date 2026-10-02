import pytest
from tools.diagnose_phase2_paired_outcomes_v1 import paired


def record(seed,success,time,collision=False):
    return dict(seed=seed,traffic_variant='a',success=success,completion_time_seconds=time,collision=collision)


def test_time_excludes_gained_and_lost_success():
    a=[record(1,True,10),record(2,False,100,True),record(3,True,20)]
    b=[record(1,True,12),record(2,True,500),record(3,False,100,True)]
    r=paired(a,b)
    assert r['both_success']==r['lost_success']==r['gained_success']==1
    assert r['paired_mean_delta_seconds']==2
    assert r['paired_relative_increase']==pytest.approx(.2)
    assert r['collision_to_success']==r['success_to_collision']==1


def test_no_shared_success_has_no_time_estimate():
    r=paired([record(1,False,4)],[record(1,True,8)])
    assert r['paired_mean_delta_seconds'] is None
    assert r['paired_relative_increase'] is None


def test_pair_order_and_duplicates_rejected():
    a=[record(1,True,2),record(2,True,3)]
    with pytest.raises(ValueError):paired(a,a[::-1])
    with pytest.raises(ValueError):paired([a[0],a[0]],[a[0],a[0]])
