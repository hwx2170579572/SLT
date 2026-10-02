"""Synthetic unit fixtures only; never used as experiment results."""
import copy
import pytest
from tools.analyze_phase2_screen_v2 import candidate_guards,CONTRACT,read


def cell(success,time=10):
    return {'summary':{'success_rate':success/100,'collision_rate':(100-success)/100,'timeout_rate':0},
            'episode_records':[{'seed':i,'traffic_variant':str(i),'success':i<success,'completion_time_seconds':time} for i in range(100)]}


def limits():return read(CONTRACT)['promotion_guards_relative_to_own_control']


def test_better_success_and_safety_pass():
    assert candidate_guards([cell(90)]*6,[cell(92)]*6,limits())['passed']


def test_aggregate_gain_cannot_hide_scene_regression():
    d=candidate_guards([cell(90)]*6,[cell(86)]+[cell(100)]*5,limits())
    assert not d['passed'] and 'scene_0_success' in d['reasons']


def test_efficiency_or_insufficient_overlap_cannot_pass():
    assert not candidate_guards([cell(90)]*6,[cell(92,11.1)]*6,limits())['passed']
    assert not candidate_guards([cell(10)]*6,[cell(12)]*6,limits())['passed']


def test_misaligned_traffic_is_rejected():
    b=cell(92);b['episode_records'][0]['traffic_variant']='wrong'
    with pytest.raises(ValueError,match='pairing'):
        candidate_guards([cell(90)]*6,[b]*6,limits())
