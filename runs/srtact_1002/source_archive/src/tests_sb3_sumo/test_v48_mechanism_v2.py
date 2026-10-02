"""Boundary and grouping tests that prevent invalid mechanism evidence."""
import numpy as np

from tools.v48_mechanism_v2.common import sha256

from tools.v48_mechanism_v2.control_data import future_labels
from tools.v48_mechanism_v2.probes import grouped_split, standardize, paired_error_interval
from tools.v48_mechanism_v2.gradient_diagnostics import nstep_targets,soft_bootstrap_value
from tools.v48_mechanism_v2.branches import ranking_accuracy


def test_fingerprint_accepts_string_and_path_equally(tmp_path):
    path=tmp_path / "fingerprint.txt"
    path.write_text("fixed diagnostic evidence",encoding="utf-8")
    assert sha256(path)==sha256(str(path))


def test_future_labels_stop_at_terminal_and_censor_missing_position():
    data=dict(reward=np.array([0.,0.,-1.]),collision=np.array([False,False,True]),
        timeout=np.zeros(3,dtype=bool),success=np.zeros(3,dtype=bool),
        distance_start=np.array([0.,1.,2.]),distance_end=np.array([1.,2.,2.]),
        distance_end_observed=np.array([True,True,False]),raw_executed=np.array([3,3,1]))
    labels=future_labels(data,horizons=(1,4))
    np.testing.assert_array_equal(labels["collision_h1"],[False,False,True])
    np.testing.assert_array_equal(labels["collision_h4"],[True,True,True])
    np.testing.assert_array_equal(labels["actual_h_h4"],[3,2,1])
    np.testing.assert_allclose(labels["seconds_h4"],[.7,.4,.1])
    assert np.isnan(labels["progress_m_h4"]).all()
    assert labels["progress_m_h1"][0]==1.


def test_policy_and_branch_rows_never_leak_across_seed_split():
    groups=np.tile(np.repeat(np.arange(40),5),3)
    masks,receipt=grouped_split(groups)
    for group in np.unique(groups):
        assert sum(bool(mask[groups==group].all()) for mask in masks)==1
    assert [len(receipt[k]) for k in ("fit_groups","validation_groups","test_groups")]==[22,6,12]


def test_standardization_does_not_fit_test_outlier():
    x=np.array([[0.],[2.],[100.]])
    got=standardize(x,np.array([True,True,False]))
    np.testing.assert_allclose(got[:,0],[-1.,1.,99.])


def test_paired_interval_has_known_direction_and_group_count():
    y=np.zeros((12,1))
    groups=np.repeat(np.arange(3),4)
    result=paired_error_interval(y,np.ones_like(y),np.zeros_like(y),groups,np.ones(12,dtype=bool),np.ones(1))
    assert result["groups"]==3 and result["second_minus_first"]==-1.
    assert result["ci95"]==[-1.,-1.]


def test_diagnostic_nstep_never_crosses_episodes_and_preserves_truncation_bootstrap():
    data=dict(episode_id=np.array([0,0,1,1,1]),reward=np.array([0.,1.,5.,7.,9.]),
              terminated=np.array([False,False,False,False,True]),raw_executed=np.array([3,1,3,3,2]))
    target=nstep_targets(data,np.array([0,1,2]),16,gamma=.5)
    np.testing.assert_array_equal(target["actual_h"],[2,1,3])
    np.testing.assert_allclose(target["reward"],[.5,1.,10.75])
    np.testing.assert_allclose(target["discount"],[.25,.5,.125])
    np.testing.assert_array_equal(target["done"],[0.,0.,1.])


def test_branch_ranking_excludes_ties_and_distinct_snapshots():
    y=np.array([0.,1.,1.,100.,100.])
    prediction=np.array([[0.],[2.],[3.],[-3.],[-2.]])
    groups=np.array(["a","a","a","b","b"])
    result=ranking_accuracy(y,prediction,groups,np.ones(5,dtype=bool),.01)
    assert result["eligible_pairs"]==2 and result["pairwise_accuracy"]==1.


def test_full_td_uses_native_factorized_speed_entropy_not_joint_entropy():
    import types
    import torch
    from algos.sb3_torch.sac_v4_2 import FactorizedEntropyHybridSACV42
    batch=types.SimpleNamespace(actions=torch.zeros(1,3,2),lane_probabilities=torch.tensor([[.2,.3,.5]]),
        speed_log_probabilities=torch.tensor([[-1.,-2.,-3.]]),lane_log_probabilities=torch.tensor([[-4.,-5.,-6.]]),
        joint_log_probabilities=torch.tensor([[-5.,-7.,-9.]]))
    model=types.SimpleNamespace(lane_entropy_scale=0.,log_ent_coef=torch.tensor(np.log(.5)),
        actor=types.SimpleNamespace(all_action_samples=lambda obs,deterministic_speed:batch),
        critic_target=types.SimpleNamespace(features_extractor=lambda obs:torch.zeros(1,128),
            all_q_from_features=lambda feature,action:(torch.full((1,3,1),2.),torch.full((1,3,1),3.))))
    model.entropy_log_probabilities=types.MethodType(FactorizedEntropyHybridSACV42.entropy_log_probabilities,model)
    torch.testing.assert_close(soft_bootstrap_value(model,{}),torch.tensor([[3.15]]))
