from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from tests_sb3_sumo.test_gradient_isolated_tempered_support_v4_13 import _Env, _small_model
from tools.phase3_mechanism_v1.common import model_state_sha
from tools.phase3_mechanism_v1.evaluation import apply_decoder, discounted_collision_returns, selection_key
from tools.phase3_mechanism_v1.gradients import gradient_metrics
from tools.phase3_mechanism_v1.model import VARIANTS, apply_training_variant, assert_variant
from tools.phase3_mechanism_v1.traffic import jitter_document


def batch():
    return {'state': torch.tensor([[.1,.2,.3,.4],[.4,.3,.2,.1],[-.2,.1,.4,-.3],[.5,-.4,.3,-.2]]),
            'lane_action_mask': torch.ones(4,3)}, torch.tensor([[0.,0.],[.2,1.],[.1,-1.],[.3,0.]])


@pytest.mark.parametrize('variant', tuple(VARIANTS))
def test_factorial_paths_and_roundtrip(variant, tmp_path):
    torch.set_num_threads(1)
    env = _Env()
    model = _small_model(env)
    before = model_state_sha(model)
    apply_training_variant(model, variant)
    assert model_state_sha(model) == before
    observations, actions = batch()
    diagnostics = gradient_metrics(model, observations, actions)
    assert diagnostics['unisolated_counterfactual_lane_trunk_gradient_norm'] > 0
    if VARIANTS[variant]['isolate']:
        assert diagnostics['actual_lane_trunk_gradient_norm'] == 0
    else:
        assert diagnostics['actual_lane_trunk_gradient_norm'] > 0
    assert diagnostics['speed_support_trunk_gradient_norm'] > 0
    assert model_state_sha(model) == before
    model.save(tmp_path / 'model.zip')
    restored = type(model).load(tmp_path / 'model.zip', env=env, device='cpu')
    assert_variant(restored, variant)
    assert model_state_sha(restored) == before
    restored.num_timesteps = 1
    with pytest.raises(ValueError, match='before learning'):
        apply_training_variant(restored, variant)


def test_risk_and_support_interventions_are_exact_and_weight_preserving():
    torch.set_num_threads(1)
    observations, _ = batch()
    for decoder in ('no_risk_score', 'no_support_prior'):
        model = _small_model(_Env())
        proposals = model.actor.all_action_proposals(observations, deterministic_speed=True)
        values = model.policy._risk_adjusted_proposal_values(observations, proposals)
        baseline = values[-1].detach().clone()
        old_state = model_state_sha(model)
        if decoder == 'no_risk_score':
            expected = baseline + model.policy.collision_risk_coef * values[1] + model.policy.twin_uncertainty_coef * values[3]
        else:
            expected = baseline - model.policy.lane_prior_coef * proposals.lane_log_probabilities.unsqueeze(-1) - model.policy.component_prior_coef * proposals.component_log_probabilities
        apply_decoder(model, decoder)
        actual = model.policy._risk_adjusted_proposal_values(observations, proposals)[-1]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert model_state_sha(model) == old_state


def test_actor_intervention_uses_actor_without_postprocessing():
    model = _small_model(_Env())
    observations, _ = batch()
    expected = model.actor(observations, deterministic=True)
    apply_decoder(model, 'actor_deterministic')
    torch.testing.assert_close(model.policy._predict(observations, deterministic=True), expected)


def test_traffic_seeds_change_departures_preserve_routes_and_order():
    raw = b'<routes><vType id="t"/><route id="r" edges="a b"/><vehicle id="v1" type="t" route="r" depart="2"/><vehicle id="v2" type="t" route="r" depart="1.5"/><flow id="f" begin="5" end="10" route="r" number="4"/></routes>'
    original_sha = hashlib.sha256(raw).hexdigest()
    left, count = jitter_document(raw, 310000)
    right, _ = jitter_document(raw, 320000)
    assert left == jitter_document(raw, 310000)[0]
    assert left != right and count == 3
    assert hashlib.sha256(raw).hexdigest() == original_sha
    original = {node.attrib['id']: node for node in ET.fromstring(raw)}
    changed = ET.fromstring(left)
    departures = []
    for node in changed:
        for key, value in original[node.attrib['id']].attrib.items():
            if key not in ('depart', 'begin', 'end'):
                assert node.attrib[key] == value
        if 'depart' in node.attrib:
            departures.append(float(node.attrib['depart']))
        if node.tag == 'flow':
            assert float(node.attrib['end']) - float(node.attrib['begin']) == pytest.approx(5.)
    assert departures == sorted(departures)


def test_discounted_collision_target_uses_decision_steps():
    np.testing.assert_allclose(discounted_collision_returns([0,0,1], .9), [.81,.9,1.])
    np.testing.assert_array_equal(discounted_collision_returns([0,0,0], .9), [0.,0.,0.])


def test_selection_priorities_never_use_score_return_or_early_tie():
    def row(success, collision, timeout, step, reward):
        return {'summary': {'success_rate':success, 'collision_rate':collision,
                            'timeout_rate':timeout, 'mean_return':reward},
                'checkpoint': {'raw_steps':step}}
    rows = [row(.9,.1,0,10000,10), row(.9,.05,.05,20000,-100),
            row(.9,.05,.05,50000,-1000), row(.89,0,0,50000,10000)]
    assert max(rows,key=selection_key) is rows[2]


def test_queue_defers_multiseed_and_gates_scoring_on_selection():
    from tools.phase3_mechanism_v1.common import OUT, read, source_registry
    from tools.phase3_mechanism_v1.controller import build_jobs
    manifest = read(OUT/'manifest.json')
    jobs = build_jobs(manifest,source_registry())
    training = [j for j in jobs.values() if j['kind']=='train']
    assert len(training)==12
    assert all(not j['source'].startswith('g1_s025') for j in training)
    for job in manifest['training_jobs']:
        assert job['seed']==0 and job['variant']!='g1_s025'
    for job in jobs.values():
        if job['kind']=='eval' and job['step']=='selected':
            assert len(job['dependencies'])==1
            assert jobs[job['dependencies'][0]]['kind']=='select'
        if job['kind']=='select':
            assert len(job['dependencies'])==5
            assert {jobs[d]['step'] for d in job['dependencies']}=={10000,20000,30000,40000,50000}


def test_equivalent_jobs_cannot_execute_the_same_work_twice(tmp_path,monkeypatch):
    import concurrent.futures
    import time
    from tools.phase3_mechanism_v1 import common
    monkeypatch.setattr(common,'OUT',tmp_path)
    result = tmp_path/'result.json'
    executed=[]
    def worker():
        with common.exclusive_lock(tmp_path/'same-model.lock'):
            if not result.exists():
                time.sleep(.03)
                executed.append(1)
                common.write(result,{'completed':True})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _:worker(), range(2)))
    assert executed==[1]


def test_selection_rejects_incomplete_episode_evidence(tmp_path,monkeypatch):
    from tools.phase3_mechanism_v1 import evaluation as module
    from tools.phase3_mechanism_v1.common import write
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'OUT',tmp_path)
    paths=[]
    for step in (10000,20000,30000,40000,50000):
        path=f'{step}.json'
        write(tmp_path/path,{'source_id':'source', 'checkpoint':{'raw_steps':step},
                             'identity':{'mode':'selection','smoke':False,'episodes':100},
                             'summary':{'episodes':100},'episode_records':[]})
        paths.append(path)
    with pytest.raises(AssertionError):
        module.select({'id':'source'},paths)
