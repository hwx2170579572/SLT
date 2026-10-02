"""Preflight tests for the multi-seed confirmation.

These cover the claims that make the phase meaningful: the matrix is exactly the
sealed one, the optimizer settings still match the frozen contract, the seed-0
reuse boundary is what it says it is, the bound prior evidence really belongs to
the checkpoints it is attached to, and the fresh evaluation traffic is identical
for the baseline and v4 adapters.
"""
from __future__ import annotations

import pytest

from tools.phase4_multiseed_v1.common import ROOT, read, sha
from tools.phase4_multiseed_v1.config import (METHODS, SCENES, TRAINING_SEEDS, cell_id, cells,
                                              is_reused, phase3_cell_id)
from tools.phase4_multiseed_v1.evaluation import selection_key
from tools.phase4_multiseed_v1.plan import (assert_environment_equivalence, bind_seed_zero_evidence,
                                            phase1_source)
from tools.phase4_multiseed_v1.traffic import MODES, jitter_document


def test_matrix_is_the_sealed_fifty_four_cells():
    matrix = cells()
    assert len(matrix) == 54
    assert len(set(matrix)) == 54
    assert {m for m, _, _ in matrix} == set(METHODS)
    assert {s for _, s, _ in matrix} == set(SCENES)
    assert {k for _, _, k in matrix} == set(TRAINING_SEEDS)


def test_cell_identifiers_are_unique_and_carry_the_seed():
    identifiers = [cell_id(method, scene, seed) for method, scene, seed in cells()]
    assert len(set(identifiers)) == 54
    assert all(identifier.endswith(f'__seed{seed}') for identifier, (_, _, seed) in zip(identifiers, cells()))
    assert cell_id('v4_8', 'carla', 2).startswith('v4_8__carla__lr_half__seed2')


def test_seed_zero_reuse_boundary_is_explicit():
    for method in METHODS:
        assert not is_reused(method, 1) and not is_reused(method, 2)
    assert is_reused('v4_8', 0) and is_reused('v4_13', 0)
    # MST+SLT seed 0 is retrained because its historical models are mixed-lineage.
    assert not is_reused('mst_slt', 0)


def test_optimizer_settings_still_match_the_frozen_tuning_contract():
    contract = read(ROOT / 'results_phase2_diagnosis_20260908/tuning_contract_v2.json')
    by_id = {candidate['id']: candidate for candidate in contract['candidates']}
    for method, settings in METHODS.items():
        candidate = by_id[settings['candidate']]
        assert candidate['learning_rate'] == settings['learning_rate']
        assert candidate['tau'] == settings['tau']
        assert candidate['representation_learning_rate'] == settings['learning_rate']


def test_evaluation_modes_keep_the_sealed_seeds_and_episode_counts():
    assert MODES['selection'] == (310000, 100, 1.)
    assert MODES['score'] == (320000, 100, 1.)


def test_selection_ranking_is_success_then_collision_then_timeout_then_later():
    def row(success, collision, timeout, steps):
        return {'summary': {'success_rate': success, 'collision_rate': collision,
                            'timeout_rate': timeout}, 'checkpoint': {'raw_steps': steps}}
    best = row(.9, .05, .05, 20000)
    fewer_collisions = row(.9, .04, .09, 10000)
    more_success = row(.95, .05, 0., 10000)
    later_tie = row(.9, .05, .05, 40000)
    assert max([best, fewer_collisions], key=selection_key) is fewer_collisions
    assert max([best, more_success], key=selection_key) is more_success
    assert max([best, later_tie], key=selection_key) is later_tie


def test_fresh_traffic_jitter_is_deterministic_and_actually_moves_departures():
    raw = (b'<routes><vehicle id="a" depart="1.0"/><vehicle id="b" depart="2.0"/>'
           b'<flow id="f" begin="3.0" end="9.0"/></routes>')
    first, changed_first = jitter_document(raw, 12345)
    second, changed_second = jitter_document(raw, 12345)
    assert first == second and changed_first == changed_second == 3
    other, _ = jitter_document(raw, 999)
    assert other != first


def test_fresh_traffic_rejects_documents_without_numeric_departures():
    with pytest.raises(ValueError, match='No numeric social departures'):
        jitter_document(b'<routes><vehicle id="a"/></routes>', 1)


@pytest.mark.parametrize('method,scene', [('v4_8', 'left_turn'), ('v4_8', 'cross'),
                                          ('v4_13', 'carla'), ('v4_13', 'roundabout_medium')])
def test_reused_runs_used_the_environment_arguments_we_now_declare(method, scene):
    assert assert_environment_equivalence(method, scene)


@pytest.mark.parametrize('method', ['v4_8', 'v4_13'])
def test_bound_evidence_covers_every_selection_checkpoint_and_both_scored_models(method):
    for scene in SCENES:
        bound = bind_seed_zero_evidence(method, scene)
        for step in (10000, 20000, 30000, 40000, 50000):
            assert f'selection::{step}::native' in bound['bindings']
        assert f'score::{bound["selected_step"]}::native' in bound['bindings']
        assert 'score::50000::native' in bound['bindings']
        assert len(bound['bindings']) == len({*bound['bindings']})
        for key, binding in bound['bindings'].items():
            mode, step, decoder = key.split('::')
            assert decoder == 'native'
            prior = read(ROOT / binding['path'])
            assert sha(ROOT / binding['path']) == binding['sha256']
            assert prior['identity']['checkpoint_sha256'] == bound['checkpoints'][step]['sha256']
            assert prior['identity']['mode'] == mode
            assert prior['identity']['episodes'] == 100
            assert prior['identity']['seed_start'] == (310000 if mode == 'selection' else 320000)


def test_reused_cell_refuses_an_unsealed_evaluation_instead_of_rerunning():
    from tools.phase4_multiseed_v1.common import OUT
    from tools.phase4_multiseed_v1.evaluation import bound_evidence, evaluate
    from tools.phase4_multiseed_v1.plan import aligned_environment_arguments

    manifest = OUT / 'manifest.json'
    if not manifest.exists():
        pytest.skip('sealed manifest not created yet')
    source = read(manifest)['sources']['v4_8__left_turn__lr_half__seed0']
    assert source['training_seed'] == 0 and not source['new_training']
    assert source['environment_arguments'] == aligned_environment_arguments('v4_8', 'left_turn', 0)
    assert bound_evidence(source, 15000, 'selection', 'native') is None
    # An unknown checkpoint fails loudly instead of silently starting a rollout.
    with pytest.raises(KeyError):
        evaluate(source, 15000, 'selection')
    # A reused cell with an incomplete binding set must refuse, not re-run SUMO.
    partial = {**source, 'reused_evaluations': {k: v for k, v in source['reused_evaluations'].items()
                                                if k != 'selection::20000::native'}}
    with pytest.raises(ValueError, match='unsealed evaluation'):
        evaluate(partial, 20000, 'selection')


def test_sealed_evaluation_results_carry_the_metrics_analyze_reads():
    from tools.phase4_multiseed_v1.analyze import METRICS
    bound = bind_seed_zero_evidence('v4_8', 'cross')
    for key in ('selection::10000::native', 'score::50000::native'):
        prior = read(ROOT / bound['bindings'][key]['path'])
        assert set(METRICS) <= set(prior['summary'])
        assert len(prior['episode_records']) == 100


def test_reused_checkpoints_point_at_the_completed_phase2_screen_run():
    for method, scene in (('v4_8', 'cross'), ('v4_13', 'roundabout')):
        bound = bind_seed_zero_evidence(method, scene)
        run = ROOT / bound['run']
        assert read(run / 'status.json')['status'] == 'completed'
        assert bound['checkpoints']['50000']['sha256'] == sha(run / 'final_model.zip')
        assert bound['checkpoints']['50000']['raw_steps'] == 50000


def test_only_the_mixed_lineage_cells_require_new_training():
    from tools.phase4_multiseed_v1.config import REUSED_SEED_ZERO_METHODS
    retrained_seed_zero = [method for method in METHODS if method not in REUSED_SEED_ZERO_METHODS]
    # Exactly MST+SLT, and only it, is realigned at seed 0.
    assert retrained_seed_zero == ['mst_slt']
    assert phase3_cell_id('mst_slt', 'cross') == 'mst_slt__cross__control'
    assert phase1_source('mst_slt', 'cross')['run'].startswith('results_hd_ss100_v2/')
    assert phase1_source('v4_13', 'cross')['run'].startswith('results_iv2_5m6s100e_v1/')


def test_model_factory_rejects_unregistered_candidate_and_seed():
    import torch

    from tools.phase4_multiseed_v1.model import assert_candidate, make_model
    assert_candidate('v4_8', 'lr_half')
    with pytest.raises(ValueError, match='pre-registered candidate'):
        assert_candidate('v4_8', 'control')
    with pytest.raises(ValueError, match='pre-registered candidate'):
        assert_candidate('mst_slt', 'lr_half')
    with pytest.raises(ValueError, match='Training seed outside the sealed set'):
        make_model(None, 'v4_8', 'lr_half', 'cross', 7, 'cpu', True)
    assert torch is not None


def test_fresh_traffic_pairing_matches_between_baseline_and_v4_adapters():
    """Same scenario and simulation seed must yield byte-identical fresh social traffic."""
    from tools.phase4_multiseed_v1.traffic import make_env
    receipts = {}
    for method in ('mst_slt', 'v4_13'):
        scene = 'left_turn'
        source = {'method': method, 'scenario': scene,
                  'arguments': f'{phase1_source(method, scene)["run"]}/arguments.json'}
        env = make_env(source, 'selection', f'test_pairing_{method}')
        try:
            env._sumo_command(310000)
            receipts[method] = [f['sha256'] for f in env.phase4_traffic_receipt['files']]
        finally:
            env.close()
    assert receipts['mst_slt'] == receipts['v4_13']
