from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tools.v48_stability_v4.common import BASE, CANDIDATES, DEPLOYMENT, read, OUT, lock, seal
from tools.v48_stability_v4.curves import transform, sustained_crossing, read_monitor, analyze_rows
from tools.v48_stability_v4.driving import telemetry_metrics
from tools.v48_stability_v4.model import learning_rate, actor_prediction, StabilitySAC
from tools.v48_stability_v4.report import gates


def test_round4_candidate_structure_and_single_reference_reuse():
    new = {k: v for k, v in CANDIDATES.items() if not v['reuse']}
    assert set(new) == {'tau0025_floor2e5_hold35k', 'tau0025_floor2e5_hold40k', 'tau0025_floor2e5_hold30k',
                        'tau0025_floor2e5_hold45k', 'tau0025_floor1p5e5_hold35k'}
    # No delayed-start, no constant-low-LR candidate: round 3 showed start 30k hurts
    # and round 1 showed constant 2.5e-5 underfits. All new candidates start decay at 20k.
    for name, values in new.items():
        assert values['learning_rate'] == BASE['learning_rate'] == 5e-5
        assert values['tau'] == pytest.approx(.0025)
        assert values['decay_start'] == 20000
    assert {k for k, v in CANDIDATES.items() if v['reuse']} == {'lr_half'}
    # Settle sweep at floor 2e-5: decay_end 30k/35k/40k/45k -> settle 20k/15k/10k/5k
    # (decay_end 50k = round-3 floor2e5, settle 0, already known 0.97).
    for name, end, floor in (('tau0025_floor2e5_hold35k', 35000, 2e-5),
                             ('tau0025_floor2e5_hold40k', 40000, 2e-5),
                             ('tau0025_floor2e5_hold30k', 30000, 2e-5),
                             ('tau0025_floor2e5_hold45k', 45000, 2e-5)):
        assert CANDIDATES[name]['decay_end'] == end
        assert CANDIDATES[name]['final_learning_rate'] == pytest.approx(floor)
    # Deeper-floor probe at fixed settle 15k (vs 2.5e-5=hold35k and 2e-5=floor2e5_hold35k).
    assert CANDIDATES['tau0025_floor1p5e5_hold35k']['final_learning_rate'] == pytest.approx(1.5e-5)
    assert CANDIDATES['tau0025_floor1p5e5_hold35k']['decay_end'] == 35000
    assert all(v['batch_size'] == 32 for v in new.values())
    assert DEPLOYMENT == 'exact_final_actor_deterministic'


@pytest.mark.parametrize('candidate, expected', [
    ('tau0025_floor2e5_hold35k', [(0, 5e-5), (20000, 5e-5), (27500, 3.5e-5), (35000, 2e-5), (50000, 2e-5)]),
    ('tau0025_floor2e5_hold40k', [(0, 5e-5), (20000, 5e-5), (30000, 3.5e-5), (40000, 2e-5), (50000, 2e-5)]),
    ('tau0025_floor2e5_hold30k', [(0, 5e-5), (20000, 5e-5), (25000, 3.5e-5), (30000, 2e-5), (50000, 2e-5)]),
    ('tau0025_floor2e5_hold45k', [(0, 5e-5), (20000, 5e-5), (32500, 3.5e-5), (45000, 2e-5), (50000, 2e-5)]),
    ('tau0025_floor1p5e5_hold35k', [(0, 5e-5), (20000, 5e-5), (27500, 3.25e-5), (35000, 1.5e-5), (50000, 1.5e-5)]),
])
def test_learning_rate_schedule(candidate, expected):
    for step, value in expected:
        assert learning_rate(CANDIDATES[candidate], step) == pytest.approx(value)


def test_ema_matches_closed_form_for_first_episode_and_appended_endpoint():
    # First successful episode contributes 1/20 due to zero padding, held over
    # every raw step. Closed form is independent of the implementation loop.
    rows = [dict(r=2., is_success=True, raw_simulation_steps=207)]
    success = transform(rows, 'success')
    reward = transform(rows, 'reward')
    assert success[:, 0].tolist() == [0, 100, 200, 207]
    assert success[-1, 1] == pytest.approx(.05 * (1 - .999**207), abs=1e-13)
    assert reward[-1, 1] == pytest.approx(.1 * (1 - .999**207), abs=1e-13)
    std = math.sqrt((.95**2 + 19 * .05**2) / 20) / math.sqrt(20)
    assert success[-1, 3] == pytest.approx((.05 + std) * (1 - .999**207), abs=1e-13)


def test_sustained_threshold_not_first_transient_crossing():
    a = np.array([[0, .9], [1000, .9], [2000, .5], [3000, .8], [4000, .9], [8000, .9]])
    assert sustained_crossing(a) == 3000
    assert sustained_crossing(a, .95) is None


def test_monitor_uses_raw_steps_and_excludes_over_budget_incomplete_range(tmp_path):
    path = tmp_path / 'monitor.csv'
    path.write_text('#meta\nr,l,raw_simulation_steps,is_success\n1,1,500,True\n-1,1,700,False\n', encoding='utf-8')
    rows = read_monitor(path, budget=1000)
    assert len(rows) == 1 and rows[0]['raw_simulation_steps'] == 500


def test_scheduled_update_reaches_all_optimizers_without_lost_pending_updates(monkeypatch):
    from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45
    optimizer = lambda: SimpleNamespace(param_groups=[{'lr': 5e-5}])
    m = object.__new__(StabilitySAC)
    m.actor, m.critic = SimpleNamespace(optimizer=optimizer()), SimpleNamespace(optimizer=optimizer())
    m.policy = SimpleNamespace(encoder_optimizer=None)
    m.ent_coef_optimizer, m.representation_optimizer = optimizer(), optimizer()
    m._logger = SimpleNamespace(name_to_value={}, record=lambda *a, **k: None)
    m.stability_config = dict(CANDIDATES['tau0025_floor2e5_hold35k'], decay_start=48, decay_end=50)
    m.raw_learning_starts, m._n_updates, m._pending_raw_gradient_steps, m.target_update_interval = 48, 0, 3, 1
    m.audit_parameters = lambda: None
    rates = []
    def step(self, count, batch_size):
        assert count == 1
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer])
        rates.append([o.param_groups[0]['lr'] for o in
                      (self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer, self.representation_optimizer)])
        self._n_updates += count
    monkeypatch.setattr(ConfidentActorFusionSACV45, 'train', step)
    m.train(999, 32)
    assert m._n_updates == 3 and m._pending_raw_gradient_steps is None
    assert np.array(rates) == pytest.approx(np.repeat([[5e-5], [3.5e-5], [2e-5]], 4, axis=1))
    assert m.representation_learning_rate == pytest.approx(2e-5)


def test_actor_dispatch_does_not_read_decoder_or_critic():
    calls = []
    def actor(obs, deterministic=False):
        calls.append((obs, deterministic))
        return 'actor action'
    policy = SimpleNamespace(actor=actor)
    assert actor_prediction(policy, 'observation', deterministic=True) == 'actor action'
    assert calls == [('observation', True)]


def sample(t, **kwargs):
    result = dict(time=t, speed=10., acceleration=0., angle=0., distance=10 * t, allowed_speed=9.,
                  lateral_speed=0., road='edge', lane=0, leader_gap=None, closing_speed=None, ttc=None, headway=None)
    result.update(kwargs)
    return result


def test_driving_units_missing_ttc_and_terminal_censoring():
    metrics = telemetry_metrics(sample(0), [sample(.1), sample(.2), None],
                                [dict(feasible=True), dict(feasible=False)], 3)
    assert metrics['mean_speed_mps'] == 10
    assert metrics['observed_distance_m'] == pytest.approx(2.)
    assert metrics['raw_step_coverage'] == pytest.approx(2 / 3)
    assert metrics['jerk_rms_mps3'] == 0
    assert metrics['minimum_following_ttc_s'] is None
    assert metrics['following_ttc_below_1_5s_conditional_fraction'] is None
    assert metrics['distance_censored']
    assert metrics['infeasible_command_fraction'] == .5


def test_jerk_and_lane_changes_ignore_edge_boundaries():
    initial = sample(0, angle=359.)
    samples = [sample(.1, acceleration=-4, lane=1, angle=1., leader_gap=2., closing_speed=2., ttc=1., headway=.2),
               sample(.2, acceleration=-4, lane=0, road='next', angle=1.)]
    m = telemetry_metrics(initial, samples, [], 2)
    assert m['jerk_rms_mps3'] == pytest.approx(math.sqrt(800))
    assert m['observed_lane_changes'] == 1
    assert m['minimum_following_ttc_s'] == 1
    assert m['following_ttc_below_1_5s_observed_fraction'] == .5
    assert m['lateral_accel_proxy_abs_p95_mps2'] < 4  # angle unwrap: two degrees, not 358


def test_low_variance_failure_cannot_win_screen():
    baseline = dict(training={metric: dict(tail_raw_episodes=dict(mean=.8),
        curve=dict(mean_auc=.8, max_drawdown_after_5k=.2, excess_total_variation=1., tail_curve_std=.06)) for metric in ('success', 'reward')},
        evaluation=dict(summary=dict(success_rate=.9, collision_rate=.1, timeout_rate=0.)))
    poor = copy.deepcopy(baseline)
    poor['training']['success']['curve']['tail_curve_std'] = 0.
    poor['training']['success']['tail_raw_episodes']['mean'] = 0.
    poor['evaluation']['summary'].update(success_rate=0., timeout_rate=1.)
    result = gates(poor, baseline, read(OUT / 'manifest.json')['gates'])
    assert not result['eligible'] and 'deployment success_rate' in result['reasons']


def test_manifest_reuses_exact_completed_models_and_actor_only():
    manifest = read(OUT / 'manifest.json')
    assert manifest['training_seeds'] == [0]
    assert manifest['max_gpu_workers'] == 2 and manifest['cpu_workers'] == 0
    assert manifest['new_training_cells'] == 10 and manifest['reused_training_cells'] == 2
    assert manifest['deployment'] == DEPLOYMENT
    assert not manifest['checkpoint_selector'] and not manifest['critic_deployment_decoder']
    assert all(s['deployment'] == DEPLOYMENT for s in manifest['sources'].values())


def test_immutable_artifact_rejects_protocol_drift(tmp_path):
    path = tmp_path / 'receipt.json'
    seal(path, {'value': 1})
    seal(path, {'value': 1})
    with pytest.raises(ValueError, match='Immutable'):
        seal(path, {'value': 2})
