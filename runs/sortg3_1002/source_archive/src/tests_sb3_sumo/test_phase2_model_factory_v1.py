import pytest
import torch
from tools.phase2_model_factory_v1 import make_phase2_model, verify_optimizer_settings
from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory
from tools.phase1_checkpoint_diagnostics import read, ROOT, PROTOCOL
import argparse


@pytest.mark.parametrize('method', ['mst_slt', 'temporal_graph', 'v4_8', 'v4_13'])
@pytest.mark.parametrize('lr,tau', [(1e-4,.005),(5e-5,.005),(2.5e-5,.005),(1e-4,.0025),(1e-4,.01)])
def test_real_model_optimizers_follow_candidate(method, lr, tau, tmp_path):
    torch.set_num_threads(1)
    plan = read(ROOT/'results_phase1_checkpoint_diagnostics_v1/plan.json')
    source = next(s for s in plan['sources'] if s['method']==method and s['scenario']=='cross')
    args = argparse.Namespace(**read(ROOT/source['run']/'arguments.json')['requested_raw_steps'])
    env = _make_environment_factory(adapter='base' if method in ('mst_slt','temporal_graph') else method,
        density=read(PROTOCOL)['scenarios']['cross'], overlay_root=tmp_path)(args)
    try:
        model = make_phase2_model(method, env, learning_rate=lr, tau=tau, scenario='cross',
            batch_size=2, buffer_size=64, learning_starts=48, seed=0, device='cpu', verbose=0)
        before = verify_optimizer_settings(model, learning_rate=lr, tau=tau)
        assert 'representation' in before['optimizer_rates']
        if method=='v4_13':
            assert 'encoder' in before['optimizer_rates']
            assert 'collision_critic' in before['optimizer_rates']
            assert model.optimizer_ownership_audit()['overlap_count']==0
        # Exercise the real SB3 scheduler; explicit representation rates must survive too.
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(folder=str(tmp_path/'logger'), format_strings=[]))
        model._current_progress_remaining = .4
        model._update_learning_rate([model.actor.optimizer, model.critic.optimizer])
        verify_optimizer_settings(model, learning_rate=lr, tau=tau)
        model.actor.optimizer.param_groups[0]['lr'] *= 2
        with pytest.raises(ValueError, match='learning-rate mismatch'):
            verify_optimizer_settings(model, learning_rate=lr, tau=tau)
    finally:
        env.close()


def test_excluded_method_and_invalid_parameters():
    with pytest.raises(ValueError, match='excluded'):
        make_phase2_model('full_balanced', None, learning_rate=1e-4, tau=.005)
    with pytest.raises(ValueError, match='positive'):
        make_phase2_model('v4_13', None, learning_rate=float('nan'), tau=.005)
