"""Apply explicit factorial interventions before training, without new networks."""
from __future__ import annotations

VARIANTS = {
    'g1_s025': {'isolate': True, 'lane_scale': .25, 'support_coef': .05},
    'g0_s025': {'isolate': False, 'lane_scale': .25, 'support_coef': .05},
    'g1_s1': {'isolate': True, 'lane_scale': 1., 'support_coef': .05},
    'g0_s1': {'isolate': False, 'lane_scale': 1., 'support_coef': .05},
    'g0_s1_lowall': {'isolate': False, 'lane_scale': 1., 'support_coef': .0125},
}


def apply_training_variant(model, variant):
    if model._n_updates or model.num_timesteps:
        raise ValueError('Training variant must be set before learning')
    values = VARIANTS[variant]
    for owner in (model.policy, model.actor):
        owner.isolate_lane_support_gradient = values['isolate']
        owner.lane_support_scale = values['lane_scale']
    model.policy_kwargs.update(isolate_lane_support_gradient=values['isolate'],
                               lane_support_scale=values['lane_scale'])
    model.replay_support_coef = values['support_coef']
    model.phase3_mechanism_metadata = {
        'schema': 'phase3-mechanism-v1', 'variant': variant, **values,
        'parent': 'v4.13', 'train_seed': 0, 'cross_augmentation': True,
        'new_network_parameters': False, 'deployment_rule_changed': False,
    }
    assert_variant(model, variant)
    return model


def assert_variant(model, variant):
    values = VARIANTS[variant]
    for owner in (model.policy, model.actor):
        assert owner.isolate_lane_support_gradient == values['isolate']
        assert owner.lane_support_scale == values['lane_scale']
    assert model.replay_support_coef == values['support_coef']
    assert model.policy_kwargs['lane_support_scale'] == values['lane_scale']
    assert model.policy_kwargs['isolate_lane_support_gradient'] == values['isolate']


def make_model(env, scenario, variant, device='cuda', smoke=False):
    from tools.phase2_model_factory_v1 import make_phase2_model
    model = make_phase2_model(
        'v4_13', env, learning_rate=1e-4, tau=.0025, scenario=scenario,
        batch_size=2 if smoke else 32, learning_starts=48 if smoke else 5000,
        buffer_size=128 if smoke else 20000, action_repeat=3, seed=0,
        device=device, verbose=0,
    )
    return apply_training_variant(model, variant)


def load_model(source, checkpoint, env, device='cuda'):
    from pathlib import Path
    from .common import ROOT
    checkpoint = ROOT / Path(checkpoint)
    if source['method'] == 'mst_slt':
        from algos.sb3_torch.sac import SceneRepresentationSAC
        return SceneRepresentationSAC.load(checkpoint, env=env, device=device)
    if source['method'] == 'v4_8':
        from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45
        from tools.action_diagnostics_v4_6 import load_model_for_deployment
        return load_model_for_deployment(ConfidentActorFusionSACV45, checkpoint,
                                        decoder=source['decoder'], env=env, device=device)
    from tools.action_diagnostics_v4_13_model import load_model_for_deployment_v4_13
    from algos.sb3_torch.sac_v4_13_model import GradientIsolatedTemperedJointSupportSACV413
    return load_model_for_deployment_v4_13(GradientIsolatedTemperedJointSupportSACV413,
                                          checkpoint, decoder='target_critic', env=env, device=device)
