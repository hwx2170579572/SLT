"""Construction and loading of the three compared methods.

Training uses the frozen phase-2 factory so the three methods share one trainer
adapter and one optimizer-settings verifier.  Loading reuses the phase-3
deployment rules unchanged, because the same decoder must be used for the
reused seed-0 evidence and the new seeds.
"""
from __future__ import annotations

from pathlib import Path

from .common import ROOT
from .config import ACTION_REPEAT, METHODS, SMOKE_BATCH, SMOKE_BUFFER, SMOKE_WARMUP, TRAINING_BATCH, TRAINING_BUFFER, TRAINING_WARMUP


def assert_candidate(method, candidate):
    expected = METHODS[method]['candidate']
    if candidate != expected:
        raise ValueError(f'{method} must run the pre-registered candidate {expected!r}, not {candidate!r}')


def make_model(env, method, candidate, scenario, seed, device='cuda', smoke=False):
    """Build an untrained model with the frozen optimizer settings and seed."""
    from tools.phase2_model_factory_v1 import make_phase2_model, verify_optimizer_settings
    assert_candidate(method, candidate)
    if seed not in (0, 1, 2):
        raise ValueError(f'Training seed outside the sealed set: {seed}')
    settings = METHODS[method]
    model = make_phase2_model(
        method, env, learning_rate=settings['learning_rate'], tau=settings['tau'],
        scenario=scenario,
        batch_size=SMOKE_BATCH if smoke else TRAINING_BATCH,
        learning_starts=SMOKE_WARMUP if smoke else TRAINING_WARMUP,
        buffer_size=SMOKE_BUFFER if smoke else TRAINING_BUFFER,
        action_repeat=ACTION_REPEAT, seed=seed, device=device, verbose=0,
    )
    verify_optimizer_settings(model, learning_rate=settings['learning_rate'], tau=settings['tau'])
    model.phase4_multiseed_metadata = {
        'schema': 'phase4-multiseed-v1', 'method': method, 'candidate': candidate,
        'learning_rate': settings['learning_rate'], 'tau': settings['tau'],
        'training_seed': seed, 'smoke': smoke, 'new_network_parameters': False,
        'hyperparameters_inherited_from': 'results_phase2_diagnosis_20260908/tuning_contract_v2.json',
    }
    return model


def load_model(source, checkpoint, env, device='cuda'):
    """Deployment loader; identical dispatch to phase 3 for every method."""
    checkpoint = ROOT / Path(checkpoint)
    if source['method'] == 'mst_slt':
        from algos.sb3_torch.sac import SceneRepresentationSAC
        return SceneRepresentationSAC.load(checkpoint, env=env, device=device)
    if source['method'] == 'v4_8':
        from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45
        from tools.action_diagnostics_v4_6 import load_model_for_deployment
        return load_model_for_deployment(ConfidentActorFusionSACV45, checkpoint,
                                        decoder=source['decoder'], env=env, device=device)
    if source['decoder'] != 'target_critic':
        raise ValueError(f'v4_13 deployment rule is fixed to target_critic, not {source["decoder"]!r}')
    from tools.action_diagnostics_v4_13_model import load_model_for_deployment_v4_13
    from algos.sb3_torch.sac_v4_13_model import GradientIsolatedTemperedJointSupportSACV413
    return load_model_for_deployment_v4_13(GradientIsolatedTemperedJointSupportSACV413,
                                          checkpoint, decoder='target_critic', env=env, device=device)
