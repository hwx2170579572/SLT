"""Phase-2 optimization-only adapter. Original model factories remain unchanged."""
from __future__ import annotations
import math
from typing import Any


def optimizer_rates(model: Any) -> dict[str, list[float]]:
    owners = {'actor': model.actor, 'critic': model.critic}
    if hasattr(model.policy, 'collision_critic'):
        owners['collision_critic'] = model.policy.collision_critic
    result = {name: [float(g['lr']) for g in owner.optimizer.param_groups]
              for name, owner in owners.items()}
    for name, optimizer in [('encoder', getattr(model.policy, 'encoder_optimizer', None)),
                            ('representation', model.representation_optimizer),
                            ('entropy', model.ent_coef_optimizer)]:
        if optimizer is not None:
            result[name] = [float(g['lr']) for g in optimizer.param_groups]
    return result


def verify_optimizer_settings(model: Any, *, learning_rate: float, tau: float) -> dict:
    rates = optimizer_rates(model)
    if not rates or any(not math.isclose(lr, learning_rate, rel_tol=1e-9, abs_tol=0)
                        for groups in rates.values() for lr in groups):
        raise ValueError(f'Optimizer learning-rate mismatch: {rates}')
    if not math.isclose(float(model.tau), tau, rel_tol=1e-9, abs_tol=0):
        raise ValueError('Target-update tau mismatch')
    if not math.isclose(float(model.representation_learning_rate), learning_rate, rel_tol=1e-9):
        raise ValueError('Explicit representation learning rate differs')
    return {'learning_rate': learning_rate, 'tau': tau, 'optimizer_rates': rates,
            'updates': int(model._n_updates)}


def make_phase2_model(method: str, env: Any, *, learning_rate: float, tau: float, **kwargs: Any):
    if not (math.isfinite(learning_rate) and learning_rate > 0 and math.isfinite(tau) and 0 < tau <= 1):
        raise ValueError('learning_rate must be positive and 0 < tau <= 1')
    if method in ('mst_slt', 'temporal_graph'):
        from configs.sb3_configs import make_model
        algorithm = 'scene_rep' if method == 'mst_slt' else 'temporal_graph'
        model = make_model(algorithm, env, learning_rate=learning_rate, **kwargs)
    elif method == 'v4_8':
        from configs.sb3_configs_v4_8 import make_model_v4_8, V48_CANDIDATE
        model = make_model_v4_8(V48_CANDIDATE, env, learning_rate=learning_rate, **kwargs)
    elif method == 'v4_13':
        from configs.sb3_configs_v4_13 import make_model_v4_13, V413_FULL
        model = make_model_v4_13(V413_FULL, env, learning_rate=learning_rate, **kwargs)
    else:
        raise ValueError(f'Method excluded from phase 2: {method}')
    if model._n_updates != 0 or model.num_timesteps != 0:
        raise ValueError('Phase-2 settings must be applied before training')
    model.tau = tau
    # Factories already initialize every optimizer, including representation, with lr.
    # Verify rather than repairing an unexpected optimizer configuration silently.
    receipt = verify_optimizer_settings(model, learning_rate=learning_rate, tau=tau)
    model.phase2_optimization_metadata = dict(schema='phase2-optimizer-settings/v1', method=method, **receipt)
    return model
