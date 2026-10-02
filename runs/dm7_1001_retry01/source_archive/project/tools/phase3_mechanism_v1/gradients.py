"""Gradient conflict on one common real state/action bank; no optimizer steps."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from algos.sb3_torch.evaluation import source_evaluation_augmentation
from algos.sb3_torch.hybrid_policy_v4_10_model import all_proposal_q_from_features
from .common import ROOT, OUT, digest, model_state_sha, read, relative, seal, sha
from .evaluation import resolved_checkpoint
from .model import load_model


def vector(loss, parameters):
    values = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
    return torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1)
                      for p, g in zip(parameters, values)])


def cosine(a, b):
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return None if denominator.item() == 0. else float((torch.dot(a, b) / denominator).item())


def gradient_metrics(model, observations, replay_actions):
    actor = model.actor
    parameters = tuple(actor.latent_pi.parameters())
    batch = actor.all_action_proposals(observations, deterministic_speed=False)
    support = actor.joint_replay_support_terms(batch, replay_actions)
    free_lane = F.nll_loss(batch.lane_log_probabilities, actor.replay_lane_indices(replay_actions))
    features = model.critic.extract_features(observations, model.critic.features_extractor).detach()
    reward_heads = all_proposal_q_from_features(model.critic, features, batch.actions)
    risk_heads = all_proposal_q_from_features(model.policy.collision_critic, features, batch.actions)
    min_q, reward_disagreement = model._twin_reward_values(reward_heads)
    collision_value, collision_disagreement = model._twin_collision_values(risk_heads)
    alpha = model.log_ent_coef.detach().exp() if model.log_ent_coef is not None else model.ent_coef_tensor
    per_action = (alpha * model.entropy_log_probabilities_v410(batch) - min_q
                  + model.collision_risk_coef * collision_value
                  + model.twin_uncertainty_coef * (reward_disagreement + collision_disagreement))
    objective = (batch.proposal_probabilities * per_action).sum(dim=(1, 2)).mean()
    actual = vector(support.lane_categorical_nll, parameters)
    lane = vector(free_lane, parameters)
    speed = vector(support.conditional_speed_mixture_nll, parameters)
    rl = vector(objective, parameters)
    norm = lambda x: float(torch.linalg.vector_norm(x).item())
    return {
        'actual_lane_trunk_gradient_norm': norm(actual),
        'unisolated_counterfactual_lane_trunk_gradient_norm': norm(lane),
        'speed_support_trunk_gradient_norm': norm(speed), 'rl_trunk_gradient_norm': norm(rl),
        'lane_vs_rl_cosine': cosine(lane, rl), 'lane_vs_speed_support_cosine': cosine(lane, speed),
        'effective_actual_lane_support_norm': model.replay_support_coef * actor.lane_support_scale * norm(actual),
        'effective_unisolated_lane_support_norm': model.replay_support_coef * actor.lane_support_scale * norm(lane),
        'effective_speed_support_norm': model.replay_support_coef * norm(speed),
        'lane_nll': float(free_lane.detach()), 'speed_nll': float(support.conditional_speed_mixture_nll.detach()),
        'rl_objective': float(objective.detach()),
    }


def load_bank(result_path):
    result_path = ROOT / result_path
    arrays = {}
    bindings = []
    for episode in sorted(result_path.parent.glob('episode_*.json')):
        row = read(episode)
        source = row['bank']
        path = ROOT / source['path']
        if sha(path) != source['sha256']:
            raise ValueError('Diagnostic bank hash drift')
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                arrays.setdefault(name, []).append(data[name])
        bindings.append(source)
    if not bindings:
        raise ValueError('No real diagnostic bank')
    return {k: np.concatenate(v) for k, v in arrays.items()}, bindings


def diagnose(source, step, bank_result, device='cuda', smoke=False):
    torch.set_num_threads(1)
    cp = resolved_checkpoint(source, step)
    identity = {'model': cp, 'source': source['id'], 'bank_result': bank_result,
                'bank_result_sha256': sha(ROOT / bank_result), 'smoke': smoke}
    path = OUT / ('smoke_gradients' if smoke else 'gradients') / f'{digest(identity)[:20]}.json'
    if path.exists():
        if read(path)['identity'] != identity:
            raise ValueError('Gradient cache mismatch')
        return relative(path)
    arrays, bindings = load_bank(bank_result)
    count = len(arrays['_actions'])
    maximum = 64 if smoke else 4096
    rng = np.random.default_rng(410000)
    indices = rng.choice(count, size=min(count, maximum), replace=False)
    model = load_model(source, cp['path'], None, device)
    before = model_state_sha(model)
    model.policy.set_training_mode(False)
    rows = []
    with source_evaluation_augmentation(model):
        for start in range(0, len(indices), 32):
            ix = indices[start:start + 32]
            torch.manual_seed(420000 + start)
            observations = {k: torch.as_tensor(v[ix], device=model.device)
                            for k, v in arrays.items() if not k.startswith('_')}
            actions = torch.as_tensor(arrays['_actions'][ix], device=model.device)
            rows.append(gradient_metrics(model, observations, actions))
    if model_state_sha(model) != before:
        raise ValueError('Gradient diagnostic mutated learned tensors')
    aggregates = {}
    for metric in rows[0]:
        values = [r[metric] for r in rows if r[metric] is not None]
        aggregates[metric] = {'mean': float(np.mean(values)) if values else None,
                              'min': float(np.min(values)) if values else None,
                              'max': float(np.max(values)) if values else None,
                              'valid_batches': len(values)}
        if metric.endswith('cosine'):
            aggregates[metric]['negative_fraction'] = float(np.mean(np.asarray(values) < 0)) if values else None
    seal(path, {'identity': identity, 'rows': rows, 'aggregates': aggregates,
                'actual_isolation': model.actor.isolate_lane_support_gradient,
                'lane_support_scale': model.actor.lane_support_scale,
                'replay_support_coef': model.replay_support_coef,
                'real_bank_samples': count, 'used_samples': len(indices), 'bank_bindings': bindings,
                'bank_is_original_training_replay': False,
                'counterfactual_lane_gradient_does_not_imply_observed_training_conflict': True,
                'tensor_state_unchanged': True, 'training_performed': False})
    return relative(path)
