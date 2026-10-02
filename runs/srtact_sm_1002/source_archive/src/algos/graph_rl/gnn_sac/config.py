"""GNN+SAC baseline factory: continuous head over a pure vehicle-graph encoder.

The continuous action head (``SceneRepSACPolicy``) is shared with MST+SLT, so
this arm isolates the encoder dimension: graph message passing vs. the MST
transformer hierarchy, under an identical SAC objective (no SLT).
"""

from __future__ import annotations

from typing import Any

import torch

from stable_baselines3 import SAC

from algos.sb3_torch.policies import SceneRepSACPolicy
from algos.sb3_torch.replay_buffer import DictNStepReplayBuffer
from algos.sb3_torch.sac import SceneRepresentationSAC

from .gnn_encoder import GnnSceneExtractor

GNN_SAC_ALGORITHMS = ("gnn_sac",)


def source_action_repeat_gnn_sac(algo: str, requested: int | None = None) -> int:
    if algo not in GNN_SAC_ALGORITHMS:
        raise ValueError(f"GNN+SAC algorithm must be one of {GNN_SAC_ALGORITHMS}")
    return 3 if requested is None else int(requested)


def make_gnn_sac_model(
    algo: str,
    env: Any,
    *,
    scenario: str,
    learning_rate: float | None = None,
    batch_size: int = 32,
    discount: float = 0.99,
    learning_starts: int = 5000,
    buffer_size: int = 20_000,
    action_repeat: int | None = None,
    seed: int = 0,
    device: str = "auto",
    tensorboard_log: str | None = None,
    verbose: int = 1,
    **kwargs: Any,
) -> SAC:
    """Construct a GNN+SAC model (continuous head + vehicle-graph encoder)."""

    action_repeat = source_action_repeat_gnn_sac(algo, action_repeat)
    if learning_rate is None:
        learning_rate = 1e-4

    feature_kwargs = {
        "features_dim": 128,
        "hidden_dim": 128,
        "num_layers": 2,
        "vehicle_sigma": 20.0,
        "random_augmentation": scenario != "cross",
        "carla_contract": scenario == "carla",
    }
    policy_kwargs = {
        "features_extractor_class": GnnSceneExtractor,
        "features_extractor_kwargs": feature_kwargs,
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False,
        "net_arch": {"pi": [128, 32], "qf": [128, 32]},
        "optimizer_class": torch.optim.NAdam,
        "optimizer_kwargs": {"eps": 1e-7},
        "n_critics": 2,
        "action_embedding_dim": 64,
    }
    return SceneRepresentationSAC(
        SceneRepSACPolicy,
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=0,
        batch_size=batch_size,
        tau=5e-3,
        gamma=discount,
        train_freq=(1, "step"),
        gradient_steps=action_repeat,
        n_steps=4,
        replay_buffer_class=DictNStepReplayBuffer,
        replay_buffer_kwargs={
            "n_steps": 4,
            "gamma": discount,
            "source_action_repeat": action_repeat,
        },
        ent_coef="auto_0.2",
        target_entropy="auto",
        policy_kwargs=policy_kwargs,
        representation_coef=0.0,
        action_embedding_dim=64,
        max_grad_norm=5.0,
        raw_learning_starts=learning_starts,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
