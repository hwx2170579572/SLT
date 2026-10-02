"""HSAC baseline factory: hybrid action head over a simple MLP/LSTM encoder.

This is the "hybrid action space" ablation arm.  It keeps the exact hybrid
head / hybrid critic from ``hybrid_policy_v4.py`` (``DecisionAlignedSACPolicy``)
but swaps the Scene-Rep / graph representation for a plain ``SimpleMlpLstmExtractor``
and disables the auxiliary SLT objective (``representation_coef=0.0``), so the
contribution of the representation can be read off against TASAC(hold35k).

The environment must expose ``lane_action_mask`` (the v4_8 contract used by
``make_hold35k_env``), because the hybrid head masks infeasible lane commands.
"""

from __future__ import annotations

from typing import Any

import torch

from stable_baselines3 import SAC

from algos.sb3_torch.hybrid_policy_v4 import DecisionAlignedSACPolicy
from algos.sb3_torch.replay_buffer import DictNStepReplayBuffer
from algos.sb3_torch.sac import SceneRepresentationSAC

from .simple_encoder import SimpleMlpLstmExtractor

HSAC_ALGORITHMS = ("hsac_mlp", "hsac_lstm")


def _backbone(algo: str) -> str:
    if algo == "hsac_mlp":
        return "mlp"
    if algo == "hsac_lstm":
        return "lstm"
    raise ValueError(f"Unknown HSAC algorithm {algo!r}")


def source_action_repeat_hsac(algo: str, requested: int | None = None) -> int:
    if algo not in HSAC_ALGORITHMS:
        raise ValueError(f"HSAC algorithm must be one of {HSAC_ALGORITHMS}")
    return 3 if requested is None else int(requested)


def make_hsac_model(
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
    """Construct an HSAC model (hybrid head + simple MLP/LSTM encoder)."""

    action_repeat = source_action_repeat_hsac(algo, action_repeat)
    if learning_rate is None:
        learning_rate = 1e-4

    feature_kwargs = {
        "features_dim": 128,
        "hidden_dim": 128,
        "backbone": _backbone(algo),
        "random_augmentation": scenario != "cross",
        "carla_contract": scenario == "carla",
    }
    policy_kwargs = {
        "features_extractor_class": SimpleMlpLstmExtractor,
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
        DecisionAlignedSACPolicy,
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
