"""SB3/PyTorch hyperparameters corresponding to ``configs/init_configs.py``."""

from __future__ import annotations

from typing import Any

import torch
from stable_baselines3 import PPO, SAC

from algos.sb3_torch import (
    DictNStepReplayBuffer,
    HierarchicalSceneExtractor,
    SceneRepSACPolicy,
    SceneRepresentationSAC,
    SourceSacLstmExtractor,
    TopoTemporalGraphExtractor,
    SourcePPO,
    SourcePpoCarlaExtractor,
    SourcePpoPolicy,
)
from envs.sumo.topology_graph import build_topology_graph


def _feature_kwargs(scenario: str) -> dict[str, Any]:
    return {
        "features_dim": 128,
        "num_heads": 2,
        "goal_heads": 2 if scenario in ("carla", "cross") else 1,
        "num_modes": 1,
        # These are the same scenario exceptions as the TensorFlow config.
        "random_augmentation": scenario != "cross",
        "carla_contract": scenario == "carla",
        "no_neighbor_future": scenario == "roundabout",
    }


def source_action_repeat(algo: str, requested: int | None = None) -> int:
    """Resolve the environment clock used by the released runners.

    Off-policy runners hold one policy action for three simulator ticks. PPO
    always exposes every raw tick to its rollout buffer; the CARLA runner
    performs its three-tick hold inside the policy/collector instead.
    """

    if algo not in (
        "scene_rep",
        "mst",
        "topo_scene",
        "topo_scene_balanced",
        "temporal_graph",
        "sac",
        "ppo",
    ):
        raise ValueError(f"Unknown algorithm {algo!r}")
    action_repeat = (1 if algo == "ppo" else 3) if requested is None else int(requested)
    if action_repeat <= 0:
        raise ValueError("action_repeat must be positive")
    if algo == "ppo" and action_repeat != 1:
        raise ValueError(
            "Source-equivalent PPO requires action_repeat=1; CARLA's three-step "
            "action hold is implemented inside SourcePPO"
        )
    return action_repeat


def make_model(
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
) -> SAC | PPO:
    """Construct a model without importing the legacy TensorFlow stack."""

    action_repeat = source_action_repeat(algo, action_repeat)
    if learning_rate is None:
        learning_rate = 5e-4 if algo == "ppo" else 1e-4

    feature_kwargs = _feature_kwargs(scenario)
    common_policy_kwargs = {
        "features_extractor_class": HierarchicalSceneExtractor,
        "features_extractor_kwargs": feature_kwargs,
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False,
    }
    if algo in ("scene_rep", "mst"):
        use_representation = algo == "scene_rep"
        policy_kwargs = {
            **common_policy_kwargs,
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
            # Raw warm-up/action selection is driven by the source-equivalent
            # callback; SB3's counter is in held-action decisions instead.
            learning_starts=0,
            batch_size=batch_size,
            tau=5e-3,
            gamma=discount,
            train_freq=(1, "step"),
            # The legacy runner updates once per raw simulator step while a
            # replay transition is appended every held-action decision.
            gradient_steps=action_repeat,
            n_steps=4,
            replay_buffer_class=DictNStepReplayBuffer,
            replay_buffer_kwargs={
                "n_steps": 4,
                "gamma": discount,
                "duplicate_episode_end_transition": use_representation,
                "source_action_repeat": action_repeat,
            },
            ent_coef="auto_0.2",
            target_entropy="auto",
            policy_kwargs=policy_kwargs,
            representation_coef=1.0 if use_representation else 0.0,
            representation_learning_rate=learning_rate,
            representation_heads=feature_kwargs["goal_heads"],
            representation_separate_target_projector=scenario == "cross",
            representation_online_target_encoder=scenario != "cross",
            action_embedding_dim=64,
            max_grad_norm=5.0,
            raw_learning_starts=learning_starts,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
        )
    if algo in ("topo_scene", "topo_scene_balanced", "temporal_graph"):
        use_topology = algo != "temporal_graph"
        normalize_slots = algo == "topo_scene_balanced"
        raw_env = env.unwrapped
        specification = getattr(raw_env, "specification", None)
        topology_graph = None
        topology_info = None
        if use_topology:
            if specification is None or not hasattr(specification, "network_path"):
                raise TypeError(
                    "topo_scene requires env.unwrapped.specification.network_path"
                )
            topology_graph, topology_info = build_topology_graph(
                specification.network_path,
                coordinate_offset=tuple(
                    getattr(specification, "coordinate_offset", (0.0, 0.0))
                ),
                return_info=True,
            )
        graph_feature_kwargs = {
            "topology_graph": topology_graph,
            "features_dim": 128,
            "hidden_dim": 128,
            "num_heads": 2,
            "random_augmentation": scenario != "cross",
            "carla_contract": scenario == "carla",
            "use_topology": use_topology,
            "normalize_slots": normalize_slots,
        }
        policy_kwargs = {
            "features_extractor_class": TopoTemporalGraphExtractor,
            "features_extractor_kwargs": graph_feature_kwargs,
            "activation_fn": torch.nn.ReLU,
            "normalize_images": False,
            "net_arch": {"pi": [128, 32], "qf": [128, 32]},
            "optimizer_class": torch.optim.NAdam,
            "optimizer_kwargs": {"eps": 1e-7},
            "n_critics": 2,
            "action_embedding_dim": 64,
        }
        model = SceneRepresentationSAC(
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
                "duplicate_episode_end_transition": True,
                "source_action_repeat": action_repeat,
            },
            ent_coef="auto_0.2",
            target_entropy="auto",
            policy_kwargs=policy_kwargs,
            structured_representation=True,
            representation_coef=1.0,
            representation_learning_rate=learning_rate,
            representation_heads=2,
            representation_separate_target_projector=False,
            representation_online_target_encoder=scenario != "cross",
            action_embedding_dim=64,
            max_grad_norm=5.0,
            raw_learning_starts=learning_starts,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
        )
        model.topology_graph_info = (
            topology_info.to_dict() if topology_info is not None else None
        )
        model.graph_ablation = (
            "full_topology_temporal_graph_with_balanced_slots"
            if normalize_slots
            else "full_topology_temporal_graph"
            if use_topology
            else "temporal_vehicle_graph_without_topology_query"
        )
        return model
    if algo == "sac":
        policy_kwargs = {
            "features_extractor_class": SourceSacLstmExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
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
    if algo == "ppo":
        ppo_policy_kwargs = (
            {"features_extractor_class": SourcePpoCarlaExtractor}
            if scenario == "carla"
            else None
        )
        return SourcePPO(
            SourcePpoPolicy,
            env,
            learning_rate=learning_rate,
            n_steps=512,
            batch_size=batch_size,
            n_epochs=10,
            gamma=discount,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=float("inf"),
            policy_kwargs=ppo_policy_kwargs,
            source_action_hold=3 if scenario == "carla" else 1,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
        )
    raise ValueError(
        f"Unknown algorithm {algo!r}; choose scene_rep, mst, topo_scene, "
        "topo_scene_balanced, temporal_graph, sac, or ppo"
    )
