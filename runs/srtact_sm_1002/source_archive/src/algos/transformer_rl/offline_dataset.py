"""Shared offline dataset collector for the Transformer+RL baselines.

Rolls out a trained (or random) SUMO policy and records ``(state, action,
reward, return_to_go, timestep, done)`` transitions.  The state tokenizer and
action mapper are injected so the same collector serves both the continuous
Decision Transformer (``act_dim=2``) and the hybrid-action Decision Transformer
(``act_dim=4``: speed + lane one-hot).

Collection is inference-only.  The full offline training/tuning pipeline for DT
and Hybrid-DT is intentionally left to a follow-up step; this module provides
the exact data contract they will consume.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def continuous_action_to_vector(action: Any) -> np.ndarray:
    """Keep the Box(2) ``[speed, lateral]`` action unchanged (``act_dim=2``)."""
    return np.asarray(action, dtype=np.float32).reshape(-1)


def hybrid_action_to_vector(action: Any) -> np.ndarray:
    """Map ``[speed, lateral]`` to ``[speed, lane_one_hot(3)]`` (``act_dim=4``).

    The lateral threshold semantics match ``SumoSceneEnv.adapt_action`` and
    ``algos.sb3_torch.hybrid_policy_v4.hybrid_action_features``: lateral below
    -1/3 is the first lane command, above 1/3 the third, otherwise keep.
    """
    speed = float(np.asarray(action)[0])
    lateral = float(np.asarray(action)[1])
    if lateral < -1.0 / 3.0:
        lane = 0
    elif lateral > 1.0 / 3.0:
        lane = 2
    else:
        lane = 1
    one_hot = np.eye(3, dtype=np.float32)[lane]
    return np.concatenate([[speed], one_hot]).astype(np.float32)


def collect_episode_transitions(
    model: Any,
    env: Any,
    state_tokenizer: Callable[[Any], np.ndarray],
    action_mapper: Callable[[Any], np.ndarray],
    *,
    deterministic: bool = True,
    max_steps: int = 1000,
) -> list[dict[str, Any]]:
    """Roll out one episode and return its transitions."""

    transitions: list[dict[str, Any]] = []
    observation, _ = env.reset()
    for timestep in range(max_steps):
        state = state_tokenizer(observation)
        action = model.predict(observation, deterministic=deterministic)[0]
        action_vector = action_mapper(action)
        next_observation, reward, terminated, truncated, _info = env.step(action)
        transitions.append(
            {
                "state": np.asarray(state, dtype=np.float32),
                "action": np.asarray(action_vector, dtype=np.float32),
                "reward": float(reward),
                "done": bool(terminated or truncated),
                "timestep": timestep,
            }
        )
        observation = next_observation
        if terminated or truncated:
            break
    return transitions


def compute_returns_to_go(transitions: list[dict[str, Any]], discount: float = 0.99) -> np.ndarray:
    """Backward cumulative discounted return for every transition."""
    returns = np.zeros(len(transitions), dtype=np.float32)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        running = transitions[index]["reward"] + discount * running
        returns[index] = running
    return returns


def collect_dataset(
    model: Any,
    env: Any,
    state_tokenizer: Callable[[Any], np.ndarray],
    action_mapper: Callable[[Any], np.ndarray],
    *,
    episodes: int,
    discount: float = 0.99,
    deterministic: bool = True,
    max_steps: int = 1000,
) -> dict[str, np.ndarray]:
    """Collect ``episodes`` rollouts and pack them into a flat dataset."""

    transitions: list[dict[str, Any]] = []
    for _ in range(episodes):
        transitions.extend(
            collect_episode_transitions(
                model,
                env,
                state_tokenizer,
                action_mapper,
                deterministic=deterministic,
                max_steps=max_steps,
            )
        )
    returns_to_go = compute_returns_to_go(transitions, discount)
    return {
        "states": np.stack([t["state"] for t in transitions]).astype(np.float32),
        "actions": np.stack([t["action"] for t in transitions]).astype(np.float32),
        "rewards": np.asarray([t["reward"] for t in transitions], dtype=np.float32),
        "returns_to_go": returns_to_go.astype(np.float32),
        "timesteps": np.asarray([t["timestep"] for t in transitions], dtype=np.int64),
        "dones": np.asarray([t["done"] for t in transitions], dtype=np.float32),
    }
