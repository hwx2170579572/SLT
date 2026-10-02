from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest

from algos.sb3_torch.replay_buffer import DictNStepReplayBuffer
from algos.sb3_torch.replay_buffer_v4_7 import (
    HorizonCorrectDictNStepReplayBufferV47,
)
from configs import sb3_configs_v4_7 as config


def _spaces():
    return (
        gym.spaces.Dict(
            {"state": gym.spaces.Box(-100.0, 100.0, shape=(1,), dtype=np.float32)}
        ),
        gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
    )


def _add(buffer, step: int, reward: float, done: bool, *, timeout: bool = False) -> None:
    buffer.add(
        {"state": np.array([[step]], dtype=np.float32)},
        {"state": np.array([[step + 1]], dtype=np.float32)},
        np.array([[0.0]], dtype=np.float32),
        np.array([reward], dtype=np.float32),
        np.array([done]),
        [{"TimeLimit.truncated": timeout}],
    )


def test_horizon_correct_full_window_uses_gamma_power_n() -> None:
    observation_space, action_space = _spaces()
    buffer = HorizonCorrectDictNStepReplayBufferV47(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        gamma=0.9,
    )
    for step, reward in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
        _add(buffer, step, reward, False)
    sample = buffer._get_samples(np.array([0]))
    assert sample.rewards.item() == pytest.approx(
        1.0 + 0.9 * 2.0 + 0.9**2 * 3.0 + 0.9**3 * 4.0
    )
    assert sample.discounts.item() == pytest.approx(0.9**4)
    assert sample.next_observations["state"].item() == pytest.approx(4.0)
    assert sample.one_step_next_observations["state"].item() == pytest.approx(1.0)
    assert buffer.last_sample_actual_horizons.tolist() == [4]


def test_horizon_correct_terminal_tail_uses_actual_horizon() -> None:
    observation_space, action_space = _spaces()
    buffer = HorizonCorrectDictNStepReplayBufferV47(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        gamma=0.9,
    )
    _add(buffer, 0, 1.0, False)
    _add(buffer, 1, 2.0, False)
    _add(buffer, 2, 3.0, True)
    sample = buffer._get_samples(np.array([0]))
    assert sample.rewards.item() == pytest.approx(1.0 + 0.9 * 2.0 + 0.9**2 * 3.0)
    assert sample.discounts.item() == pytest.approx(0.9**3)
    assert sample.dones.item() == pytest.approx(1.0)
    assert buffer.last_sample_actual_horizons.tolist() == [3]
    diagnostics = buffer.return_estimator_diagnostics()
    assert diagnostics["horizon_correct_bootstrap"] is True
    assert diagnostics["mean_sampled_actual_horizon"] == pytest.approx(3.0)
    assert diagnostics["short_horizon_sample_rate"] == pytest.approx(1.0)


def test_horizon_correct_timeout_preserves_parent_bootstrap_semantics() -> None:
    observation_space, action_space = _spaces()
    buffer = HorizonCorrectDictNStepReplayBufferV47(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        gamma=0.9,
    )
    _add(buffer, 0, 0.0, False)
    _add(buffer, 1, 0.0, False)
    _add(buffer, 2, 0.0, True, timeout=True)
    sample = buffer._get_samples(np.array([0]))
    assert sample.discounts.item() == pytest.approx(0.9**3)
    assert sample.dones.item() == pytest.approx(0.0)
    assert buffer.last_sample_actual_horizons.tolist() == [3]


def test_v47_factory_replaces_only_empty_candidate_buffer() -> None:
    observation_space, action_space = _spaces()
    parent_buffer = DictNStepReplayBuffer(
        20_000,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        gamma=0.99,
    )
    fake = SimpleNamespace(
        replay_buffer=parent_buffer,
        replay_buffer_class=DictNStepReplayBuffer,
        replay_buffer_kwargs={},
        buffer_size=20_000,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        n_envs=1,
        optimize_memory_usage=False,
        n_steps=4,
        graph_ablation="parent",
        v4_method_metadata={"implementation_id": "parent"},
    )
    with mock.patch.object(config, "make_model_v4_6", return_value=fake) as parent:
        observed = config.make_model_v4_7(
            config.V47_CANDIDATE,
            object(),
            scenario="cross",
            action_repeat=3,
        )
    assert observed is fake
    assert parent.call_args.args[0] == config.PARENT_CONTROL
    assert isinstance(observed.replay_buffer, HorizonCorrectDictNStepReplayBufferV47)
    assert observed.n_steps == 16
    assert observed.replay_buffer.n_steps == 16
    assert observed.replay_buffer.requested_buffer_size == 20_000
    assert observed.v4_method_metadata["single_change"] == (
        "horizon_correct_16_step_terminal_credit"
    )
    assert observed.v4_method_metadata["reward_changed"] is False


def test_v47_matched_parent_control_keeps_original_buffer() -> None:
    fake = SimpleNamespace(
        replay_buffer=object(),
        graph_ablation="parent",
        v4_method_metadata={"implementation_id": "parent"},
    )
    with mock.patch.object(config, "make_model_v4_6", return_value=fake):
        observed = config.make_model_v4_7(
            config.PARENT_CONTROL,
            object(),
            scenario="cross",
            action_repeat=3,
        )
    assert observed.replay_buffer is fake.replay_buffer
    assert observed.v4_method_metadata["return_estimator_changed"] is False
    assert observed.v4_method_metadata["n_step"] == 4
