from __future__ import annotations

import math
import json
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO as Sb3PPO
from stable_baselines3.common.callbacks import EvalCallback as Sb3EvalCallback

from algos.sb3_torch import (
    BestTrainingSuccessCallback,
    DictNStepReplayBuffer,
    FutureRepresentationObjective,
    HierarchicalSceneExtractor,
    RawStepControlCallback,
    SceneRepresentationSAC,
    SourceEvaluationCallback,
    SourceSacLstmExtractor,
    SourcePPO,
    SourcePpoCarlaExtractor,
    SourcePpoPolicy,
)
from configs.sb3_configs import make_model, source_action_repeat
from algos.sb3_torch.features import (
    KerasStyleMultiheadAttention,
    _nonzero_mask,
    _safe_attention,
    keras_initialize_linear,
)
from algos.sb3_torch.evaluation import (
    evaluate_model,
    evaluate_model_detailed,
    source_evaluation_augmentation,
)
from tools.train_sb3 import (
    _audit_checkpoints,
    _decision_steps,
    _model_learning_starts,
    _source_ppo_collected_steps,
    _source_ppo_test_checkpoint_step,
)
from algos.sb3_torch.sac import _clip_gradients_like_keras


class _DetailedEvaluationModel:
    policy = None

    def __init__(self) -> None:
        self.predictions = 0

    def predict(self, observation: Any, deterministic: bool = True):
        assert deterministic
        self.predictions += 1
        return np.zeros(2, dtype=np.float32), None


class _DetailedEvaluationEnv:
    def reset(self, *, seed: int):
        self.seed = seed
        self.steps = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        del action
        self.steps += 1
        target_steps = 2 if self.seed == 10 else 3
        done = self.steps == target_steps
        success = done and self.seed == 10
        collision = done and self.seed == 11
        info = {
            "undiscounted_reward": 1.0,
            "raw_simulation_steps": self.steps,
            "is_success": success,
            "collision": collision,
            "off_route": False,
            "max_time": False,
            "traffic_variant": f"traffic_{self.seed}.rou.xml",
        }
        return np.zeros(1, dtype=np.float32), -99.0, done, False, info


def test_detailed_evaluation_uses_only_successful_raw_completion_times() -> None:
    report = evaluate_model_detailed(
        _DetailedEvaluationModel(), _DetailedEvaluationEnv(), episodes=2, seed=10
    )
    assert report.summary.mean_return == pytest.approx(2.5)
    assert report.summary.mean_raw_steps == pytest.approx(2.5)
    assert report.summary.success_rate == pytest.approx(0.5)
    assert report.summary.collision_rate == pytest.approx(0.5)
    assert report.successful_episodes == 1
    assert report.mean_success_completion_time_seconds == pytest.approx(0.2)
    assert report.std_success_completion_time_seconds == pytest.approx(0.0)
    assert report.episode_records[0].completion_time_seconds == pytest.approx(0.2)
    assert report.episode_records[0].environment_steps == 2
    assert report.episode_records[1].completion_time_seconds is None


def test_summary_evaluation_holds_carla_ppo_action_on_raw_clock() -> None:
    model = _DetailedEvaluationModel()
    summary = evaluate_model(
        model,
        _DetailedEvaluationEnv(),
        episodes=1,
        seed=10,
        policy_action_hold=3,
    )
    assert model.predictions == 1
    assert summary.mean_decision_steps == pytest.approx(1.0)
    assert summary.mean_raw_steps == pytest.approx(2.0)
    assert summary.success_rate == pytest.approx(1.0)


def _space(carla: bool = False) -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "trajectory": gym.spaces.Box(
                -1000.0, 1000.0, shape=(6, 10, 5), dtype=np.float32
            ),
            "map": gym.spaces.Box(
                -1000.0,
                1000.0,
                shape=(18, 10, 2) if carla else (12, 10, 5),
                dtype=np.float32,
            ),
        }
    )


def _batch(carla: bool = False, batch_size: int = 3) -> dict[str, torch.Tensor]:
    trajectory = torch.zeros(batch_size, 6, 10, 5)
    trajectory[:, 0, :4, 0] = torch.arange(1, 5)
    trajectory[:, 0, :4, 1] = 10.0
    trajectory[:, 0, :4, 3] = 2.0
    trajectory[:, 1, :3, 0] = torch.arange(4, 7)
    trajectory[:, 1, :3, 1] = 12.0
    if carla:
        map_state = torch.zeros(batch_size, 18, 10, 2)
    else:
        map_state = torch.zeros(batch_size, 12, 10, 5)
    map_state[:, :2, :, 0] = torch.arange(1, 11)
    map_state[:, :2, :, 1] = 10.0
    if not carla:
        map_state[:, :2, :, 3] = 1.0
    return {"trajectory": trajectory, "map": map_state}


def _sac_space() -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "state_lstm": gym.spaces.Box(
                -1000.0, 1000.0, shape=(10, 30), dtype=np.float32
            ),
            "state_lstm_mask": gym.spaces.Box(
                0.0, 1.0, shape=(10,), dtype=np.float32
            ),
        }
    )


def _sac_batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    state_lstm = torch.zeros(batch_size, 10, 30)
    state_lstm[:, :4, 0] = torch.arange(1, 5)
    state_lstm[:, :4, 2] = 2.0
    state_lstm[:, :4, 4] = 0.25
    state_lstm_mask = torch.zeros(batch_size, 10)
    state_lstm_mask[:, :4] = 1.0
    return {
        "state_lstm": state_lstm,
        "state_lstm_mask": state_lstm_mask,
    }


@pytest.mark.parametrize("carla", [False, True])
def test_feature_extractor_shape_and_finiteness(carla: bool) -> None:
    extractor = HierarchicalSceneExtractor(
        _space(carla), random_augmentation=False, carla_contract=carla
    )
    output = extractor(_batch(carla))
    assert output.shape == (3, 128)
    assert torch.isfinite(output).all()


def test_carla_neighbor_paths_preserve_released_i_times_two_indexing() -> None:
    extractor = HierarchicalSceneExtractor(
        _space(True), random_augmentation=False, carla_contract=True
    )
    features = torch.arange(18, dtype=torch.float32).reshape(1, 18, 1).repeat(
        1, 1, extractor.feature_dim
    )
    valid = torch.ones(1, 6, 3, dtype=torch.bool)
    grouped_features = features.reshape(1, 6, 3, extractor.feature_dim)
    selected, selected_valid = extractor._source_neighbor_maps(
        grouped_features, valid
    )
    assert selected.shape == (1, 5, 2, extractor.feature_dim)
    assert selected[0, :, :, 0].tolist() == [
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ]
    assert selected_valid.all()


def test_batched_neighbor_attention_matches_per_neighbor_loop() -> None:
    torch.manual_seed(17)
    batch, neighbors, paths, feature_dim = 3, 5, 2, 16
    layer = KerasStyleMultiheadAttention(
        feature_dim,
        feature_dim,
        feature_dim,
        num_heads=2,
        head_dim=feature_dim,
        output_dim=feature_dim,
    )
    actor_features = torch.randn(batch, neighbors, feature_dim)
    map_features = torch.randn(batch, neighbors, paths, feature_dim)
    actor_valid = torch.tensor(
        [[True, True, False, True, False], [True] * 5, [False] * 5]
    )
    path_valid = torch.tensor(
        [
            [[True, True], [True, False], [False, False], [False, True], [False, False]],
            [[True, False]] * 5,
            [[False, False]] * 5,
        ]
    )

    loop_outputs = []
    for neighbor_index in range(neighbors):
        query = actor_features[:, neighbor_index : neighbor_index + 1]
        keys = torch.cat([query, map_features[:, neighbor_index]], dim=1)
        key_valid = torch.cat(
            [actor_valid[:, neighbor_index : neighbor_index + 1], path_valid[:, neighbor_index]],
            dim=1,
        )
        related = torch.relu(_safe_attention(layer, query, keys, key_valid)).squeeze(1)
        loop_outputs.append(related * actor_valid[:, neighbor_index : neighbor_index + 1])
    loop_result = torch.stack(loop_outputs, dim=1)

    flat_query = actor_features.reshape(batch * neighbors, 1, feature_dim)
    flat_keys = torch.cat(
        [actor_features[:, :, None, :], map_features], dim=2
    ).reshape(batch * neighbors, 1 + paths, feature_dim)
    flat_valid = torch.cat(
        [actor_valid[:, :, None], path_valid], dim=2
    ).reshape(batch * neighbors, 1 + paths)
    batched_result = torch.relu(
        _safe_attention(layer, flat_query, flat_keys, flat_valid)
    ).squeeze(1).reshape(batch, neighbors, feature_dim)
    batched_result = batched_result * actor_valid[:, :, None]
    assert torch.allclose(batched_result, loop_result, atol=1e-6, rtol=1e-6)


def test_source_mask_uses_only_first_coordinate() -> None:
    values = torch.tensor(
        [[[0.0, 5.0], [2.0, 0.0], [1e-12, 0.0], [float("nan"), 0.0]]]
    )
    # tf.not_equal(x, 0) is exact: tiny nonzero values and NaN are both valid.
    assert _nonzero_mask(values).tolist() == [[False, True, True, True]]


def test_source_action_repeat_defaults_follow_each_runner_clock() -> None:
    assert source_action_repeat("scene_rep") == 3
    assert source_action_repeat("mst") == 3
    assert source_action_repeat("sac") == 3
    assert source_action_repeat("ppo") == 1
    with pytest.raises(ValueError, match="requires action_repeat=1"):
        source_action_repeat("ppo", 3)


def test_source_heading_difference_is_not_wrapped() -> None:
    extractor = HierarchicalSceneExtractor(
        _space(False), random_augmentation=False, carla_contract=False
    )
    trajectories = torch.zeros(1, 6, 10, 5)
    trajectories[0, 0, 0] = torch.tensor([1.0, 1.0, 3.0, 0.0, 0.0])
    trajectories[0, 1, 0] = torch.tensor([2.0, 1.0, -3.0, 0.0, 0.0])
    valid = _nonzero_mask(trajectories)
    rotated, _, _ = extractor._rotate_trajectories(trajectories, valid)
    assert rotated[0, 1, 0, 2].item() == pytest.approx(-6.0)


def test_keras_dense_initialization_contract() -> None:
    torch.manual_seed(23)
    linear = torch.nn.Linear(64, 32)
    keras_initialize_linear(linear)
    limit = np.sqrt(6.0 / (64 + 32))
    assert torch.max(torch.abs(linear.weight)).item() <= limit + 1e-7
    assert torch.count_nonzero(linear.bias).item() == 0


def test_keras_mha_einsum_initialization_contract() -> None:
    torch.manual_seed(29)
    layer = KerasStyleMultiheadAttention(
        5,
        5,
        5,
        num_heads=2,
        head_dim=128,
        output_dim=128,
    )
    # TensorFlow's logical query kernel is [5, 2, 128], so its generic
    # rank-three fan calculation gives fan_in=10 and fan_out=640.
    query_limit = np.sqrt(6.0 / (10 + 640))
    assert torch.max(torch.abs(layer.query_projection.weight)).item() <= (
        query_limit + 1e-7
    )
    # The output kernel is [2, 128, 128], giving fan_in=fan_out=256.
    output_limit = np.sqrt(6.0 / (256 + 256))
    assert torch.max(torch.abs(layer.output_projection.weight)).item() <= (
        output_limit + 1e-7
    )


def test_keras_all_masked_query_row_remains_uniform_and_finite() -> None:
    torch.manual_seed(31)
    layer = KerasStyleMultiheadAttention(
        3,
        3,
        3,
        num_heads=2,
        head_dim=4,
        output_dim=5,
    )
    query = torch.randn(1, 2, 3)
    key = torch.randn(1, 3, 3)
    attention_mask = torch.tensor(
        [[[True, False, True], [False, False, False]]]
    )
    output, weights = layer(
        query,
        key,
        key,
        attention_mask=attention_mask,
        need_weights=True,
    )
    assert weights is not None
    assert torch.isfinite(output).all()
    assert torch.allclose(
        weights[:, :, 1],
        torch.full_like(weights[:, :, 1], 1.0 / 3.0),
        atol=1e-7,
        rtol=0.0,
    )


def test_source_rotation_augmentation_is_independent_of_torch_eval_mode() -> None:
    extractor = HierarchicalSceneExtractor(
        _space(False), random_augmentation=True, carla_contract=False
    )
    extractor.eval()
    torch.manual_seed(37)
    enabled = extractor._rotation_angle(8, torch.float32, torch.device("cpu"))
    assert torch.count_nonzero(enabled).item() > 0
    extractor.set_source_augmentation(False)
    disabled = extractor._rotation_angle(8, torch.float32, torch.device("cpu"))
    assert torch.count_nonzero(disabled).item() == 0


def test_keras_clipnorm_is_per_gradient_tensor() -> None:
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(2))
    first.grad = torch.tensor([6.0, 8.0])
    second.grad = torch.tensor([6.0, 8.0])
    _clip_gradients_like_keras([first, second], 5.0)
    assert torch.linalg.vector_norm(first.grad).item() == pytest.approx(5.0)
    assert torch.linalg.vector_norm(second.grad).item() == pytest.approx(5.0)
    assert torch.linalg.vector_norm(
        torch.cat([first.grad, second.grad])
    ).item() > 5.0


class DummySceneEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    def __init__(self) -> None:
        self.observation_space = _space(False)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.steps = 0

    def _observation(self) -> dict[str, np.ndarray]:
        batch = _batch(False, 1)
        return {key: value[0].numpy().astype(np.float32) for key, value in batch.items()}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.steps = 0
        return self._observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.steps += 1
        return self._observation(), 0.0, False, self.steps >= 8, {}


class DummyRawStepSceneEnv(DummySceneEnv):
    def __init__(self) -> None:
        super().__init__()
        self.lifetime_raw_steps = 0
        self.raw_budget: int | None = None

    def set_lifetime_raw_step_budget(self, raw_steps: int | None) -> None:
        self.lifetime_raw_steps = 0
        self.raw_budget = raw_steps

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        del action
        remaining = (
            3
            if self.raw_budget is None
            else max(0, self.raw_budget - self.lifetime_raw_steps)
        )
        executed = min(3, remaining)
        self.lifetime_raw_steps += executed
        self.steps += 1
        return self._observation(), 0.0, False, False, {
            "raw_steps_executed": executed,
            "lifetime_raw_simulation_steps": self.lifetime_raw_steps,
        }


class DummySacSceneEnv(DummySceneEnv):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = _sac_space()

    def _observation(self) -> dict[str, np.ndarray]:
        batch = _sac_batch(1)
        return {
            key: value[0].numpy().astype(np.float32)
            for key, value in batch.items()
        }


class DummyRawStepSacEnv(DummyRawStepSceneEnv):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = _sac_space()

    def _observation(self) -> dict[str, np.ndarray]:
        batch = _sac_batch(1)
        return {
            key: value[0].numpy().astype(np.float32)
            for key, value in batch.items()
        }


class DummyRgbEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            0.0, 255.0, shape=(80, 80, 3), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros((80, 80, 3), dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        return np.zeros((80, 80, 3), dtype=np.float32), 0.0, False, False, {}


class OneStepSuccessEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(1,), dtype=np.float32
        )
        self.completed = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        self.completed += 1
        success = self.completed % 3 != 0
        return (
            np.zeros(1, dtype=np.float32),
            float(success),
            True,
            False,
            {"is_success": success},
        )


class RecordingCarlaPpoEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            -1000.0, 1000.0, shape=(6, 10, 5), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.actions: list[np.ndarray] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.actions.clear()
        observation = np.zeros((6, 10, 5), dtype=np.float32)
        observation[0, 0, 0] = 1.0
        return observation, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.actions.append(np.asarray(action).copy())
        observation = np.zeros((6, 10, 5), dtype=np.float32)
        observation[0, 0, 0] = 1.0 + len(self.actions)
        return observation, 0.0, False, False, {}


def test_dict_n_step_returns() -> None:
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)}
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    buffer = DictNStepReplayBuffer(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        gamma=0.9,
    )
    for step, reward in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
        buffer.add(
            {"state": np.array([[step]], dtype=np.float32)},
            {"state": np.array([[step + 1]], dtype=np.float32)},
            np.array([[0.0]], dtype=np.float32),
            np.array([reward], dtype=np.float32),
            np.array([False]),
            [{}],
        )
        assert buffer.size() == max(0, step - 2)
    sample = buffer._get_samples(np.array([0]))
    expected = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0 + 0.9**3 * 4.0
    assert sample.rewards.item() == pytest.approx(expected)
    # Released SAC bootstraps with one gamma after cpprb's N-step reward.
    assert sample.discounts.item() == pytest.approx(0.9)
    assert sample.next_observations["state"].item() == pytest.approx(4.0)
    assert sample.one_step_next_observations["state"].item() == pytest.approx(1.0)


def test_dict_n_step_flushes_terminal_tail_and_keeps_global_capacity() -> None:
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-100.0, 100.0, shape=(1,), dtype=np.float32)}
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    buffer = DictNStepReplayBuffer(
        4,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        gamma=0.9,
    )

    for step in range(10):
        buffer.add(
            {"state": np.array([[step]], dtype=np.float32)},
            {"state": np.array([[step + 1]], dtype=np.float32)},
            np.array([[0.0]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([False]),
            [{}],
        )
    assert buffer.size() == 4
    active = buffer._finalized_slots[: buffer.size()]
    assert set(buffer._insertion_ids[active].tolist()) == {3, 4, 5, 6}

    buffer.add(
        {"state": np.array([[10]], dtype=np.float32)},
        {"state": np.array([[11]], dtype=np.float32)},
        np.array([[0.0]], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        np.array([True]),
        [{}],
    )
    assert buffer.size() == 4
    active = buffer._finalized_slots[: buffer.size()]
    assert set(buffer._insertion_ids[active].tolist()) == {7, 8, 9, 10}
    terminal_sample = buffer._get_samples(
        np.asarray([int(np.flatnonzero(buffer._insertion_ids == 10)[0])])
    )
    assert terminal_sample.rewards.item() == pytest.approx(1.0)
    assert terminal_sample.dones.item() == pytest.approx(1.0)


def test_source_prediction_queue_duplicates_only_the_terminal_transition() -> None:
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)}
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    buffer = DictNStepReplayBuffer(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        duplicate_episode_end_transition=True,
    )

    def add(step: int, done: bool) -> None:
        buffer.add(
            {"state": np.array([[step]], dtype=np.float32)},
            {"state": np.array([[step + 1]], dtype=np.float32)},
            np.array([[0.25]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([done]),
            [{}],
        )

    add(0, False)
    assert buffer._next_insertion_id == 1
    add(1, True)
    assert buffer._next_insertion_id == 3
    assert buffer.size() == 3
    active = buffer._finalized_slots[: buffer.size()]
    assert set(buffer._insertion_ids[active].tolist()) == {0, 1, 2}
    np.testing.assert_array_equal(buffer.observations["state"][1], buffer.observations["state"][2])
    np.testing.assert_array_equal(buffer.next_observations["state"][1], buffer.next_observations["state"][2])
    np.testing.assert_array_equal(buffer.actions[1], buffer.actions[2])
    np.testing.assert_array_equal(buffer.rewards[1], buffer.rewards[2])
    np.testing.assert_array_equal(buffer.dones[1], buffer.dones[2])


def test_source_budget_partial_action_updates_without_replay_insertion() -> None:
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)}
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    buffer = DictNStepReplayBuffer(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        source_action_repeat=3,
    )

    def add(step: int, raw_steps: int, done: bool) -> None:
        buffer.add(
            {"state": np.array([[step]], dtype=np.float32)},
            {"state": np.array([[step + 1]], dtype=np.float32)},
            np.array([[0.25]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([done]),
            [{"raw_steps_executed": raw_steps}],
        )

    add(0, 3, False)
    assert buffer._next_insertion_id == 1
    add(1, 2, False)
    assert buffer._next_insertion_id == 1
    assert buffer.source_incomplete_actions_skipped == 1
    # An early episode boundary is still a source replay boundary even if it
    # happens before all three held-action ticks execute.
    add(2, 1, True)
    assert buffer._next_insertion_id == 2


def test_raw_step_conversion() -> None:
    assert _decision_steps(0, 3) == 0
    assert _decision_steps(1, 3) == 1
    assert _decision_steps(5_000, 3) == 1_667
    assert _model_learning_starts(
        5_000, 3, source_raw_step_control=True
    ) == 5_000
    assert _model_learning_starts(
        5_000, 3, source_raw_step_control=False
    ) == 1_667


def test_source_ppo_finishes_horizon_that_crosses_paper_budget() -> None:
    assert _source_ppo_collected_steps(100_000) == 100_352
    assert _source_ppo_collected_steps(512) == 512
    assert _source_ppo_test_checkpoint_step(100_000, 20_000) == 100_000
    assert _source_ppo_test_checkpoint_step(100_100, 20_000) == 100_000


def test_best_training_success_callback_captures_paper_selection_rule(
    tmp_path: Any,
) -> None:
    output_path = tmp_path / "best_training_success_model"
    model = Sb3PPO(
        "MlpPolicy",
        OneStepSuccessEnv(),
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        device="cpu",
        verbose=0,
    )
    callback = BestTrainingSuccessCallback(output_path)
    model.learn(total_timesteps=24, callback=callback)
    assert output_path.with_suffix(".zip").is_file()
    metadata = json.loads(
        output_path.with_name("best_training_success.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["paper_rule_reconstruction"] is True
    assert metadata["fixed_denominator"] == 20
    assert metadata["episodes_completed"] >= 21
    assert 0.0 <= metadata["success_rate_last_20"] <= 1.0


def test_raw_step_callback_enforces_exact_budget_and_update_count() -> None:
    env = DummyRawStepSceneEnv()
    model = make_model(
        "scene_rep",
        env,
        scenario="left_turn",
        learning_starts=14,
        buffer_size=64,
        batch_size=2,
        action_repeat=3,
        device="cpu",
        verbose=0,
    )
    callback = RawStepControlCallback(raw_step_budget=16)
    model.learn(total_timesteps=16, callback=callback)
    assert callback.total_raw_steps == 16
    assert env.lifetime_raw_steps == 16
    assert model.num_timesteps == 6
    # Source updates at raw steps 14..16 inclusive.
    assert model._n_updates == 3
    assert model.replay_buffer.source_incomplete_actions_skipped == 1


def test_raw_step_checkpoint_captures_exact_mid_hold_learner_state(tmp_path) -> None:
    """A source checkpoint may fall between two held-action boundaries."""

    env = DummyRawStepSceneEnv()
    model = make_model(
        "scene_rep",
        env,
        scenario="left_turn",
        learning_starts=14,
        buffer_size=64,
        batch_size=2,
        action_repeat=3,
        device="cpu",
        verbose=0,
    )
    run_dir = tmp_path / "run"
    callback = RawStepControlCallback(
        raw_step_budget=16,
        checkpoint_frequency=14,
        checkpoint_path=run_dir / "checkpoints",
        checkpoint_prefix="exact",
    )
    model.learn(total_timesteps=16, callback=callback)

    checkpoint = run_dir / "checkpoints" / "exact_raw_14_steps.zip"
    assert checkpoint.is_file()
    restored = SceneRepresentationSAC.load(checkpoint, device="cpu")
    # The released runner updates at step 14 and saves immediately. It must
    # not include the step-15 update merely because both raw steps were
    # executed inside one SB3 environment call.
    assert restored._raw_steps_seen == 14
    assert restored._n_updates == 1
    assert model._raw_steps_seen == 16
    assert model._n_updates == 3
    audit = _audit_checkpoints(
        run_dir,
        algorithm="exact",
        scenario="left_turn",
        requested_raw_steps=16,
        checkpoint_frequency=14,
        action_repeat=3,
        source_raw_step_control=True,
    )
    assert audit["expected_checkpoint_count"] == 1
    assert audit["checkpoints"][0]["recorded_clock"] == 14
    assert audit["checkpoints"][0]["zip_crc_ok"] is True


def test_raw_step_updates_use_old_buffer_before_final_transition() -> None:
    env = DummyRawStepSceneEnv()
    model = make_model(
        "scene_rep",
        env,
        scenario="left_turn",
        learning_starts=14,
        buffer_size=64,
        batch_size=2,
        action_repeat=3,
        device="cpu",
        verbose=0,
    )
    records: list[tuple[int, int | None, int]] = []
    original_train = model.train

    def recorded_train(gradient_steps: int, batch_size: int = 64) -> None:
        records.append(
            (
                gradient_steps,
                model._pending_raw_gradient_steps,
                model.replay_buffer.size(),
            )
        )
        original_train(gradient_steps, batch_size)

    model.train = recorded_train  # type: ignore[method-assign]
    model.learn(
        total_timesteps=16,
        callback=RawStepControlCallback(raw_step_budget=16),
    )
    # Raw step 14 is updated in the callback before decision five is stored.
    assert (1, None, 1) in records
    # Raw step 15 is the final held-action step and is updated after storage.
    assert any(pending == 1 and size == 2 for _, pending, size in records)


def test_formal_evaluation_temporarily_disables_source_augmentation() -> None:
    env = DummySceneEnv()
    model = make_model(
        "scene_rep",
        env,
        scenario="left_turn",
        learning_starts=5,
        buffer_size=64,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    extractors = [
        module
        for module in model.policy.modules()
        if isinstance(module, HierarchicalSceneExtractor)
    ]
    assert extractors and all(
        module.source_augmentation_enabled for module in extractors
    )
    with source_evaluation_augmentation(model):
        assert all(
            not module.source_augmentation_enabled for module in extractors
        )
    assert all(module.source_augmentation_enabled for module in extractors)


def test_periodic_evaluation_uses_source_test_augmentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = DummySceneEnv()
    model = make_model(
        "scene_rep",
        env,
        scenario="left_turn",
        learning_starts=5,
        buffer_size=64,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    extractors = [
        module
        for module in model.policy.modules()
        if isinstance(module, HierarchicalSceneExtractor)
    ]
    observed: list[bool] = []

    def probe(_callback: Sb3EvalCallback) -> bool:
        observed.append(
            bool(extractors)
            and all(not module.source_augmentation_enabled for module in extractors)
        )
        return True

    monkeypatch.setattr(Sb3EvalCallback, "_on_step", probe)
    callback = SourceEvaluationCallback(
        env, eval_freq=1, evaluation_seed_start=10_000
    )
    callback.init_callback(model)
    assert callback._on_step()
    assert observed == [True]
    assert callback.eval_env._seeds == [10_000]
    assert all(module.source_augmentation_enabled for module in extractors)


def test_separate_target_projector_polyak_update() -> None:
    objective = FutureRepresentationObjective(
        feature_dim=128,
        action_dim=2,
        separate_target_projector=True,
    )
    assert objective.target_projector is not None
    for online, target in zip(
        objective.projector.parameters(), objective.target_projector.parameters()
    ):
        assert torch.equal(online, target)
    target_before = [parameter.detach().clone() for parameter in objective.target_projector.parameters()]
    with torch.no_grad():
        for parameter in objective.projector.parameters():
            parameter.add_(1.0)
    objective.update_target(0.25)
    for online, before, target in zip(
        objective.projector.parameters(),
        target_before,
        objective.target_projector.parameters(),
    ):
        assert torch.allclose(target, before * 0.75 + online * 0.25)
        assert not target.requires_grad


def test_scene_rep_policy_build_train_save_load(tmp_path: Any) -> None:
    env = DummyRawStepSceneEnv()
    model = make_model(
        "scene_rep",
        env,
        scenario="left_turn",
        learning_starts=14,
        buffer_size=64,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    extractor = model.critic.features_extractor
    assert extractor.goal_heads == 1
    assert not extractor.no_neighbor_future
    assert model.gradient_steps == 3
    assert model.representation.target_projector is None
    assert model.representation_online_target_encoder
    assert model.representation.temporal_attention.num_heads == 1
    assert model.replay_buffer.duplicate_episode_end_transition
    assert model.ent_coef_optimizer is not None
    assert model.ent_coef_optimizer.param_groups[0]["betas"] == (0.5, 0.999)
    assert model.ent_coef_optimizer.param_groups[0]["eps"] == pytest.approx(1e-7)
    assert model.actor.optimizer.param_groups[0]["eps"] == pytest.approx(1e-7)
    assert model.critic.optimizer.param_groups[0]["eps"] == pytest.approx(1e-7)
    assert model.representation_optimizer.param_groups[0]["eps"] == pytest.approx(1e-7)
    observation, _ = env.reset(seed=3)
    action, _ = model.predict(observation, deterministic=True)
    assert action.shape == (2,)
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    target_modes: list[bool] = []
    target_extractor = model.critic_target.features_extractor
    original_rotation_angle = target_extractor._rotation_angle

    def record_target_mode(*args: Any, **kwargs: Any) -> torch.Tensor:
        target_modes.append(target_extractor.training)
        return original_rotation_angle(*args, **kwargs)

    target_extractor._rotation_angle = record_target_mode
    model.learn(
        total_timesteps=18,
        callback=RawStepControlCallback(raw_step_budget=18),
    )
    assert model._n_updates > 0
    assert target_modes and all(target_modes)
    assert not target_extractor.training
    checkpoint = tmp_path / "scene_rep_smoke"
    model.save(checkpoint)
    restored = SceneRepresentationSAC.load(checkpoint, env=env, device="cpu")
    restored_action, _ = restored.predict(observation, deterministic=True)
    assert restored_action.shape == (2,)
    assert np.isfinite(restored_action).all()


def test_mst_uses_source_sac_clock_without_constructing_slt(tmp_path: Any) -> None:
    env = DummyRawStepSceneEnv()
    model = make_model(
        "mst",
        env,
        scenario="left_turn",
        learning_starts=14,
        buffer_size=64,
        batch_size=2,
        action_repeat=3,
        device="cpu",
        verbose=0,
    )
    assert isinstance(model, SceneRepresentationSAC)
    assert model.representation_coef == 0.0
    assert model.representation is None
    assert model.representation_optimizer is None
    assert not model.replay_buffer.duplicate_episode_end_transition
    model.learn(
        total_timesteps=16,
        callback=RawStepControlCallback(raw_step_budget=16),
    )
    assert model._raw_steps_seen == 16
    assert model._n_updates == 3
    checkpoint = tmp_path / "mst_smoke"
    model.save(checkpoint)
    restored = SceneRepresentationSAC.load(checkpoint, env=env, device="cpu")
    assert restored.representation is None
    assert restored._raw_steps_seen == 16


def test_smarts_ppo_uses_source_cnn_and_asymmetric_log_probability() -> None:
    env = DummyRgbEnv()
    model = make_model(
        "ppo",
        env,
        scenario="left_turn",
        batch_size=32,
        action_repeat=1,
        device="cpu",
        verbose=0,
    )
    assert isinstance(model, SourcePPO)
    assert isinstance(model.policy, SourcePpoPolicy)
    assert model.n_steps == 512
    assert model.n_epochs == 10
    assert model.lr_schedule(1.0) == pytest.approx(5e-4)
    assert not model.normalize_advantage
    assert math.isinf(model.max_grad_norm)
    observation, _ = env.reset(seed=7)
    action, _ = model.predict(observation, deterministic=True)
    assert action.shape == (2,)
    observations = torch.as_tensor(observation[None])
    sampled_actions, _, collection_log_prob = model.policy(observations)
    _, update_log_prob, entropy = model.policy.evaluate_actions(
        observations, sampled_actions
    )
    assert torch.isfinite(collection_log_prob).all()
    assert torch.isfinite(update_log_prob).all()
    assert torch.isfinite(entropy).all()
    assert not torch.allclose(collection_log_prob, update_log_prob)


def test_source_ppo_actor_loss_stops_at_shared_encoder() -> None:
    model = make_model(
        "ppo",
        DummyRgbEnv(),
        scenario="left_turn",
        batch_size=32,
        action_repeat=1,
        device="cpu",
        verbose=0,
    )
    observations = torch.rand((2, 80, 80, 3), dtype=torch.float32)
    actions = torch.full((2, 2), 0.25, dtype=torch.float32)
    model.policy.zero_grad(set_to_none=True)
    _, log_prob, entropy = model.policy.evaluate_actions(observations, actions)
    actor_only_loss = -(log_prob + 0.05 * entropy).mean()
    actor_only_loss.backward()
    extractor_gradients = [
        parameter.grad for parameter in model.policy.features_extractor.parameters()
    ]
    actor_gradients = [
        parameter.grad for parameter in model.policy.action_net.parameters()
    ]
    assert all(
        gradient is None or torch.count_nonzero(gradient).item() == 0
        for gradient in extractor_gradients
    )
    assert any(
        gradient is not None and torch.count_nonzero(gradient).item() > 0
        for gradient in actor_gradients
    )

    model.policy.zero_grad(set_to_none=True)
    values = model.policy.predict_values(observations)
    values.square().mean().backward()
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.policy.features_extractor.parameters()
    )


def test_source_ppo_short_rollout_and_update() -> None:
    model = SourcePPO(
        SourcePpoPolicy,
        DummyRgbEnv(),
        learning_rate=5e-4,
        n_steps=8,
        batch_size=4,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=8)
    assert model.num_timesteps == 8
    assert model._n_updates == 2
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())


def test_carla_ppo_reuses_action_value_and_log_prob_for_three_raw_steps() -> None:
    env = RecordingCarlaPpoEnv()
    model = SourcePPO(
        SourcePpoPolicy,
        env,
        learning_rate=5e-4,
        n_steps=6,
        batch_size=3,
        n_epochs=1,
        policy_kwargs={"features_extractor_class": SourcePpoCarlaExtractor},
        source_action_hold=3,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=6)
    assert len(env.actions) == 6
    np.testing.assert_array_equal(env.actions[0], env.actions[1])
    np.testing.assert_array_equal(env.actions[1], env.actions[2])
    np.testing.assert_array_equal(env.actions[3], env.actions[4])
    np.testing.assert_array_equal(env.actions[4], env.actions[5])
    assert not np.array_equal(env.actions[2], env.actions[3])


def test_cross_and_roundabout_scenario_specific_model_flags() -> None:
    env = DummySceneEnv()
    cross = make_model(
        "scene_rep",
        env,
        scenario="cross",
        learning_starts=2,
        buffer_size=64,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    assert cross.critic.features_extractor.goal_heads == 2
    assert cross.representation.target_projector is not None
    assert not cross.representation_online_target_encoder
    assert cross.representation.temporal_attention.num_heads == 2
    roundabout = make_model(
        "scene_rep",
        env,
        scenario="roundabout",
        learning_starts=2,
        buffer_size=64,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    assert roundabout.critic.features_extractor.no_neighbor_future


def test_reconstructed_sac_lstm_uses_gru_without_transformer() -> None:
    env = DummySacSceneEnv()
    model = make_model(
        "sac",
        env,
        scenario="left_turn",
        learning_starts=14,
        buffer_size=64,
        batch_size=2,
        action_repeat=3,
        device="cpu",
        verbose=0,
    )
    assert isinstance(model, SceneRepresentationSAC)
    assert isinstance(model.critic.features_extractor, SourceSacLstmExtractor)
    assert not any(
        isinstance(module, HierarchicalSceneExtractor)
        for module in model.policy.modules()
    )
    assert model.representation is None
    assert model.raw_learning_starts == 14
    assert model.gradient_steps == 3
    observation, _ = env.reset(seed=9)
    action, _ = model.predict(observation, deterministic=True)
    assert action.shape == (2,)
    assert np.isfinite(action).all()


def test_sac_lstm_extractor_ignores_maps_and_masks_right_padding() -> None:
    extractor = SourceSacLstmExtractor(_sac_space())
    observations = _sac_batch(batch_size=2)
    first = extractor(observations)
    changed_map = {
        "state_lstm": observations["state_lstm"].clone(),
        "state_lstm_mask": observations["state_lstm_mask"].clone(),
    }
    # Timestamps 4..9 are invalid under the explicit source runner mask;
    # arbitrary padded STATE_LSTM values must not alter the final GRU state.
    changed_map["state_lstm"][:, 4:] = torch.randn_like(
        changed_map["state_lstm"][:, 4:]
    )
    second = extractor(changed_map)
    assert first.shape == (2, 256)
    assert torch.allclose(first, second)
    assert torch.isfinite(first).all()


def test_reconstructed_sac_uses_source_raw_step_clock() -> None:
    env = DummyRawStepSacEnv()
    model = make_model(
        "sac",
        env,
        scenario="left_turn",
        learning_starts=14,
        buffer_size=64,
        batch_size=2,
        action_repeat=3,
        device="cpu",
        verbose=0,
    )
    model.learn(
        total_timesteps=16,
        callback=RawStepControlCallback(raw_step_budget=16),
    )
    assert model._raw_steps_seen == 16
    assert model._n_updates == 3
