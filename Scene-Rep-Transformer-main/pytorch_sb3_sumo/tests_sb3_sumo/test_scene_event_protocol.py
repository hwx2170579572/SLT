from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn


FAST_DEVELOPER = Path(__file__).resolve().parents[1] / "fast-developer"
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

from scene_event.protocol import (  # noqa: E402
    ExperimentConfig,
    discounted_return,
    readonly_observation,
    soft_bellman_target,
)
from scene_event.replay import (  # noqa: E402
    DecisionTransition,
    NStepAssembler,
    ObservationReplay,
)


def _obs(state: float, remaining: float) -> dict[str, np.ndarray]:
    return {
        "state": np.asarray([state], dtype=np.float32),
        "remaining_time_s": np.asarray([remaining], dtype=np.float32),
    }


def _decision(
    *,
    episode: int,
    decision: int,
    reward: float,
    state: float,
    next_state: float,
    remaining: float = 10.0,
    next_remaining: float = 10.0,
    raw_steps: int = 3,
    terminated: bool = False,
    truncated: bool = False,
) -> DecisionTransition:
    return DecisionTransition(
        observation=_obs(state, remaining),
        action=np.asarray([0.1, -0.2], dtype=np.float32),
        reward=reward,
        next_observation=_obs(next_state, next_remaining),
        terminated=terminated,
        truncated=truncated,
        raw_steps=raw_steps,
        episode_id=episode,
        decision_id=decision,
    )


def test_discounted_return_uses_actual_decision_horizon() -> None:
    total, discount = discounted_return([1.0, 2.0, 3.0, 4.0], gamma=0.9)
    expected = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0 + 0.9**3 * 4.0
    assert total == pytest.approx(8.146)
    assert total == pytest.approx(expected)
    assert discount == pytest.approx(0.9**4)


@pytest.mark.parametrize("k", (1, 2, 3))
def test_boundary_prefix_uses_k_decisions_not_raw_ticks(k: int) -> None:
    gamma = 0.9
    assembler = NStepAssembler(n_step=4, gamma=gamma)
    emitted = []
    for index in range(k):
        emitted.extend(
            assembler.append(
                _decision(
                    episode=4,
                    decision=index,
                    reward=float(index + 1),
                    state=float(index),
                    next_state=float(index + 1),
                    raw_steps=1 if index == k - 1 else 3,
                    terminated=index == k - 1,
                )
            )
        )
    first = emitted[0]
    expected_return = sum(gamma**j * float(j + 1) for j in range(k))
    assert first.decisions == k
    assert first.reward == pytest.approx(expected_return)
    assert first.discount == pytest.approx(gamma**k)
    assert first.raw_steps == 3 * (k - 1) + 1
    assert first.terminated is True
    assert first.truncated is False


def test_confirmed_deadline_is_terminal_and_remaining_time_survives_replay() -> None:
    config = ExperimentConfig(device="cpu")
    config.validate()
    assert config.deadline_seconds == 60.0
    assert config.deadline_is_terminal is True
    with pytest.raises(ValueError, match="60 s finite task deadline"):
        replace(config, deadline_seconds=59.9).validate()
    with pytest.raises(ValueError, match="60 s finite task deadline"):
        replace(config, deadline_is_terminal=False).validate()

    assembler = NStepAssembler(n_step=4, gamma=0.9)
    for index, reward in enumerate((1.0, 2.0, 3.0)):
        assert assembler.append(
            _decision(
                episode=9,
                decision=index,
                reward=reward,
                state=float(index),
                next_state=float(index + 1),
                remaining=0.3 * (3 - index),
                next_remaining=0.3 * (2 - index),
            )
        ) == []
    deadline_rows = assembler.append(
        _decision(
            episode=9,
            decision=3,
            reward=4.0,
            state=3.0,
            next_state=4.0,
            remaining=0.1,
            next_remaining=0.0,
            raw_steps=1,
            terminated=True,
        )
    )
    first = deadline_rows[0]
    assert first.decisions == 4
    assert first.raw_steps == 10
    assert first.reward == pytest.approx(8.146)
    assert first.discount == pytest.approx(0.9**4)
    assert first.terminated is True and first.truncated is False

    replay = ObservationReplay(capacity=8, seed=0)
    replay.add(first)
    batch = replay.sample(1)
    assert batch["next_observation"]["remaining_time_s"][0, 0] == pytest.approx(0.0)
    target = soft_bellman_target(
        reward=torch.tensor([[first.reward]], dtype=torch.float32),
        discount=torch.tensor([[first.discount]], dtype=torch.float32),
        terminated=torch.tensor([[first.terminated]], dtype=torch.bool),
        target_q=torch.tensor([[1000.0]], dtype=torch.float32),
        next_log_prob=torch.tensor([[-50.0]], dtype=torch.float32),
        alpha=torch.tensor(0.2),
    )
    assert target.item() == pytest.approx(first.reward)


def test_external_truncation_bootstraps_from_last_physical_observation() -> None:
    assembler = NStepAssembler(n_step=4, gamma=0.9)
    assert assembler.append(
        _decision(
            episode=12,
            decision=0,
            reward=1.0,
            state=10.0,
            next_state=11.0,
            remaining=7.0,
            next_remaining=6.7,
        )
    ) == []
    emitted = assembler.append(
        _decision(
            episode=12,
            decision=1,
            reward=2.0,
            state=11.0,
            next_state=12.0,
            remaining=6.7,
            next_remaining=6.6,
            raw_steps=1,
            truncated=True,
        )
    )
    first = emitted[0]
    assert first.decisions == 2
    assert first.raw_steps == 4
    assert first.reward == pytest.approx(1.0 + 0.9 * 2.0)
    assert first.discount == pytest.approx(0.9**2)
    assert first.terminated is False and first.truncated is True
    assert first.next_observation["state"][0] == pytest.approx(12.0)
    assert first.next_observation["remaining_time_s"][0] == pytest.approx(6.6)

    target = soft_bellman_target(
        reward=torch.tensor([[first.reward]], dtype=torch.float32),
        discount=torch.tensor([[first.discount]], dtype=torch.float32),
        terminated=torch.tensor([[first.terminated]], dtype=torch.bool),
        target_q=torch.tensor([[4.0]], dtype=torch.float32),
        next_log_prob=torch.tensor([[-1.0]], dtype=torch.float32),
        alpha=torch.tensor(0.2),
    )
    assert target.item() == pytest.approx(first.reward + (0.9**2) * 4.2)


def test_collection_flush_marks_truncation_and_keeps_real_final_state() -> None:
    assembler = NStepAssembler(n_step=4, gamma=0.9)
    assert assembler.append(
        _decision(episode=15, decision=0, reward=2.0, state=1.0, next_state=2.0)
    ) == []
    assert assembler.append(
        _decision(episode=15, decision=1, reward=3.0, state=2.0, next_state=3.0)
    ) == []
    flushed = assembler.flush_external_truncation()
    assert len(flushed) == 2
    assert flushed[0].truncated is True and flushed[0].terminated is False
    assert flushed[0].next_observation["state"][0] == pytest.approx(3.0)
    assert flushed[0].discount == pytest.approx(0.9**2)
    assert flushed[1].reward == pytest.approx(3.0)
    assert flushed[1].discount == pytest.approx(0.9)


def test_collection_cutoff_omits_nonterminal_partial_macro_action() -> None:
    gamma = 0.9
    assembler = NStepAssembler(n_step=3, gamma=gamma)
    emitted = []
    for decision in range(3):
        ready, omitted = assembler.append_collected(
            _decision(
                episode=16,
                decision=decision,
                reward=1.0,
                state=float(decision * 3),
                next_state=float((decision + 1) * 3),
                raw_steps=3,
            ),
            action_repeat=3,
            collection_cutoff=False,
        )
        assert omitted is False
        emitted.extend(ready)

    first = emitted[0]
    assert first.decisions == 3
    assert first.raw_steps == 9
    assert first.reward == pytest.approx(1.0 + gamma + gamma**2)
    assert first.discount == pytest.approx(gamma**3)
    assert first.next_observation["state"][0] == pytest.approx(9.0)

    ready, omitted = assembler.append_collected(
        _decision(
            episode=16,
            decision=3,
            reward=100.0,
            state=9.0,
            next_state=10.0,
            raw_steps=1,
            truncated=True,
        ),
        action_repeat=3,
        collection_cutoff=True,
    )
    assert omitted is True
    assert len(ready) == 2
    assert [row.decisions for row in ready] == [2, 1]
    assert [row.raw_steps for row in ready] == [6, 3]
    assert [row.discount for row in ready] == pytest.approx([gamma**2, gamma])
    assert all(row.terminated is False and row.truncated is True for row in ready)
    assert all(row.next_observation["state"][0] == pytest.approx(9.0) for row in ready)
    assert all(row.next_observation["state"][0] != pytest.approx(10.0) for row in ready)


def test_collection_cutoff_retains_short_true_terminal_macro_action() -> None:
    gamma = 0.9
    assembler = NStepAssembler(n_step=3, gamma=gamma)
    for decision in range(2):
        ready, omitted = assembler.append_collected(
            _decision(
                episode=17,
                decision=decision,
                reward=1.0,
                state=float(decision * 3),
                next_state=float((decision + 1) * 3),
                raw_steps=3,
            ),
            action_repeat=3,
            collection_cutoff=False,
        )
        assert ready == [] and omitted is False

    ready, omitted = assembler.append_collected(
        _decision(
            episode=17,
            decision=2,
            reward=2.0,
            state=6.0,
            next_state=7.0,
            raw_steps=1,
            terminated=True,
        ),
        action_repeat=3,
        collection_cutoff=True,
    )
    assert omitted is False
    assert len(ready) == 3
    first = ready[0]
    assert first.decisions == 3
    assert first.raw_steps == 7
    assert first.reward == pytest.approx(1.0 + gamma + gamma**2 * 2.0)
    assert first.discount == pytest.approx(gamma**3)
    assert first.terminated is True and first.truncated is False
    assert first.next_observation["state"][0] == pytest.approx(7.0)
    assert all(row.terminated is True and row.truncated is False for row in ready)


@pytest.mark.parametrize(
    "bad_transition",
    (
        _decision(episode=21, decision=0, reward=1.0, state=0.0, next_state=1.0),
        _decision(episode=20, decision=2, reward=1.0, state=0.0, next_state=1.0),
    ),
)
def test_pending_return_rejects_episode_change_or_decision_gap(
    bad_transition: DecisionTransition,
) -> None:
    assembler = NStepAssembler(n_step=4, gamma=0.9)
    assembler.append(
        _decision(episode=20, decision=0, reward=1.0, state=0.0, next_state=1.0)
    )
    with pytest.raises(ValueError, match="Cannot join n-step returns"):
        assembler.append(bad_transition)


def test_terminal_boundary_flushes_before_accepting_next_episode() -> None:
    assembler = NStepAssembler(n_step=4, gamma=0.9)
    terminal = assembler.append(
        _decision(
            episode=31,
            decision=0,
            reward=100.0,
            state=0.0,
            next_state=1.0,
            terminated=True,
        )
    )
    assert len(terminal) == 1
    assert terminal[0].reward == pytest.approx(100.0)
    assert terminal[0].episode_id == 31

    assembler.append(
        _decision(episode=32, decision=0, reward=3.0, state=10.0, next_state=11.0)
    )
    next_episode = assembler.append(
        _decision(
            episode=32,
            decision=1,
            reward=5.0,
            state=11.0,
            next_state=12.0,
            truncated=True,
        )
    )
    assert next_episode[0].episode_id == 32
    assert next_episode[0].reward == pytest.approx(3.0 + 0.9 * 5.0)
    assert next_episode[0].decisions == 2


def test_readonly_observation_and_replay_preserve_raw_input_for_reencoding() -> None:
    collector_state = np.asarray([2.0], dtype=np.float32)
    owned = readonly_observation(
        {
            "state": collector_state,
            "remaining_time_s": np.asarray([4.0], dtype=np.float32),
        }
    )
    assert owned["state"].flags.writeable is False
    collector_state[0] = 99.0
    assert owned["state"][0] == pytest.approx(2.0)

    transition = DecisionTransition(
        observation=owned,
        action=np.asarray([0.0, 0.0], dtype=np.float32),
        reward=1.0,
        next_observation=readonly_observation(
            {
                "state": np.asarray([3.0], dtype=np.float32),
                "remaining_time_s": np.asarray([3.9], dtype=np.float32),
            }
        ),
        terminated=False,
        truncated=False,
        raw_steps=3,
        episode_id=40,
        decision_id=0,
    )
    assembled = NStepAssembler(n_step=1, gamma=0.9).append(transition)[0]
    replay = ObservationReplay(capacity=4, seed=0)
    replay.add(assembled)
    first = replay.sample(1)
    first["observation"]["state"][0, 0] = -123.0
    second = replay.sample(1)
    stored_state = second["observation"]["state"][0, 0]
    assert stored_state == pytest.approx(2.0)
    assert replay.transitions[0].observation["state"].flags.writeable is False
    assert "z" not in replay.transitions[0].observation

    encoder = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        encoder.weight.fill_(2.0)
    batch_state = torch.as_tensor(second["observation"]["state"])
    z_before = encoder(batch_state).detach().clone()
    with torch.no_grad():
        encoder.weight.fill_(3.0)
    z_after = encoder(batch_state).detach().clone()
    assert not torch.equal(z_before, z_after)
    assert replay.transitions[0].observation["state"][0] == pytest.approx(2.0)


def test_readonly_view_of_mutable_base_is_snapshotted() -> None:
    base = np.asarray([2.0], dtype=np.float32)
    view = base.view()
    view.setflags(write=False)
    assert view.flags.c_contiguous and not view.flags.writeable

    owned = readonly_observation({"state": view})
    base[0] = 99.0

    assert owned["state"][0] == pytest.approx(2.0)
    assert owned["state"].flags.writeable is False
