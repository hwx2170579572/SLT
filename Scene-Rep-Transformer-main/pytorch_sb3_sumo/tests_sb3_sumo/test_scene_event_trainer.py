from __future__ import annotations

from pathlib import Path
import sys

import pytest


FAST_DEVELOPER = Path(__file__).resolve().parents[1] / "fast-developer"
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

from scene_event.trainer import EpisodeAccount, REWARD_COMPONENTS  # noqa: E402


def _info(*, cumulative: dict[str, float], raw_reward: float) -> dict[str, object]:
    return {
        "raw_steps_executed": 3,
        "undiscounted_reward": raw_reward,
        **cumulative,
    }


def test_episode_account_differences_cumulative_components_and_keeps_raw_return_separate() -> None:
    account = EpisodeAccount("train", episode=2, seed=42)
    first = {name: 0.0 for name in REWARD_COMPONENTS}
    first.update(reward_success=0.0, reward_collision=0.0, reward_off_route=0.0,
                 reward_timeout=0.0, reward_step_cost=-0.3, reward_progress=1.2)
    first_reward = 0.9
    account.add(first_reward, _info(cumulative=first, raw_reward=-0.3))

    # Values in info are episode-cumulative: the second per-decision reward is
    # the increase in those components, not their second-step cumulative sum.
    second = dict(first)
    second["reward_step_cost"] = -0.6
    second["reward_progress"] = 1.7
    second_reward = 0.2
    account.add(second_reward, _info(cumulative=second, raw_reward=-0.3))

    result = account.result(
        terminated=True,
        truncated=False,
        info={"is_success": True, "collision": False, "off_route": False, "max_time": False},
        completed=True,
    )
    assert result["raw_steps"] == 6
    assert result["decision_steps"] == 2
    assert result["environment_step_reward_v2_return"] == pytest.approx(1.1)
    assert result["raw_environment_return"] == pytest.approx(-0.6)
    assert result["reward_progress"] == pytest.approx(1.7)
    assert result["reward_step_cost"] == pytest.approx(-0.6)
    assert result["reward_component_error"] == pytest.approx(0.0)
    assert result["max_component_step_error"] == pytest.approx(0.0)


def test_episode_account_rejects_bad_reward_component_delta_and_zero_raw_steps() -> None:
    account = EpisodeAccount("train", episode=0, seed=0)
    cumulative = {name: 0.0 for name in REWARD_COMPONENTS}
    cumulative["reward_progress"] = 1.0
    with pytest.raises(ValueError, match="Six-component reward reconciliation failed"):
        account.add(0.0, _info(cumulative=cumulative, raw_reward=0.0))

    cumulative = {name: 0.0 for name in REWARD_COMPONENTS}
    with pytest.raises(ValueError, match="positive"):
        account.add(0.0, {"raw_steps_executed": 0, **cumulative})


def test_completed_episode_requires_one_mutually_exclusive_outcome() -> None:
    account = EpisodeAccount("eval", episode=0, seed=10_000)
    with pytest.raises(ValueError, match="exactly one mutually exclusive"):
        account.result(
            terminated=True,
            truncated=False,
            info={"is_success": True, "collision": True},
            completed=True,
        )

