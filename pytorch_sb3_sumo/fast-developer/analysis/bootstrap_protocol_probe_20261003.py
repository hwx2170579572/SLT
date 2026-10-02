"""Offline probe and source audit for the active Dict n-step SAC protocol.

This script creates only CPU replay buffers containing hand-written transitions.
It never creates an environment, policy, optimizer, SUMO process, or training run.
It refuses to overwrite the dated audit JSON.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3.common.buffers import DictReplayBuffer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algos.sb3_torch.replay_buffer import DictNStepReplayBuffer  # noqa: E402

GAMMA = 0.99
N_STEPS = 4
ACTION_REPEAT = 3
SYNTHETIC_REWARDS = (1.0, 2.0, 3.0, 4.0)


def _spaces() -> tuple[gym.spaces.Dict, gym.spaces.Box]:
    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(
                low=-1000.0, high=1000.0, shape=(1,), dtype=np.float32
            )
        }
    )
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(1,), dtype=np.float32
    )
    return observation_space, action_space


def _add_transition(
    replay_buffer: Any,
    index: int,
    reward: float,
    *,
    terminal: bool = False,
    timeout: bool = False,
    raw_steps_executed: int = ACTION_REPEAT,
) -> None:
    info: dict[str, Any] = {"raw_steps_executed": raw_steps_executed}
    if timeout:
        # SB3 stores TimeLimit.truncated separately from done and masks it
        # back out of the terminal flag when samples are formed.
        info["TimeLimit.truncated"] = True
    replay_buffer.add(
        {"state": np.asarray([[index]], dtype=np.float32)},
        {"state": np.asarray([[index + 1]], dtype=np.float32)},
        np.asarray([[0.0]], dtype=np.float32),
        np.asarray([reward], dtype=np.float32),
        np.asarray([terminal], dtype=bool),
        [info],
    )


def _probe_nstep_case(
    rewards: tuple[float, ...],
    *,
    boundary: str | None,
    append_next_episode: bool = False,
) -> dict[str, Any]:
    if not 1 <= len(rewards) <= N_STEPS:
        raise ValueError("Probe sequence length must be in [1, n_steps]")
    if boundary not in {None, "terminal", "timeout"}:
        raise ValueError(f"Unsupported boundary: {boundary}")
    if boundary is None and len(rewards) != N_STEPS:
        raise ValueError("An unbounded probe must contain a full n-step window")

    observation_space, action_space = _spaces()
    replay_buffer = DictNStepReplayBuffer(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_steps=N_STEPS,
        gamma=GAMMA,
        source_action_repeat=ACTION_REPEAT,
    )
    for index, reward in enumerate(rewards):
        is_boundary = boundary is not None and index == len(rewards) - 1
        _add_transition(
            replay_buffer,
            index,
            reward,
            terminal=is_boundary,
            timeout=is_boundary and boundary == "timeout",
            # Exercise a partial held-action transition at the boundary.
            raw_steps_executed=1 if is_boundary else ACTION_REPEAT,
        )

    if append_next_episode and boundary is not None:
        _add_transition(
            replay_buffer,
            len(rewards),
            999.0,
            raw_steps_executed=ACTION_REPEAT,
        )

    sample = replay_buffer._get_samples(np.asarray([0], dtype=np.int64))
    boundary_slot = len(rewards) - 1
    stored_timeout = float(replay_buffer.timeouts[boundary_slot, 0])
    return {
        "n_decision_transitions": len(rewards),
        "boundary": boundary,
        "raw_steps_executed_on_final_transition": (
            1 if boundary is not None else ACTION_REPEAT
        ),
        "rewards": float(sample.rewards.item()),
        "next_observation_state": float(sample.next_observations["state"].item()),
        "bootstrap_discount": float(sample.discounts.item()),
        "done_mask": float(sample.dones.item()),
        "stored_timeout_flag": stored_timeout,
        "buffer_sampleable_size": int(replay_buffer.size()),
    }


def _probe_one_step_control() -> dict[str, Any]:
    observation_space, action_space = _spaces()
    replay_buffer = DictReplayBuffer(
        16, observation_space, action_space, device="cpu"
    )
    _add_transition(replay_buffer, 0, 1.0)
    sample = replay_buffer._get_samples(np.asarray([0], dtype=np.int64))
    discounts = getattr(sample, "discounts", None)
    return {
        "n_decision_transitions": 1,
        "reward": float(sample.rewards.item()),
        "done_mask": float(sample.dones.item()),
        "sample_has_discount_field": hasattr(sample, "discounts"),
        "sample_discount_is_none": discounts is None,
        "sac_fallback_discount": GAMMA,
    }


def _line_of(path: Path, needle: str) -> int | None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return line_number
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_arithmetic(reward_return: float, done_mask: float, discount: float) -> float:
    # Fixed artificial target values make the SAC target arithmetic
    # independently reproducible without loading a model.
    q1, q2 = 5.0, 3.0
    alpha, next_log_probability = 0.2, -0.5
    soft_next_q = min(q1, q2) - alpha * next_log_probability
    return reward_return + (1.0 - done_mask) * discount * soft_next_q


def build_audit_report() -> dict[str, Any]:
    replay_path = REPO_ROOT / "algos" / "sb3_torch" / "replay_buffer.py"
    sac_path = REPO_ROOT / "algos" / "sb3_torch" / "sac.py"
    base_train_path = REPO_ROOT / "fast-developer" / "train_intersection_yield_v2.py"
    d1_train_path = REPO_ROOT / "fast-developer" / "train_intersection_yield_v2_d1.py"
    paper_registry_path = REPO_ROOT / "envs" / "sumo" / "paper_scenario_registry.py"
    env_path = REPO_ROOT / "envs" / "sumo" / "sumo_env.py"
    paper_env_path = REPO_ROOT / "envs" / "sumo" / "paper_env.py"
    sorted_sumocfg_path = REPO_ROOT / "envs" / "sumo" / "scenarios" / "intersection_sorted" / "scenario.sumocfg"
    reward_path = REPO_ROOT / "fast-developer" / "reward_shaping_v2.py"

    replay_source = replay_path.read_text(encoding="utf-8")
    sac_source = sac_path.read_text(encoding="utf-8")
    observation_scan_roots = [
        REPO_ROOT / "envs" / "sumo",
        REPO_ROOT / "algos" / "sb3_torch",
        d1_train_path,
    ]
    time_field_tokens = (
        "remaining_time",
        "time_remaining",
        "time_left",
        "remaining_steps",
        "steps_remaining",
        "time_to_go",
    )
    scanned_files: list[Path] = []
    for root in observation_scan_roots:
        scanned_files.extend([root] if root.is_file() else sorted(root.rglob("*.py")))
    remaining_time_matches = [
        {"path": str(path.relative_to(REPO_ROOT)), "token": token}
        for path in scanned_files
        for token in time_field_tokens
        if token in path.read_text(encoding="utf-8", errors="replace")
    ]

    full_nonterminal = _probe_nstep_case(
        SYNTHETIC_REWARDS, boundary=None
    )
    full_timeout = _probe_nstep_case(
        SYNTHETIC_REWARDS, boundary="timeout", append_next_episode=True
    )
    full_terminal = _probe_nstep_case(
        SYNTHETIC_REWARDS, boundary="terminal", append_next_episode=True
    )
    shortened: list[dict[str, Any]] = []
    for k in (1, 2, 3):
        rewards = tuple(float(i) for i in range(1, k + 1))
        for boundary in ("timeout", "terminal"):
            row = _probe_nstep_case(
                rewards, boundary=boundary, append_next_episode=True
            )
            row["standard_k_step_discount"] = GAMMA**k
            row["current_bootstrap_discount"] = GAMMA
            shortened.append(row)

    standard_full_discount = GAMMA**N_STEPS
    synthetic_full_reward = sum(
        (GAMMA**index) * reward
        for index, reward in enumerate(SYNTHETIC_REWARDS)
    )
    current_timeout_target = _target_arithmetic(
        full_timeout["rewards"], full_timeout["done_mask"], full_timeout["bootstrap_discount"]
    )
    standard_timeout_target = _target_arithmetic(
        full_timeout["rewards"], full_timeout["done_mask"], standard_full_discount
    )
    terminal_target = _target_arithmetic(
        full_terminal["rewards"], full_terminal["done_mask"], GAMMA
    )
    one_step = _probe_one_step_control()

    required_source_fragments = {
        "buffer_one_gamma_comment": "one factor of self.discount for bootstrap rather than gamma**N",
        "buffer_discount_fill": "bootstrap_discounts = np.full(",
        "timeout_terminal_mask": "final_dones = next_dones * (1.0 - next_timeouts)",
        "sac_discount_fallback": "discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma",
        "sac_endpoint_entropy": "next_q_values = next_q_values - entropy_coefficient * next_log_prob.reshape(-1, 1)",
        "sac_target": "target_q_values = replay_data.rewards + (",
    }
    source_contract_checks = {
        name: (fragment in (replay_source if name.startswith("buffer") or name == "timeout_terminal_mask" else sac_source))
        for name, fragment in required_source_fragments.items()
    }
    if not all(source_contract_checks.values()):
        missing = [name for name, passed in source_contract_checks.items() if not passed]
        raise AssertionError(f"Source contract changed; missing expected fragments: {missing}")

    return {
        "schema_version": 1,
        "audit_id": "bootstrap_protocol_audit_20261003",
        "probe_scope": "CPU replay buffers with synthetic transitions only; no env/model/optimizer/training/evaluation/SUMO/GPU",
        "runtime": {
            "python_package": "stable_baselines3",
            "stable_baselines3_version": stable_baselines3.__version__,
            "gamma": GAMMA,
            "n_steps": N_STEPS,
            "action_repeat_raw_ticks_per_decision": ACTION_REPEAT,
        },
        "current_protocol": {
            "decision_transition_reward": "R_k = sum_{i=0}^{k-1} gamma^i * r_i, where k is the actual decision-transition count up to n_steps or first boundary",
            "bootstrap_target": "Y = R_k + (1 - done_mask) * gamma * (min(Q1_target,Q2_target) - alpha * log_pi(a_next|s_next))",
            "discount_for_bootstrap": "always gamma, not gamma**k",
            "true_terminal": "done_mask=1; no bootstrap",
            "timeout_truncation": "boundary ends reward accumulation; timeout flag is removed from final done mask, so bootstrap continues from that boundary's next_observation",
            "short_tail": "actual k determines accumulated reward and next_observation, but bootstrap multiplier remains gamma",
            "one_step_control": "standard DictReplayBuffer sample provides no transition-specific discount value (discounts=None); SAC falls back to gamma",
            "entropy": "one endpoint soft-value adjustment -alpha*log_pi is applied to target next Q; no intermediate entropy reward is inserted into n-step replay return",
            "off_policy_correction": "no intermediate importance/trace correction is applied to the replayed n-step sequence",
            "released_intent_evidence": "local replay_buffer.py comment explicitly says this one-gamma bootstrap preserves the released SAC implementation; this confirms project intent, not independent provenance of the upstream source",
        },
        "standard_comparison": {
            "standard_k_step_bootstrap": "gamma**k for a k-transition return, including shortened tails; terminal mask still removes bootstrap",
            "full_4_step_gamma_power_4": standard_full_discount,
            "current_full_4_step_bootstrap": GAMMA,
            "one_step_equal": GAMMA**1 == GAMMA,
        },
        "synthetic_probe": {
            "rewards_for_full_cases": list(SYNTHETIC_REWARDS),
            "full_nonterminal_4step": full_nonterminal,
            "timeout_on_fourth": full_timeout,
            "true_terminal_on_fourth": full_terminal,
            "shortened_k_1_to_3_terminal_and_timeout": shortened,
            "episode_boundary_no_crossing_check": {
                "post_boundary_reward": 999.0,
                "full_timeout_return_stayed_at_expected_four_step_sum": abs(full_timeout["rewards"] - synthetic_full_reward) < 1e-6,
                "full_terminal_return_stayed_at_expected_four_step_sum": abs(full_terminal["rewards"] - synthetic_full_reward) < 1e-6,
                "short_tail_returns_exclude_post_boundary_episode": all(
                    row["rewards"] < 999.0 for row in shortened
                ),
            },
            "one_step_control": one_step,
            "fixed_sac_target_arithmetic": {
                "target_critic_values": [5.0, 3.0],
                "alpha": 0.2,
                "next_log_probability": -0.5,
                "soft_next_q": 3.1,
                "timeout_current_protocol_target": current_timeout_target,
                "timeout_standard_gamma_to_four_target": standard_timeout_target,
                "timeout_target_difference_current_minus_standard": current_timeout_target - standard_timeout_target,
                "terminal_target_no_bootstrap": terminal_target,
            },
        },
        "intersection_sorted_depart4p0_task_horizon": {
            "paper_scenario_config": "intersection_sorted has max_episode_steps=600 raw SUMO ticks",
            "raw_tick_seconds": 0.1,
            "nominal_cap_seconds": 60.0,
            "action_repeat": ACTION_REPEAT,
            "nominal_decisions_to_cap": 600 // ACTION_REPEAT,
            "nominal_4_transition_window_seconds": N_STEPS * ACTION_REPEAT * 0.1,
            "cap_handling": "source_observation_contract='smarts' is not CARLA, so max_time is Gymnasium truncated (not terminated); the n-step buffer closes the reward sequence at that boundary and preserves bootstrap",
            "training_timeout_shaping": "timeout reward is -5.0 in the shared v2 reward wrapper; this penalty is accumulated in R_k while the cap transition remains bootstrap-eligible",
            "remaining_time_observation": "No remaining-time token was found in the static scan of envs/sumo, algos/sb3_torch, and the D1 trainer; base observation builder exposes trajectory/map and optional state_lstm/state_lstm_mask. This scan does not classify the intended task semantics.",
            "interpretation_boundary": "If completing within 60 s is intrinsic to the task, finite-horizon theory treats the deadline as termination and the remaining time belongs in the state for Markov semantics. If the 60 s cap is only external collection truncation, bootstrapping is appropriate. The code's timeout penalty does not by itself decide which task definition was intended.",
            "remaining_time_token_matches": remaining_time_matches,
        },
        "source_locations": {
            "replay_buffer.py": {
                "path": str(replay_path.relative_to(REPO_ROOT)),
                "lines": {
                    "boundary_tail_flush": _line_of(replay_path, "def _add_one("),
                    "incomplete_partial_action_handling": _line_of(replay_path, "def add("),
                    "sample_and_return_logic": _line_of(replay_path, "def _get_samples("),
                    "release_comment": _line_of(replay_path, "# Preserve the released SAC implementation exactly"),
                    "one_gamma_bootstrap": _line_of(replay_path, "bootstrap_discounts = np.full("),
                    "discounted_n_step_reward": _line_of(replay_path, "reward_discounts = self.gamma ** np.arange("),
                    "timeout_mask": _line_of(replay_path, "final_dones = next_dones * (1.0 - next_timeouts)"),
                },
            },
            "sac.py": {
                "path": str(sac_path.relative_to(REPO_ROOT)),
                "lines": {
                    "discount_fallback": _line_of(sac_path, "discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma"),
                    "endpoint_entropy_adjustment": _line_of(sac_path, "next_q_values = next_q_values - entropy_coefficient * next_log_prob.reshape(-1, 1)"),
                    "sac_target": _line_of(sac_path, "target_q_values = replay_data.rewards + ("),
                },
            },
            "train_config": {
                "base_path": str(base_train_path.relative_to(REPO_ROOT)),
                "base_discount_line": _line_of(base_train_path, "DISCOUNT = 0.99"),
                "base_action_repeat_line": _line_of(base_train_path, "ACTION_REPEAT = 3"),
                "d1_path": str(d1_train_path.relative_to(REPO_ROOT)),
                "d1_n_steps_line": _line_of(d1_train_path, "n_steps=4,"),
            },
            "scenario_and_env": {
                "registry_path": str(paper_registry_path.relative_to(REPO_ROOT)),
                "sorted_scenario_line": _line_of(paper_registry_path, '"intersection_sorted": PaperScenarioSpec('),
                "sumo_env_path": str(env_path.relative_to(REPO_ROOT)),
                "max_episode_steps_property_line": _line_of(env_path, "def max_episode_steps("),
                "timeout_terminal_split_line": _line_of(env_path, "max_time_is_terminal = source_contract == \"carla\""),
                "observation_builder_line": _line_of(env_path, "def _make_observation("),
                "raw_tick_length": "0.1 seconds",
                "raw_tick_length_sumo_command_path": str(paper_env_path.relative_to(REPO_ROOT)),
                "raw_tick_length_sumo_command_line": _line_of(paper_env_path, '"--step-length"'),
                "sorted_sumocfg_path": str(sorted_sumocfg_path.relative_to(REPO_ROOT)),
                "sorted_sumocfg_step_length_line": _line_of(sorted_sumocfg_path, '<step-length value="0.1"/>'),
                "reward_shaping_path": str(reward_path.relative_to(REPO_ROOT)),
                "timeout_reward_line": _line_of(reward_path, "timeout_reward: float = -5.0"),
            },
        },
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                replay_path,
                sac_path,
                base_train_path,
                d1_train_path,
                paper_registry_path,
                env_path,
                paper_env_path,
                sorted_sumocfg_path,
                reward_path,
            )
        },
        "source_contract_checks": source_contract_checks,
        "external_theory_sources": [
            {
                "title": "Gymnasium: Handling Time Limits",
                "url": "https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/",
                "relevant_claim": "intrinsic finite-horizon deadline is termination and remaining time is needed in observation; externally imposed truncation bootstraps",
            },
            {
                "title": "Pardo et al., Time Limits in Reinforcement Learning (ICML 2018)",
                "url": "https://arxiv.org/abs/1712.00378",
                "relevant_claim": "distinguishes finite-horizon task deadlines from external time limits and their state/bootstrapping treatment",
            },
            {
                "title": "Stable-Baselines3 SAC source documentation",
                "url": "https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/sac/sac.html",
                "relevant_claim": "n-step replay target convention uses gamma**n_steps for a full n-step horizon; compare versioned local implementation before attributing behavior",
            },
        ],
    }


def write_audit_json() -> Path:
    output = Path(__file__).resolve().with_name("bootstrap_protocol_audit_20261003.json")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output}")
    report = build_audit_report()
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(write_audit_json())
