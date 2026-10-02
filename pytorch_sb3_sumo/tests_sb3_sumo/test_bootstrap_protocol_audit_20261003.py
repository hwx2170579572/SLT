from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "fast-developer"
    / "analysis"
    / "bootstrap_protocol_probe_20261003.py"
)
SPEC = importlib.util.spec_from_file_location(
    "bootstrap_protocol_probe_20261003", PROBE_PATH
)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


@pytest.fixture(scope="module")
def audit() -> dict:
    return PROBE.build_audit_report()


def test_full_window_and_fourth_step_boundaries(audit: dict) -> None:
    cases = audit["synthetic_probe"]
    expected_return = sum(
        (0.99**i) * reward for i, reward in enumerate((1.0, 2.0, 3.0, 4.0))
    )
    for name in ("full_nonterminal_4step", "timeout_on_fourth", "true_terminal_on_fourth"):
        row = cases[name]
        assert row["n_decision_transitions"] == 4
        assert row["rewards"] == pytest.approx(expected_return)
        assert row["next_observation_state"] == pytest.approx(4.0)
        assert row["bootstrap_discount"] == pytest.approx(0.99)
    assert cases["full_nonterminal_4step"]["done_mask"] == pytest.approx(0.0)
    assert cases["timeout_on_fourth"]["done_mask"] == pytest.approx(0.0)
    assert cases["timeout_on_fourth"]["stored_timeout_flag"] == pytest.approx(1.0)
    assert cases["true_terminal_on_fourth"]["done_mask"] == pytest.approx(1.0)


@pytest.mark.parametrize("k", (1, 2, 3))
@pytest.mark.parametrize("boundary,expected_done", (("timeout", 0.0), ("terminal", 1.0)))
def test_shortened_tail_uses_actual_k_for_reward_but_one_gamma_for_bootstrap(
    audit: dict, k: int, boundary: str, expected_done: float
) -> None:
    rows = audit["synthetic_probe"]["shortened_k_1_to_3_terminal_and_timeout"]
    row = next(item for item in rows if item["n_decision_transitions"] == k and item["boundary"] == boundary)
    expected_return = sum((0.99**i) * float(i + 1) for i in range(k))
    assert row["rewards"] == pytest.approx(expected_return)
    assert row["next_observation_state"] == pytest.approx(float(k))
    assert row["done_mask"] == pytest.approx(expected_done)
    assert row["bootstrap_discount"] == pytest.approx(0.99)
    assert row["standard_k_step_discount"] == pytest.approx(0.99**k)
    assert row["raw_steps_executed_on_final_transition"] == 1
    if boundary == "timeout":
        assert row["stored_timeout_flag"] == pytest.approx(1.0)


def test_boundary_flush_does_not_include_following_episode_reward(audit: dict) -> None:
    checks = audit["synthetic_probe"]["episode_boundary_no_crossing_check"]
    assert checks["post_boundary_reward"] == 999.0
    assert checks["full_timeout_return_stayed_at_expected_four_step_sum"] is True
    assert checks["full_terminal_return_stayed_at_expected_four_step_sum"] is True
    assert checks["short_tail_returns_exclude_post_boundary_episode"] is True


def test_one_step_control_and_actual_soft_sac_target(audit: dict) -> None:
    cases = audit["synthetic_probe"]
    one_step = cases["one_step_control"]
    assert one_step["sample_discount_is_none"] is True
    assert one_step["sac_fallback_discount"] == pytest.approx(0.99)

    target = cases["fixed_sac_target_arithmetic"]
    assert target["soft_next_q"] == pytest.approx(3.1)
    assert target["timeout_current_protocol_target"] == pytest.approx(12.8704965)
    assert target["timeout_standard_gamma_to_four_target"] == pytest.approx(
        12.7793441
    )
    assert target["terminal_target_no_bootstrap"] == pytest.approx(9.8014965)


def test_source_and_task_horizon_facts_are_present(audit: dict) -> None:
    assert all(audit["source_contract_checks"].values())
    horizon = audit["intersection_sorted_depart4p0_task_horizon"]
    assert horizon["nominal_cap_seconds"] == pytest.approx(60.0)
    assert horizon["nominal_decisions_to_cap"] == 200
    assert horizon["nominal_4_transition_window_seconds"] == pytest.approx(1.2)
    assert horizon["remaining_time_token_matches"] == []
    assert "max_episode_steps=600 raw SUMO ticks" in horizon["paper_scenario_config"]
    scenario_sources = audit["source_locations"]["scenario_and_env"]
    assert scenario_sources["sorted_sumocfg_step_length_line"] is not None
    assert scenario_sources["raw_tick_length_sumo_command_line"] is not None
    assert scenario_sources["timeout_reward_line"] is not None
    assert "timeout reward is -5.0" in horizon["training_timeout_shaping"]
