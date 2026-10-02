from __future__ import annotations

import pytest

from tools import run_remaining_v4_8_3_promotion as automation
from tools import run_topo_v4_8_experiments as v48


def _state(
    name: str,
    *,
    accepted: bool,
    directory_exists: bool,
) -> automation.PromotionJobState:
    return automation.PromotionJobState(
        name=name,
        accepted=accepted,
        acceptance_reason="accepted" if accepted else "incomplete",
        directory_exists=directory_exists,
    )


def test_frozen_promotion_order_is_baseline_first() -> None:
    contract = v48.validate_contract(v48.load_contract())
    jobs = v48.jobs_for_stage(contract, v48.contract_sha256(), "promotion")
    assert [job.name for job in jobs] == [
        "P__tg__cross__s0__p42f54135",
        "P__tg__cross__s1__p42f54135",
        "P__tg__ram__s0__p42f54135",
        "P__tg__ram__s1__p42f54135",
        "P__tg__carla__s0__p42f54135",
        "P__tg__carla__s1__p42f54135",
        "P__cand__cross__s0__p42f54135",
        "P__cand__cross__s1__p42f54135",
        "P__cand__ram__s0__p42f54135",
        "P__cand__ram__s1__p42f54135",
        "P__cand__carla__s0__p42f54135",
        "P__cand__carla__s1__p42f54135",
    ]


def test_takeover_waits_for_first_incomplete_job() -> None:
    states = [
        _state("a", accepted=True, directory_exists=True),
        _state("b", accepted=True, directory_exists=True),
        _state("c", accepted=False, directory_exists=True),
        _state("d", accepted=False, directory_exists=False),
    ]
    assert automation.takeover_job_name(states) == "c"


def test_takeover_rejects_out_of_order_incomplete_directory() -> None:
    states = [
        _state("a", accepted=True, directory_exists=True),
        _state("b", accepted=False, directory_exists=False),
        _state("c", accepted=False, directory_exists=True),
    ]
    with pytest.raises(automation.PromotionAutomationError, match="out of frozen order"):
        automation.takeover_job_name(states)


def test_takeover_rejects_multiple_incomplete_directories() -> None:
    states = [
        _state("a", accepted=True, directory_exists=True),
        _state("b", accepted=False, directory_exists=True),
        _state("c", accepted=False, directory_exists=True),
    ]
    with pytest.raises(automation.PromotionAutomationError, match="multiple immutable"):
        automation.takeover_job_name(states)


def test_runner_commands_are_promotion_only_and_serial() -> None:
    command = automation.runner_command("python", "run_all_promotion", device="cuda")
    assert command[-8:] == [
        "run",
        "--stage",
        "promotion",
        "--job",
        "all",
        "--device",
        "cuda",
        "--workers",
        "1",
    ][-8:]
    assert "formal" not in command
    assert automation.runner_command("python", "gate_promotion", device="cuda")[-3:] == [
        "gate",
        "--stage",
        "promotion",
    ]
    with pytest.raises(automation.PromotionAutomationError, match="unsupported"):
        automation.runner_command("python", "run_formal", device="cuda")


def test_completion_guard_requires_12_promotion_and_zero_formal() -> None:
    valid = {
        "stages": {
            "promotion": {"accepted": 12, "complete": True},
            "formal": {"accepted": 0, "complete": False},
        }
    }
    automation.assert_promotion_complete_without_formal(valid)
    invalid = {
        "stages": {
            "promotion": {"accepted": 12, "complete": True},
            "formal": {"accepted": 1, "complete": False},
        }
    }
    with pytest.raises(automation.PromotionAutomationError, match="formal stage"):
        automation.assert_promotion_complete_without_formal(invalid)
