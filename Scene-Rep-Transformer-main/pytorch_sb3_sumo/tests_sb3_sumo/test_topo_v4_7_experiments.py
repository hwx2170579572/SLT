from __future__ import annotations

import copy
from pathlib import Path

import pytest

from configs.sb3_configs_v4_7 import (
    PARENT_CONTROL,
    V47_CANDIDATE,
    V47_FORMAL_ALGORITHMS,
)
from tools import run_topo_v4_7_experiments as v47


def _context():
    contract = v47.validate_contract(v47.load_contract(v47.DEFAULT_CONTRACT))
    return contract, v47.contract_sha256(v47.DEFAULT_CONTRACT)


def _hashes() -> dict[str, str]:
    return {
        "experiment_contract_sha256": "a" * 64,
        "stage0_results_sha256": "b" * 64,
        "stage0_attribution_sha256": "c" * 64,
        "failure_attribution_sha256": "d" * 64,
        "deep_attribution_sha256": "e" * 64,
        "preregistration_receipt_sha256": "f" * 64,
        "implementation_freeze_sha256": "1" * 64,
    }


def _episodes(successes: int, collisions: int) -> list[dict]:
    records = []
    for index in range(12):
        success = index < successes
        collision = successes <= index < successes + collisions
        records.append(
            {
                "episode": index,
                "seed": 67_000 + index,
                "traffic_variant": f"traffic_{index}.rou.xml",
                "success": success,
                "collision": collision,
                "off_route": False,
                "timeout": not success and not collision,
            }
        )
    return records


def _passing_row(
    *,
    algorithm: str = V47_CANDIDATE,
    successes: int = 8,
    collisions: int = 2,
    decoder: str = "target_critic",
) -> dict:
    records = _episodes(successes, collisions)
    row = {
        "algorithm": algorithm,
        "scenario": "cross",
        "success_rate": successes / 12,
        "collision_rate": collisions / 12,
        "off_route_rate": 0.0,
        "timeout_rate": (12 - successes - collisions) / 12,
        "calibration_episodes": 12,
        "selector_candidate_count": 4,
        "selector_mode": "joint_checkpoint_decoder",
        "selector_receipt_precedes_validation": True,
        "selected_deployment_decoder": decoder,
        "selected_decoder_records": 10,
        "selected_decoder_exact_rule_match_rate": 1.0,
        "selected_decoder_exact_action_match_rate": 1.0,
        "selected_action_mask_feasible_rate": 1.0,
        "hybrid_exact_lateral_code_rate": 1.0,
        "episode_records": records,
        "return_n_step": 16 if algorithm == V47_CANDIDATE else 4,
        "return_bootstrap_discount": (
            "gamma_power_actual_horizon"
            if algorithm == V47_CANDIDATE
            else "single_gamma_source_equivalent"
        ),
    }
    if decoder == "target_critic":
        row.update(
            {
                "exact_target_critic_argmax_rate": 1.0,
                "target_keep_tie_rule_match_rate": 1.0,
            }
        )
    else:
        row.update(
            {
                "actor_non_keep_confidence_threshold": 0.90,
                "actor_override_predicate_valid_rate": 1.0,
                "target_fallback_rule_match_rate": 1.0,
            }
        )
    return row


def test_v4_7_contract_and_stage_cardinalities() -> None:
    contract, digest = _context()
    development = v47.jobs_for_stage(contract, digest, "development")
    promotion = v47.jobs_for_stage(contract, digest, "promotion")
    formal = v47.jobs_for_stage(contract, digest, "formal")
    assert [job.job_id for job in development] == ["G1", "G2", "G3", "G4", "G5"]
    assert [job.algorithm for job in development] == [
        V47_CANDIDATE,
        PARENT_CONTROL,
        V47_CANDIDATE,
        V47_CANDIDATE,
        V47_CANDIDATE,
    ]
    assert [job.seed for job in development] == [6, 6, 8, 5, 3]
    assert [job.calibration_seed_start for job in development] == [
        76_000,
        76_000,
        77_000,
        78_000,
        79_000,
    ]
    assert [job.evaluation_seed_start for job in development] == [
        67_000,
        67_000,
        68_000,
        69_000,
        70_000,
    ]
    assert len(promotion) == 12
    assert len(formal) == 120
    assert PARENT_CONTROL not in V47_FORMAL_ALGORITHMS


def test_v4_7_commands_use_new_trainer_and_keep_formal_locked() -> None:
    contract, digest = _context()
    development = v47.jobs_for_stage(contract, digest, "development")
    candidate_command = v47.command_for(development[0], _hashes(), device="cuda")
    parent_command = v47.command_for(development[1], _hashes(), device="cuda")
    assert Path(candidate_command[1]).name == "train_paper_sb3_sumo_v4_7.py"
    assert candidate_command[candidate_command.index("--algo") + 1] == V47_CANDIDATE
    assert parent_command[parent_command.index("--algo") + 1] == PARENT_CONTROL
    assert "--formal-unlock-receipt" not in candidate_command
    formal = v47.jobs_for_stage(contract, digest, "formal")[0]
    formal_command = v47.command_for(formal, _hashes(), device="cuda")
    assert formal_command[formal_command.index("--evaluation-split") + 1] == "test"
    assert "--formal-unlock-receipt" in formal_command


def test_v4_7_contract_rejects_return_estimator_drift() -> None:
    contract, _ = _context()
    drifted = copy.deepcopy(contract)
    drifted["single_change"]["candidate_return_estimator"]["n_step"] = 8
    with pytest.raises(v47.V47ProtocolError, match="candidate n-step"):
        v47.validate_contract(drifted)


def test_v4_7_candidate_and_parent_integrity_gates() -> None:
    contract, _ = _context()
    candidate = v47._candidate_gate(_passing_row(), contract)
    parent = v47._parent_control_gate(
        _passing_row(algorithm=PARENT_CONTROL, successes=2, collisions=8), contract
    )
    assert candidate["passed"] is True
    assert parent["passed"] is True
    assert parent["outcome_not_used_as_standalone_gate"] is True

    bad_parent = _passing_row(algorithm=PARENT_CONTROL)
    bad_parent["return_n_step"] = 16
    assert v47._parent_control_gate(bad_parent, contract)["passed"] is False


def test_v4_7_paired_causal_gate_requires_noninferiority_and_positive_effect() -> None:
    contract, _ = _context()
    candidate = _passing_row(successes=8, collisions=2)
    parent = _passing_row(algorithm=PARENT_CONTROL, successes=6, collisions=4)
    passed = v47._causal_gate(candidate, parent, contract)
    assert passed["passed"] is True
    assert passed["success_count_gain"] == 2
    assert passed["collision_count_reduction"] == 2

    tie = v47._causal_gate(
        _passing_row(successes=6, collisions=4), parent, contract
    )
    assert tie["passed"] is False
    assert tie["checks"]["positive_effect"] is False

    worse = v47._causal_gate(
        _passing_row(successes=5, collisions=2), parent, contract
    )
    assert worse["passed"] is False
    assert worse["checks"]["candidate_success_not_lower"] is False


def test_v4_7_development_state_machine_runs_matched_control_before_g3(
    monkeypatch,
) -> None:
    contract, digest = _context()
    accepted_ids = {"G1"}
    rows = {
        "G1": _passing_row(successes=8, collisions=2),
        "G2": _passing_row(
            algorithm=PARENT_CONTROL, successes=6, collisions=4
        ),
    }

    def accepted(job, _hashes_value):
        del _hashes_value
        return (
            job.job_id in accepted_ids,
            "accepted" if job.job_id in accepted_ids else "absent",
        )

    monkeypatch.setattr(v47, "accepted_run_reason", accepted)
    monkeypatch.setattr(v47, "_run_row", lambda job: rows[job.job_id])
    status = v47.development_status(contract, digest, _hashes())
    assert status["decision"] == "incomplete"
    assert status["next_job"] == "G2"

    accepted_ids.add("G2")
    status = v47.development_status(contract, digest, _hashes())
    assert status["decision"] == "incomplete"
    assert status["next_job"] == "G3"
    assert status["paired_parent_causal_gate"]["passed"] is True

    rows["G1"] = _passing_row(successes=6, collisions=4)
    stopped = v47.development_status(contract, digest, _hashes())
    assert stopped["decision"] == "fail"
    assert stopped["next_job"] is None
    assert stopped["stopped_after"] == "G2"
    assert stopped["stop_reason"] == "paired_parent_causal_gate_failure"


def test_v4_7_real_attribution_is_revalidated() -> None:
    hashes = v47._validate_failure_attribution(
        v47.DEFAULT_FAILURE_ATTRIBUTION, v47.DEFAULT_DEEP_ATTRIBUTION
    )
    assert hashes["failure_attribution_sha256"] == v47.LINEAGE_HASHES[
        "failure_attribution_sha256"
    ]
