from __future__ import annotations

from tools import run_topo_v4_8_experiments as v48
from tools import run_topo_v4_8_2_experiments as parent_patch
from tools import run_topo_v4_8_3_experiments as runner


def _jobs() -> tuple[v48.Job, v48.Job, dict[str, str]]:
    contract = v48.validate_contract(v48.load_contract())
    digest = v48.contract_sha256()
    hashes = v48.protocol_hashes(freeze_path=v48.DEFAULT_FREEZE)
    promotion = next(
        job
        for job in v48.jobs_for_stage(contract, digest, "promotion")
        if job.name == runner.PROMOTION_JOB_NAME
    )
    candidate = v48.jobs_for_stage(contract, digest, "development")[0]
    return promotion, candidate, hashes


def test_v4_8_3_patch_contract_is_control_acceptance_only() -> None:
    contract = runner.validate_patch_contract(runner.load_patch_contract())
    change = contract["single_engineering_change"]
    assert change["acceptance_only"] is True
    assert change["control_only"] is True
    assert change["control_expected_value"] is None
    assert change["candidate_expected_value"] == "calibration_seed_start_plus_100"
    assert change["candidate_acceptance_changed"] is False
    assert change["training_changed"] is False
    assert change["selector_changed"] is False
    assert change["scientific_gates_changed"] is False
    assert change["completed_promotion_job_rerun_forbidden"] is True


def test_v4_8_3_preflight_reproduces_rejection_and_adopts_in_place() -> None:
    before = runner._validate_promotion_artifact_hashes()
    value = runner.promotion_adoption_preflight()
    after = runner._validate_promotion_artifact_hashes()
    assert value["frozen_v4_8_2_rejection_reproduced"] is True
    assert value["frozen_v4_8_2_reason"] == "argument secondary_calibration_seed_start mismatch"
    assert value["patched_runner_accepts_same_artifacts"] is True
    assert value["delegated_frozen_validator_after_single_read_only_normalization"] is True
    assert value["required_artifact_count"] == 13
    assert value["existing_candidate_acceptance_preserved"] is True
    assert value["control_non_null_mutation_rejected"] is True
    assert value["candidate_null_mutation_rejected"] is True
    assert value["promotion_job_rerun"] is False
    assert value["promotion_artifacts_modified"] is False
    assert value["next_promotion_job"] == runner.NEXT_PROMOTION_JOB
    assert before == after == value["promotion_artifact_hashes"]


def test_v4_8_3_changes_only_control_acceptance() -> None:
    promotion, candidate, hashes = _jobs()
    with parent_patch._patched_v4_8_2_interfaces():
        original_control = v48.accepted_run_reason(promotion, hashes)
        original_candidate = v48.accepted_run_reason(candidate, hashes)
    with runner._patched_v4_8_3_interfaces():
        patched_control = v48.accepted_run_reason(promotion, hashes)
        patched_candidate = v48.accepted_run_reason(candidate, hashes)
    assert original_control == (False, "argument secondary_calibration_seed_start mismatch")
    assert patched_control == (True, "accepted")
    assert original_candidate == patched_candidate == (True, "accepted")


def test_v4_8_3_mutation_guards_are_directional_and_read_only() -> None:
    promotion, candidate, hashes = _jobs()
    before = runner._validate_promotion_artifact_hashes()
    with runner._patched_v4_8_3_interfaces():
        with runner._temporary_secondary_seed_view(
            promotion, promotion.calibration_seed_start + 100
        ):
            control = v48.accepted_run_reason(promotion, hashes)
        with runner._temporary_secondary_seed_view(candidate, None):
            candidate_result = v48.accepted_run_reason(candidate, hashes)
    after = runner._validate_promotion_artifact_hashes()
    assert control == (False, "control secondary_calibration_seed_start must be null")
    assert candidate_result == (False, "argument secondary_calibration_seed_start mismatch")
    assert before == after


def test_v4_8_3_contexts_restore_parent_functions() -> None:
    original_validator = v48._validate_return_estimator
    original_reason = v48.accepted_run_reason
    original_accepted = v48.accepted_run
    original_assert = v48._assert_stage_can_run
    original_gate = v48.scientific._candidate_gate
    with runner._patched_v4_8_3_interfaces():
        assert v48.accepted_run_reason is runner.accepted_run_reason_v4_8_3
        assert v48.accepted_run is runner.accepted_run_v4_8_3
        assert v48._assert_stage_can_run is runner._assert_stage_can_run_v4_8_3
        with parent_patch._patched_scientific_gate():
            assert v48.scientific._candidate_gate is parent_patch._candidate_gate_v4_8_2
    assert v48._validate_return_estimator is original_validator
    assert v48.accepted_run_reason is original_reason
    assert v48.accepted_run is original_accepted
    assert v48._assert_stage_can_run is original_assert
    assert v48.scientific._candidate_gate is original_gate


def test_v4_8_3_completed_development_supersedes_h1_only_snapshot() -> None:
    contract = v48.validate_contract(v48.load_contract())
    digest = v48.contract_sha256()
    hashes = v48.protocol_hashes(freeze_path=v48.DEFAULT_FREEZE)
    with (
        parent_patch._patched_v4_8_2_interfaces(),
        v48._patched_scientific_runner(),
        parent_patch._patched_scientific_gate(),
    ):
        status = v48.scientific.development_status(contract, digest, hashes)
    assert status["prerequisites"] == {"H1": True, "H2": True, "H3": True, "H4": True}
    assert status["complete"] is True
    assert status["decision"] == "pass"
    assert status["next_job"] is None
    assert status["formal_test_unlocked"] is False

