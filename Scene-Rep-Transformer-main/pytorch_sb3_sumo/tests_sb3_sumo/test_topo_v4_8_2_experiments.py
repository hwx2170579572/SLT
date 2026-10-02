from __future__ import annotations

from tools import run_topo_v4_8_1_experiments as acceptance_patch
from tools import run_topo_v4_8_2_experiments as runner
from tools import run_topo_v4_8_experiments as v48


def test_v4_8_2_patch_contract_is_gate_only() -> None:
    contract = runner.validate_patch_contract(runner.load_patch_contract())
    change = contract["single_engineering_change"]
    assert change["old_expected_mode"] == "joint_checkpoint_decoder"
    assert change["new_expected_mode"] == "tie_only_replicated_joint_checkpoint_decoder"
    assert change["acceptance_adapter_changed"] is False
    assert change["training_changed"] is False
    assert change["selector_changed"] is False
    assert change["thresholds_changed"] is False
    assert change["h1_rerun_forbidden"] is True


def test_v4_8_2_h1_preflight_reproduces_both_bugs_and_adopts_in_place() -> None:
    before = acceptance_patch._validate_h1_hashes()
    value = runner.h1_adoption_preflight()
    after = acceptance_patch._validate_h1_hashes()
    assert value["original_v4_8_acceptance_rejection_reproduced"] is True
    assert value["v4_8_1_acceptance_passed"] is True
    assert value["inherited_gate_failed_checks"] == ["selector_mode"]
    assert value["h1_outcome_gate_passed"] is True
    assert value["h1_mechanism_gate_passed"] is True
    assert value["h1_other_selector_checks_passed"] is True
    assert value["patched_gate_passed"] is True
    assert value["next_development_job"] == "H2"
    assert value["h1_rerun"] is False
    assert value["h1_artifacts_modified"] is False
    assert before == after == value["h1_artifact_hashes"]


def test_v4_8_2_gate_adapter_changes_only_selector_mode_check() -> None:
    contract = v48.validate_contract(v48.load_contract())
    hashes = v48.protocol_hashes(freeze_path=v48.DEFAULT_FREEZE)
    job = v48.jobs_for_stage(contract, v48.contract_sha256(), "development")[0]
    with runner._patched_v4_8_2_interfaces(), v48._patched_scientific_runner():
        row = v48.scientific._run_row(job)
    inherited = runner._FROZEN_CANDIDATE_GATE(row, contract)
    patched = runner._candidate_gate_v4_8_2(row, contract)
    changed = {
        key
        for key in inherited["checks"]
        if inherited["checks"][key] != patched["checks"][key]
    }
    assert changed == {"selector_mode"}
    assert inherited["passed"] is False
    assert patched["passed"] is True
    assert patched["expected_selector_mode"] == row["selector_mode"]


def test_v4_8_2_contexts_restore_parent_functions() -> None:
    original_validator = v48._validate_return_estimator
    original_assert = v48._assert_stage_can_run
    original_gate = v48.scientific._candidate_gate
    with runner._patched_v4_8_2_interfaces():
        assert v48._validate_return_estimator is acceptance_patch._validate_return_estimator_v4_8_1
        assert v48._assert_stage_can_run is runner._assert_stage_can_run_v4_8_2
        with runner._patched_scientific_gate():
            assert v48.scientific._candidate_gate is runner._candidate_gate_v4_8_2
    assert v48._validate_return_estimator is original_validator
    assert v48._assert_stage_can_run is original_assert
    assert v48.scientific._candidate_gate is original_gate

