from __future__ import annotations

from pathlib import Path

from tools import run_topo_v4_7_2_experiments as patch
from tools import run_topo_v4_7_1_experiments as parent_patch
from tools import run_topo_v4_7_experiments as scientific


def _context():
    contract = scientific.validate_contract(scientific.load_contract())
    patch.validate_patch_contract(patch.load_patch_contract())
    return contract, scientific.contract_sha256()


def _hashes() -> dict[str, str]:
    return {
        "experiment_contract_sha256": "a" * 64,
        "stage0_results_sha256": "b" * 64,
        "failure_attribution_sha256": "c" * 64,
        "implementation_freeze_sha256": "d" * 64,
    }


def test_v4_7_2_contract_changes_only_runner_context() -> None:
    contract = patch.validate_patch_contract(patch.load_patch_contract())
    assert contract["scientific_protocol"]["unchanged"] is True
    assert contract["single_engineering_change"]["removed_context_override"] == "DEFAULT_PREREGISTRATION"
    assert contract["parent_engineering_patch"]["training_adapter_changed"] is False


def test_v4_7_2_context_preserves_parent_preregistration_identity() -> None:
    before = scientific.DEFAULT_PREREGISTRATION
    with patch._patched_scientific_runner():
        assert scientific.DEFAULT_PREREGISTRATION is before
        assert scientific.stage_root is patch.stage_root
    assert scientific.DEFAULT_PREREGISTRATION is before


def test_v4_7_2_preflight_revalidates_parent_and_targets_g1() -> None:
    result = patch.corrected_context_preflight()
    assert result["parent_preregistration_identity_preserved"] is True
    assert result["parent_v4_7_freeze_revalidated"] is True
    assert result["next_job"] == "G1"
    assert result["trainer"] == "train_paper_sb3_sumo_v4_7_1.py"
    assert result["formal_test_accessed"] is False


def test_v4_7_2_command_uses_isolated_root_and_frozen_writer() -> None:
    contract, digest = _context()
    job = patch.jobs_for_stage(contract, digest, "development")[0]
    command = patch.command_for(job, _hashes(), device="cuda")
    assert Path(command[1]).name == "train_paper_sb3_sumo_v4_7_1.py"
    assert "results_topo_v4_7_2_dev" in command[command.index("--output-dir") + 1]
    assert scientific._sha256(patch.PARENT_V471_FREEZE) == patch.PARENT_V471_FREEZE_SHA256
    parent_patch.validate_implementation_freeze(patch.PARENT_V471_FREEZE)
