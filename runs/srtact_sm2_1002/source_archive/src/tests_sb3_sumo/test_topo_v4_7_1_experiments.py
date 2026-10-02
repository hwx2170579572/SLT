from __future__ import annotations

from pathlib import Path

from tools import run_topo_v4_7_1_experiments as patch
from tools import run_topo_v4_7_experiments as frozen


def _context():
    scientific = frozen.validate_contract(
        frozen.load_contract(patch.DEFAULT_SCIENTIFIC_CONTRACT)
    )
    patch.validate_patch_contract(patch.load_patch_contract())
    return scientific, frozen.contract_sha256(patch.DEFAULT_SCIENTIFIC_CONTRACT)


def _hashes() -> dict[str, str]:
    return {
        "experiment_contract_sha256": "a" * 64,
        "stage0_results_sha256": "b" * 64,
        "stage0_attribution_sha256": "c" * 64,
        "failure_attribution_sha256": "d" * 64,
        "deep_attribution_sha256": "e" * 64,
        "preregistration_receipt_sha256": "f" * 64,
        "engineering_patch_contract_sha256": "2" * 64,
        "implementation_freeze_sha256": "1" * 64,
    }


def test_v4_7_1_patch_contract_freezes_science_and_invalid_attempt() -> None:
    contract = patch.validate_patch_contract(patch.load_patch_contract())
    assert contract["scientific_protocol"]["unchanged"] is True
    assert contract["retry"]["from_scratch"] is True
    assert contract["retry"]["allowed_attempts"] == 1
    assert contract["rejected_attempt"]["validation_outcome_gate_eligible"] is False


def test_v4_7_1_uses_separate_roots_and_patched_trainer() -> None:
    contract, digest = _context()
    job = patch.jobs_for_stage(contract, digest, "development")[0]
    command = patch.command_for(job, _hashes(), device="cuda")
    assert Path(command[1]).name == "train_paper_sb3_sumo_v4_7_1.py"
    output = Path(command[command.index("--output-dir") + 1])
    assert output == patch.stage_root("development") / "runs"
    assert "results_topo_v4_7_1_dev" in str(patch.run_directory(job))
    assert command[command.index("--experiment-contract-sha256") + 1] == "a" * 64


def test_v4_7_1_patch_context_restores_frozen_runner() -> None:
    original_root = frozen.stage_root
    original_command = frozen.command_for
    with patch._patched_frozen_runner():
        assert frozen.stage_root is patch.stage_root
        assert frozen.command_for is patch.command_for
    assert frozen.stage_root is original_root
    assert frozen.command_for is original_command


def test_v4_7_1_parent_scientific_freeze_is_still_exact() -> None:
    assert frozen._sha256(patch.PARENT_V47_FREEZE) == patch.PARENT_V47_FREEZE_SHA256
    frozen.validate_implementation_freeze(patch.PARENT_V47_FREEZE)
