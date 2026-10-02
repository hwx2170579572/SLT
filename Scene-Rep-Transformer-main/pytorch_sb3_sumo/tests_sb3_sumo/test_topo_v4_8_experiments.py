from __future__ import annotations

from pathlib import Path

import pytest

from configs.sb3_configs_v4_8 import V48_CANDIDATE
from tools import run_topo_v4_8_experiments as runner


def test_v4_8_contract_and_development_matrix_are_frozen() -> None:
    contract = runner.validate_contract(runner.load_contract())
    digest = runner.contract_sha256()
    jobs = runner.jobs_for_stage(contract, digest, "development")
    assert [job.job_id for job in jobs] == ["H1", "H2", "H3", "H4"]
    assert [job.scenario for job in jobs] == [
        "cross",
        "cross",
        "carla",
        "roundabout_medium",
    ]
    assert [job.seed for job in jobs] == [8, 9, 5, 3]
    assert [job.calibration_seed_start for job in jobs] == [77000, 78000, 79000, 80000]
    assert [job.evaluation_seed_start for job in jobs] == [68100, 69100, 70100, 71100]
    assert all(job.algorithm == V48_CANDIDATE for job in jobs)


def test_v4_8_command_uses_isolated_trainer_and_hash_bindings() -> None:
    contract = runner.validate_contract(runner.load_contract())
    job = runner.jobs_for_stage(contract, runner.contract_sha256(), "development")[0]
    hashes = {
        "experiment_contract_sha256": "a" * 64,
        "stage0_results_sha256": "b" * 64,
        "failure_attribution_sha256": "c" * 64,
        "implementation_freeze_sha256": "d" * 64,
    }
    command = runner.command_for(job, hashes, device="cuda")
    assert Path(command[1]).name == "train_paper_sb3_sumo_v4_8.py"
    assert command[command.index("--calibration-seed-start") + 1] == "77000"
    assert command[command.index("--evaluation-seed-start") + 1] == "68100"
    assert command[command.index("--implementation-freeze-sha256") + 1] == "d" * 64
    assert "--formal-unlock-receipt" not in command


def test_v4_8_formal_command_requires_promotion_receipt() -> None:
    contract = runner.validate_contract(runner.load_contract())
    job = runner.jobs_for_stage(contract, runner.contract_sha256(), "formal")[0]
    hashes = {
        "experiment_contract_sha256": "a" * 64,
        "stage0_results_sha256": "b" * 64,
        "failure_attribution_sha256": "c" * 64,
        "implementation_freeze_sha256": "d" * 64,
    }
    command = runner.command_for(job, hashes, device="cuda")
    assert command[command.index("--evaluation-split") + 1] == "test"
    assert command[command.index("--formal-unlock-receipt") + 1] == str(
        runner.DEFAULT_PROMOTION_GATE.resolve()
    )


def test_v4_8_contract_rejects_secondary_scope_drift() -> None:
    contract = runner.load_contract()
    contract["single_change"]["secondary_calibration"]["candidate_scope"] = "all_pairs"
    with pytest.raises(runner.V48ProtocolError, match="scope"):
        runner.validate_contract(contract)


def test_v4_8_versioned_writer_does_not_emit_parent_schema(tmp_path: Path) -> None:
    output = tmp_path / "status.json"
    runner._write_json(
        output,
        {
            "schema_version": "topo-scene-v4.7.status/v1",
            "nested": {"schema_version": "topo-scene-v4.7.stage-results/v1"},
        },
    )
    text = output.read_text(encoding="utf-8")
    assert "topo-scene-v4.8.status/v1" in text
    assert "topo-scene-v4.8.stage-results/v1" in text
    assert "topo-scene-v4.7." not in text


def test_v4_8_runtime_patch_restores_parent_runner() -> None:
    original = runner.scientific.stage_root
    with runner._patched_scientific_runner():
        assert runner.scientific.stage_root is runner.stage_root
        assert runner.scientific.V47_CANDIDATE == V48_CANDIDATE
    assert runner.scientific.stage_root is original
