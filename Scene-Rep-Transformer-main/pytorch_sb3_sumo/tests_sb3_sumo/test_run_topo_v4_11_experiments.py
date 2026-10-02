from __future__ import annotations

from types import SimpleNamespace

import pytest

from configs.sb3_configs_v4_11 import V411_FULL
from tools import run_topo_v4_11_experiments as subject


def _job(job_id: str, seed: int) -> subject.Job:
    return subject.Job(
        stage="development",
        job_id=job_id,
        kind="test",
        method="prcr_full",
        algorithm=V411_FULL,
        implementation_id=subject.V411_IMPLEMENTATION_IDS[V411_FULL],
        scenario="cross",
        seed=seed,
        raw_steps=20_000,
        calibration_episodes=12,
        calibration_seed_start=131_000 + seed,
        evaluation_episodes=20,
        evaluation_split="validation",
        evaluation_seed_start=141_000 + seed,
        role="test",
        protocol_tag=subject.CONTRACT_SHA256[:8],
    )


def _passing_candidate_row() -> dict[str, object]:
    return {
        "scenario": "cross",
        "success_rate": 0.65,
        "collision_rate": 0.20,
        "off_route_rate": 0.0,
        "timeout_rate": 0.15,
        "shared_risk_encoder_present": True,
        "exact_supported_mixture_model_argmax_rate": 1.0,
        "supported_mixture_score_equation_match_rate": 1.0,
        "learned_proposal_source_rate": 1.0,
        "no_action_rewrite_rate": 1.0,
        "optimizer_overlap_count": 0,
        "inference_safety_rule_added": False,
        "external_kinematic_projection": False,
        "kinematic_safety_projection": False,
        "traffic_risk_in_lane_mask": False,
        "action_postprocessing_override": False,
        "selected_action_mask_feasible_rate": 1.0,
        "collision_return_loss": "soft_target_binary_cross_entropy_with_logits",
        "collision_pairwise_rank_coef": 0.25,
        "collision_temporal_consistency_coef": 0.05,
        "collision_soft_bce_loss": 0.25,
        "collision_pairwise_rank_loss": 0.15,
        "collision_pairwise_active_pair_rate_maximum": 0.4,
        "collision_temporal_consistency_loss": 0.01,
        "collision_probability_absolute_error": 0.2,
    }


def test_contract_and_all_stage_matrices_are_frozen() -> None:
    contract = subject.validate_contract(subject.load_contract())
    assert {
        stage: len(subject.jobs_for_stage(contract, subject.CONTRACT_SHA256, stage))
        for stage in subject.STAGES
    } == {
        "development": 4,
        "ablation": 2,
        "promotion": 12,
        "formal": 120,
    }


def test_command_binds_model_evidence_and_formal_receipt_only_for_formal() -> None:
    contract = subject.validate_contract(subject.load_contract())
    hashes = {
        "experiment_contract_sha256": "a" * 64,
        "stage0_results_sha256": "b" * 64,
        "attribution_sha256": "c" * 64,
        "implementation_freeze_sha256": "d" * 64,
    }
    development = subject.jobs_for_stage(
        contract, subject.CONTRACT_SHA256, "development"
    )[0]
    command = subject.command_for(development, hashes, device="cuda")
    assert "tools\\train_paper_sb3_sumo_v4_11.py" in command[1]
    assert "--formal-unlock-receipt" not in command
    assert command[command.index("--ego-control-profile") + 1] == "direct"
    assert command[command.index("--implementation-freeze-sha256") + 1] == "d" * 64
    formal = subject.jobs_for_stage(contract, subject.CONTRACT_SHA256, "formal")[0]
    assert "--formal-unlock-receipt" in subject.command_for(
        formal, hashes, device="cuda"
    )


def test_candidate_gate_requires_proper_model_diagnostics_and_no_projection() -> None:
    contract = subject.validate_contract(subject.load_contract())
    row = _passing_candidate_row()
    assert subject._candidate_gate(row, contract)["passed"] is True

    row["collision_pairwise_active_pair_rate_maximum"] = 0.0
    inactive = subject._candidate_gate(row, contract)
    assert inactive["outcome_passed"] is True
    assert inactive["model_integrity_passed"] is False
    assert inactive["passed"] is False

    row = _passing_candidate_row()
    row["kinematic_safety_projection"] = True
    projected = subject._candidate_gate(row, contract)
    assert projected["model_integrity_passed"] is False
    assert projected["checks"]["no_kinematic_safety_projection"] is False


def test_execute_jobs_attempts_every_selected_job_after_failures(
    monkeypatch, tmp_path
) -> None:
    jobs = [_job("D1", 10), _job("D2", 11), _job("D3", 12)]
    calls: list[str] = []
    writes: list[dict[str, object]] = []

    monkeypatch.setattr(subject, "_assert_stage_can_run", lambda *args: None)
    monkeypatch.setattr(subject, "jobs_for_stage", lambda *args: jobs)
    monkeypatch.setattr(subject, "stage_root", lambda stage: tmp_path / stage)
    monkeypatch.setattr(subject, "accepted_run", lambda *args: False)
    monkeypatch.setattr(
        subject, "accepted_run_reason", lambda *args: (False, "synthetic failure")
    )
    monkeypatch.setattr(subject, "write_plan", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(subject, "command_for", lambda job, *args, **kwargs: [job.name])
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda command, **kwargs: (
            calls.append(command[0]) or SimpleNamespace(returncode=9)
        ),
    )
    monkeypatch.setattr(subject, "_write_json", lambda path, value: writes.append(value))
    monkeypatch.setattr(
        subject,
        "summarize_stage",
        lambda *args, **kwargs: {"complete": False},
    )
    code = subject.execute_jobs(
        {},
        subject.CONTRACT_SHA256,
        "development",
        {},
        selector="all",
        device="cuda",
        workers=1,
        dry_run=False,
    )
    assert code == 1
    assert calls == [job.name for job in jobs]
    assert len(writes) == 1
    assert writes[0]["failure_does_not_cancel_remaining_jobs"] is True
    assert len(writes[0]["attempts"]) == 3
    assert all(row["status"] == "failed" for row in writes[0]["attempts"])


def test_formal_stage_cannot_run_without_passing_promotion(monkeypatch) -> None:
    monkeypatch.setattr(subject, "validate_implementation_freeze", lambda: {})
    monkeypatch.setattr(subject, "validate_engineering_receipt", lambda: {})
    monkeypatch.setattr(subject, "validate_development_decision", lambda: {})
    monkeypatch.setattr(subject, "validate_ablation_decision", lambda: {})
    monkeypatch.setattr(
        subject,
        "_load_json",
        lambda path: {
            "decision": "fail",
            "experiment_contract_sha256": "a" * 64,
            "implementation_freeze_sha256": "b" * 64,
        },
    )
    with pytest.raises(subject.V411ProtocolError, match="promotion failed"):
        subject._assert_stage_can_run(
            {
                "experiment_contract_sha256": "a" * 64,
                "implementation_freeze_sha256": "b" * 64,
            },
            "formal",
        )
