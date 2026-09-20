from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_topo_v4_13_experiments as subject


HASHES = {
    "experiment_contract_sha256": subject.CONTRACT_SHA256,
    "stage0_results_sha256": subject.PARENT_DEVELOPMENT_SUMMARY_SHA256,
    "attribution_sha256": subject.PARENT_ATTRIBUTION_SHA256,
    "preregistration_receipt_sha256": "pre",
    "implementation_freeze_sha256": "freeze",
    "engineering_receipt_sha256": "engineering",
}


def _candidate_row(job: subject.Job, *, success: float, collision: float) -> dict:
    return {
        "job": job.name,
        "job_id": job.job_id,
        "method": job.method,
        "scenario": job.scenario,
        "seed": job.seed,
        "success_rate": success,
        "collision_rate": collision,
        "off_route_rate": 0.0,
        "timeout_rate": 0.0,
        "joint_replay_action_support_present": True,
        "lane_support_gradient_isolated": True,
        "same_deployed_lane_head_for_support": True,
        "exact_joint_support_model_argmax_rate": 1.0,
        "joint_support_score_equation_match_rate": 1.0,
        "learned_proposal_source_rate": 1.0,
        "no_action_rewrite_rate": 1.0,
        "no_actor_confidence_threshold_rate": 1.0,
        "optimizer_overlap_count": 0,
        "inference_safety_rule_added": False,
        "external_kinematic_projection": False,
        "kinematic_safety_projection": False,
        "traffic_risk_in_lane_mask": False,
        "ttc_or_headway_threshold": False,
        "lane_change_veto": False,
        "actor_confidence_gate": False,
        "action_postprocessing_override": False,
        "scenario_conditioned_inference_rule": False,
        "selected_action_mask_feasible_rate": 1.0,
        "collision_return_loss": "soft_target_binary_cross_entropy_with_logits",
        "collision_pairwise_rank_coef": 0.25,
        "collision_temporal_consistency_coef": 0.05,
        "replay_joint_action_nll": 1.0,
        "replay_lane_categorical_nll": 0.5,
    }


def test_contract_and_all_stage_matrices_are_frozen() -> None:
    contract = subject.validate_contract(subject.load_contract())
    digest = subject._sha256(subject.DEFAULT_CONTRACT)
    expected = {"development": 4, "ablation": 3, "promotion": 12, "formal": 120}
    for stage, count in expected.items():
        jobs = subject.jobs_for_stage(contract, digest, stage)
        assert len(jobs) == count
        assert len({job.name for job in jobs}) == count
        assert all(len(job.name) <= 42 for job in jobs)


def test_command_uses_v413_trainer_short_root_and_formal_receipt() -> None:
    contract = subject.load_contract()
    digest = subject._sha256(subject.DEFAULT_CONTRACT)
    development = subject.jobs_for_stage(contract, digest, "development")[0]
    command = subject.command_for(development, HASHES, device="cpu", attempt=2)
    assert command[1].endswith("train_paper_sb3_sumo_v4_13.py")
    assert "--formal-unlock-receipt" not in command
    assert "__r2" in command[command.index("--model-name") + 1]
    formal = subject.jobs_for_stage(contract, digest, "formal")[0]
    assert "--formal-unlock-receipt" in subject.command_for(
        formal, HASHES, device="cuda"
    )


def test_development_gate_requires_d3_recovery_and_cross_retention(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = subject.load_contract()
    digest = subject._sha256(subject.DEFAULT_CONTRACT)
    jobs = subject.jobs_for_stage(contract, digest, "development")
    outcomes = {
        "D1": (0.85, 0.15),
        "D2": (0.80, 0.20),
        "D3": (0.85, 0.15),
        "D4": (0.95, 0.05),
    }
    rows = [
        _candidate_row(
            job,
            success=outcomes[job.job_id][0],
            collision=outcomes[job.job_id][1],
        )
        for job in jobs
    ]
    monkeypatch.setattr(
        subject,
        "summarize_stage",
        lambda *args, **kwargs: {
            "complete": True,
            "per_run": rows,
            "accepted_runs": 4,
            "expected_runs": 4,
        },
    )
    monkeypatch.setattr(
        subject, "DEFAULT_DEVELOPMENT_DECISION", tmp_path / "decision.json"
    )
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(subject.REPORT_ROOTS, "development", tmp_path)
    value = subject.compute_development_gate(contract, digest, HASHES)
    assert value["decision"] == "pass"
    assert value["checks"]["D3_improves_parent"] is True
    assert value["checks"]["cross_aggregate_success"] is True

    rows[2]["success_rate"] = 0.80
    rows[2]["collision_rate"] = 0.20
    value = subject.compute_development_gate(contract, digest, HASHES)
    assert value["decision"] == "fail"


def test_execute_jobs_attempts_every_selected_job_after_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = subject.load_contract()
    digest = subject._sha256(subject.DEFAULT_CONTRACT)
    calls: list[list[str]] = []
    monkeypatch.setattr(subject, "_assert_stage_can_run", lambda *args: None)
    monkeypatch.setitem(subject.REPORT_ROOTS, "development", tmp_path / "reports")
    monkeypatch.setitem(subject.SHORT_RUN_ROOTS, "development", tmp_path / "runs")
    monkeypatch.setattr(subject, "accepted_run_location", lambda *args: None)
    monkeypatch.setattr(subject, "accepted_run", lambda *args: False)
    monkeypatch.setattr(
        subject,
        "_validate_run",
        lambda *args: (_ for _ in ()).throw(ValueError("invalid")),
    )
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or SimpleNamespace(returncode=7),
    )
    monkeypatch.setattr(
        subject,
        "summarize_stage",
        lambda *args, **kwargs: {
            "complete": False,
            "accepted_runs": 0,
            "expected_runs": 4,
        },
    )
    code = subject.execute_jobs(
        contract,
        digest,
        "development",
        HASHES,
        selector="all",
        device="cpu",
        workers=1,
        dry_run=False,
    )
    assert code == 1
    assert len(calls) == 4
    execution = subject._load_json(
        tmp_path / "reports" / "last_execution.json"
    )
    assert len(execution["attempts"]) == 4
    assert execution["failure_does_not_cancel_remaining_jobs"] is True


def test_formal_stage_remains_locked_on_failed_promotion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subject, "validate_implementation_freeze", lambda: {})
    monkeypatch.setattr(subject, "validate_engineering_receipt", lambda: {})
    monkeypatch.setattr(subject, "validate_development_decision", lambda: {})
    monkeypatch.setattr(subject, "validate_ablation_decision", lambda: {})
    receipt = tmp_path / "promotion_gate.json"
    receipt.write_text('{"decision":"fail"}', encoding="utf-8")
    monkeypatch.setattr(subject, "DEFAULT_PROMOTION_GATE", receipt)
    with pytest.raises(subject.V413ProtocolError, match="promotion failed"):
        subject._assert_stage_can_run(HASHES, "formal")
