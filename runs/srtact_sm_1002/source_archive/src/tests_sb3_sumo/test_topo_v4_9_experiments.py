from __future__ import annotations

from types import SimpleNamespace

from configs.sb3_configs_v4_9 import V49_CANDIDATE
from tools import run_topo_v4_9_experiments as runner


def test_v4_9_contract_and_fresh_matrices_are_locked() -> None:
    contract = runner.validate_contract(runner.load_contract())
    digest = runner.contract_sha256()
    development = runner.jobs_for_stage(contract, digest, "development")
    promotion = runner.jobs_for_stage(contract, digest, "promotion")
    formal = runner.jobs_for_stage(contract, digest, "formal")
    assert [(job.job_id, job.scenario, job.seed) for job in development] == [
        ("M1", "cross", 10),
        ("M2", "cross", 11),
        ("M3", "roundabout_medium", 12),
        ("M4", "carla", 13),
    ]
    assert len(promotion) == 12
    assert {job.seed for job in promotion} == {20, 21}
    assert len(formal) == 120
    assert all(job.algorithm == V49_CANDIDATE for job in development)


def test_v4_9_command_uses_isolated_trainer_and_llm_pipeline() -> None:
    contract = runner.validate_contract(runner.load_contract())
    digest = runner.contract_sha256()
    job = runner.jobs_for_stage(contract, digest, "promotion")[0]
    hashes = {
        "experiment_contract_sha256": "a" * 64,
        "stage0_results_sha256": "b" * 64,
        "failure_attribution_sha256": "c" * 64,
        "implementation_freeze_sha256": "d" * 64,
    }
    command = runner.command_for(job, hashes, device="cuda")
    assert command[0].replace("\\", "/").endswith(
        "/Anaconda/envs/llm_pipeline/python.exe"
    )
    assert command[1].endswith("train_paper_sb3_sumo_v4_9.py")
    assert "--formal-unlock-receipt" not in command
    formal = runner.jobs_for_stage(contract, digest, "formal")[0]
    formal_command = runner.command_for(formal, hashes, device="cuda")
    assert "--formal-unlock-receipt" in formal_command


def _row(job, *, success: float, collision: float):
    return {
        "job_id": job.job_id,
        "scenario": job.scenario,
        "success_rate": success,
        "collision_rate": collision,
        "off_route_rate": 0.0,
        "timeout_rate": max(0.0, 1.0 - success - collision),
        "learned_collision_critic_present": True,
        "risk_adjusted_score_equation_match_rate": 1.0,
        "inference_safety_rule_added": False,
        "selected_action_mask_feasible_rate": 1.0,
        "stored_collision_event_count": 5,
        "sampled_positive_collision_label_rate": 0.1,
    }


def test_development_decision_checks_every_cell_before_failing(monkeypatch) -> None:
    contract = runner.validate_contract(runner.load_contract())
    digest = runner.contract_sha256()
    jobs = runner.jobs_for_stage(contract, digest, "development")
    visited = []

    monkeypatch.setattr(runner, "accepted_run_reason", lambda job, hashes: (True, "accepted"))

    def fake_row(job):
        visited.append(job.job_id)
        if job.job_id == "M1":
            return _row(job, success=0.20, collision=0.70)
        if job.scenario == "carla":
            return _row(job, success=0.60, collision=0.05)
        return _row(job, success=0.70, collision=0.20)

    monkeypatch.setattr(runner, "_run_row", fake_row)
    result = runner.development_status(contract, digest, {})
    assert visited == [job.job_id for job in jobs]
    assert result["decision"] == "fail"
    assert result["jobs"]["M4"]["gate"]["passed"] is True
    assert result["execution_policy"] == "attempt_all_then_summarize"


def test_execute_jobs_continues_after_subprocess_failure(tmp_path, monkeypatch) -> None:
    contract = runner.validate_contract(runner.load_contract())
    digest = runner.contract_sha256()
    jobs = runner.jobs_for_stage(contract, digest, "development")
    attempted = []
    successful = set()

    monkeypatch.setattr(runner, "_assert_stage_can_run", lambda hashes, stage: None)
    monkeypatch.setattr(runner, "write_plan", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "summarize_stage", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "write_development_decision", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "stage_root", lambda stage: tmp_path / stage)
    monkeypatch.setattr(
        runner,
        "command_for",
        lambda job, hashes, device: [job.name],
    )
    monkeypatch.setattr(
        runner,
        "accepted_run",
        lambda job, hashes: job.name in successful,
    )
    monkeypatch.setattr(
        runner,
        "accepted_run_reason",
        lambda job, hashes: (
            (True, "accepted")
            if job.name in successful
            else (False, "synthetic failure")
        ),
    )

    def fake_run(command, **kwargs):
        name = command[0]
        attempted.append(name)
        if len(attempted) != 1:
            successful.add(name)
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.execute_jobs(
        contract,
        digest,
        "development",
        {},
        selector="all",
        device="cpu",
        workers=1,
        dry_run=False,
    )
    assert result == 1
    assert attempted == [job.name for job in jobs]

