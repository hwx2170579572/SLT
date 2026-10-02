from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from tools import reproduce_paper_sb3_sumo as reproduction
from tools.analyze_systematic_matrix import (
    _bootstrap_paired_delta_interval,
    _effect_classification,
    _holm_adjust,
    analyze,
)
from tools.validate_systematic_matrix import (
    EXPECTED_METHODS,
    EXPECTED_RUNS,
    EXPECTED_SCENARIOS,
    validate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PROJECT_ROOT / "experiments" / "systematic_matrix" / "protocol.json"
)


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_systematic_protocol_is_machine_valid() -> None:
    result = validate(PROTOCOL_PATH)
    assert result["passed"] is True
    assert result["expected_runs_per_profile"] == EXPECTED_RUNS
    assert all(
        count == EXPECTED_RUNS
        for count in result["profile_job_counts"].values()
    )


def test_validator_can_gate_on_matching_complete_preflight(tmp_path) -> None:
    protocol_result = validate(PROTOCOL_PATH)
    report_path = tmp_path / "preflight_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "passed": True,
                "protocol_sha256": protocol_result["protocol_sha256"],
                "jobs_requested": EXPECTED_RUNS,
                "unique_algorithm_scenario_pairs_expected": 36,
                "unique_algorithm_scenario_pairs_checked": 36,
                "results": [{"returncode": 0}] * 36,
            }
        ),
        encoding="utf-8",
    )
    gated_result = validate(PROTOCOL_PATH, report_path)
    assert gated_result["passed"] is True
    assert all(
        row["passed"]
        for row in gated_result["checks"]
        if row["id"].startswith("preflight_")
        or row["id"] == "all_algorithm_scenario_pairs_checked"
    )


def test_systematic_matrix_generates_every_requested_cell() -> None:
    protocol = _protocol()
    args = argparse.Namespace(
        profile="paper", methods="all", scenarios="all", seeds="all"
    )
    jobs = reproduction.make_jobs(args, protocol)
    assert len(jobs) == EXPECTED_RUNS
    assert len({job.name for job in jobs}) == EXPECTED_RUNS
    assert {job.method for job in jobs} == set(EXPECTED_METHODS)
    assert {job.scenario for job in jobs} == set(EXPECTED_SCENARIOS)
    assert {job.seed for job in jobs} == {0, 1, 2}
    assert {job.test_episodes for job in jobs} == {50}


def test_systematic_carla_extensions_are_explicit() -> None:
    methods = _protocol()["supported_methods"]
    assert methods["sac"]["carla_evidence_class"] == (
        "controlled_extension_not_reported_in_paper"
    )
    assert methods["mst"]["carla_evidence_class"] == (
        "controlled_extension_not_reported_in_paper"
    )
    assert "carla" in methods["sac"]["scenarios"]
    assert "carla" in methods["mst"]["scenarios"]


def test_parser_uses_first_protocol_method_as_default(monkeypatch) -> None:
    monkeypatch.setattr(reproduction, "PROTOCOL_PATH", PROTOCOL_PATH)
    args = reproduction.parser().parse_args(["plan", "--profile", "paper"])
    assert args.methods == "sac"


def test_preflight_persists_machine_report(tmp_path, monkeypatch) -> None:
    protocol_copy = tmp_path / "protocol.json"
    protocol_copy.write_bytes(PROTOCOL_PATH.read_bytes())
    monkeypatch.setattr(reproduction, "PROTOCOL_PATH", protocol_copy)
    protocol = _protocol()
    job = reproduction.make_jobs(
        argparse.Namespace(
            profile="paper", methods="sac", scenarios="left_turn", seeds="0"
        ),
        protocol,
    )[0]
    monkeypatch.setattr(
        reproduction,
        "command_for",
        lambda args, output_dir, selected_job: [
            "python",
            "check.py",
            "--max-steps",
            "100000",
            "--model-name",
            selected_job.name,
        ],
    )
    monkeypatch.setattr(
        reproduction.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["python", "check.py"], returncode=0, stdout="ok", stderr=""
        ),
    )
    args = argparse.Namespace(output_dir=tmp_path / "results", device="cpu")
    assert reproduction.preflight(args, [job]) == 0
    report = json.loads(
        (args.output_dir / "_preflight" / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["passed"] is True
    assert report["jobs_requested"] == 1
    assert report["unique_algorithm_scenario_pairs_checked"] == 1


def test_paired_bootstrap_preserves_zero_delta() -> None:
    arrays = [np.asarray([0.0, 1.0, 1.0])] * 3
    low, high = _bootstrap_paired_delta_interval(
        arrays,
        arrays,
        resamples=200,
        rng=np.random.default_rng(7),
    )
    assert low == 0.0
    assert high == 0.0


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    adjusted = _holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert 0.0 <= adjusted["a"] <= adjusted["b"] <= adjusted["c"] <= 1.0
    assert adjusted["a"] == 0.03


def test_effect_classification_respects_collision_guardrail() -> None:
    assert _effect_classification(
        {"success_rate_macro_delta": 0.1, "collision_rate_macro_delta": 0.0}
    ) == "directionally_positive_without_safety_tradeoff"
    assert _effect_classification(
        {"success_rate_macro_delta": 0.1, "collision_rate_macro_delta": 0.05}
    ) == "success_safety_tradeoff"


def test_incomplete_analysis_emits_tbd_without_fabrication(tmp_path) -> None:
    result_root = tmp_path / "results"
    output_dir = tmp_path / "analysis"
    summary, returncode = analyze(
        protocol_path=PROTOCOL_PATH,
        profile_name="paper",
        result_root=result_root,
        output_dir=output_dir,
        require_complete=True,
    )
    assert returncode == 2
    assert summary["matrix_complete"] is False
    assert summary["fabricated_values"] is False
    assert all(row["metrics"] == "TBD" for row in summary["cell_metrics"])
    attribution = json.loads(
        (output_dir / "attribution.json").read_text(encoding="utf-8")
    )
    assert attribution["evidence_bound"] is False
    assert (output_dir / "FINAL_REPORT.md").is_file()
