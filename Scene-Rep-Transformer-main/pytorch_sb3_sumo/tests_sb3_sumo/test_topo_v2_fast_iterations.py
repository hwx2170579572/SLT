from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools import run_topo_v2_experiments as full
from tools import run_topo_v2_fast_iterations as fast


def _context():
    return fast.load_protocol(fast.DEFAULT_PROTOCOL)


def test_fast_protocol_is_hash_bound_to_frozen_base_contract():
    protocol, effective, digest, base_path = _context()

    assert len(digest) == 64
    assert fast._sha256(base_path) == protocol["base_contract_sha256"]
    assert Path(effective["methods"]["selected_candidate"]["resolver"]).as_posix().endswith(
        "results_topo_v2_fast/development/candidate_selection.json"
    )
    assert full.frozen_dependency_hashes(effective) == {
        "freeze_manifest_sha256": "5081576f7ee92d50478131c6be78878f47f42058029d2b5543c72702668bf850",
        "topology_audit_sha256": "94b571e49cc0363b55c959437aa0097d1c457f0c49c4343e8d3230cf5c629ad4",
        "runtime_environment_sha256": "c947ab6848f9ea5db09d1afce374d8cb3d9a70f906418ee87d5b3750868321c3",
    }


def test_fast_development_budget_and_job_matrix_are_exact():
    protocol, effective, digest, _ = _context()
    counts = {
        stage: len(fast.jobs_for_stage(protocol, effective, digest, stage))
        for stage in (
            "factorial_anchor",
            "merge_probe",
            "query_probe",
            "gated_probe",
            "soft_center",
            "soft_bracket",
        )
    }

    assert counts == {
        "factorial_anchor": 8,
        "merge_probe": 3,
        "query_probe": 3,
        "gated_probe": 3,
        "soft_center": 2,
        "soft_bracket": 4,
    }
    jobs = [
        job
        for stage in counts
        for job in fast.jobs_for_stage(protocol, effective, digest, stage)
    ]
    assert len({job.name for job in jobs}) == len(jobs)
    assert {job.raw_steps for job in jobs} == {20_000}
    assert {job.evaluation_episodes for job in jobs} == {12}
    assert {job.evaluation_split for job in jobs} == {"validation"}


def test_promotion_and_formal_declarations_stay_locked_and_unchanged(tmp_path):
    protocol, effective, digest, _ = _context()
    promotion = protocol["stages"]["promotion"]
    formal = protocol["stages"]["formal_test"]

    assert len(promotion["methods"]) * len(promotion["scenarios"]) * len(
        promotion["seeds"]
    ) == 12
    assert promotion["raw_training_steps"] == 50_000
    assert promotion["evaluation_split"] == "validation"
    assert formal["methods"] == ["temporal_graph", "selected_candidate"]
    assert tuple(formal["scenarios"]) == full.SCENARIOS
    assert formal["seeds"] == list(range(10))
    assert formal["raw_training_steps"] == 100_000
    assert formal["final_evaluation_episodes"] == 50
    assert formal["evaluation_split"] == "test"
    # This is a negative pre-freeze contract test.  Point it at an isolated
    # missing receipt so the assertion remains valid after a real candidate
    # has been frozen in the shared project workspace.
    effective["methods"]["selected_candidate"]["resolver"] = str(
        tmp_path / "missing_candidate_selection.json"
    )
    with pytest.raises(full.ExperimentContractError, match="Selected candidate is not frozen"):
        fast.jobs_for_stage(protocol, effective, digest, "formal_test")


def _row(method: str, success: float, collision: float, compatible: float):
    return {
        "scenario": "cross",
        "seed": 0,
        "method": method,
        "success_rate": success,
        "collision_rate": collision,
        "mean_return": success - collision,
        "off_route_rate": 0.0,
        "timeout_rate": 1.0 - success - collision,
        "lane_command_negative_rate": 0.2,
        "lane_command_keep_rate": 0.6,
        "lane_command_positive_rate": 0.2,
        "lane_command_threshold_margin": 0.1,
        "lane_change_applied_rate": 0.3,
        "mean_speed_mps": 8.0,
        "mechanism": {"slot_rms_ratio": 1.1},
        "evaluation_mechanism": {
            "route_compatible_attention_mass": compatible,
            "topology_fallback_rate": 0.05,
            "topology_effective_lanes": 4.0,
        },
    }


def test_paired_attribution_and_mechanism_gate_use_real_fields():
    baseline = _row("v3_query", 0.5, 0.3, 0.8)
    candidate = _row("v4_gated", 0.75, 0.1, 0.9)
    contrast = fast.paired_contrast([baseline, candidate], "v4_gated", "v3_query")
    rules = {
        "compatible_attention_mass_min": 0.75,
        "topology_fallback_rate_max": 0.15,
        "topology_effective_lanes_max": 8.0,
    }
    mechanism = fast._mechanism_gate([candidate], rules)

    assert contrast["paired_cells"] == 1
    assert math.isclose(contrast["macro_delta"]["success_rate"], 0.25)
    assert math.isclose(contrast["macro_delta"]["collision_rate"], -0.2)
    assert mechanism["passed"] is True
    assert fast._balance_score([candidate]) == pytest.approx(abs(math.log(1.1)))


def test_fast_summary_preserves_legacy_slot_and_graph_diagnostics():
    original = {"mechanism": {"slot_rms_ratio": 1.1}, "method": "topology_v1"}
    diagnostics = {
        "diagnostic/ego_latent_std": {"mean": 0.6},
        "diagnostic/social_latent_std": {"mean": 0.5},
        "diagnostic/route_latent_std": {"mean": 0.3},
        "diagnostic/slot_scale_ratio": {"mean": 2.0},
        "diagnostic/graph_mean_edge_weight": {"mean": 0.25},
    }

    enriched = fast._enrich_legacy_mechanism(original, diagnostics)

    assert original["mechanism"] == {"slot_rms_ratio": 1.1}
    assert enriched["mechanism"] == {
        "slot_rms_ratio": 1.1,
        "ego_latent_std": 0.6,
        "social_latent_std": 0.5,
        "route_latent_std": 0.3,
        "slot_scale_ratio": 2.0,
        "graph_mean_edge_weight": 0.25,
    }
