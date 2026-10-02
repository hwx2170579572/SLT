from __future__ import annotations

import json
from pathlib import Path

from configs.sb3_configs_v4_11 import TEMPORAL_GRAPH, V411_FULL
from tools import train_sb3_v4_7, train_sb3_v4_8, train_sb3_v4_9
from tools import train_sb3_v4_11 as subject
from tools import train_paper_sb3_sumo_v4_11 as paper_subject
from tools.train_paper_sb3_sumo_v4_5 import _make_paper_env_v4_5
from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER
from tools.checkpoint_decoder_selector_v4_9_2 import (
    SELECTOR_MODE,
    TARGET_ONLY_DECODERS,
    select_deployment,
    tied_top_pairs,
)


def test_paper_wrapper_uses_existing_frozen_environment_factory(monkeypatch):
    captured = {}

    def fake_main(argv, **kwargs):
        captured.update(kwargs)
        return 17

    monkeypatch.setattr(paper_subject.train_sb3_v4_11, "main", fake_main)
    assert paper_subject.main(["--check-only"]) == 17
    assert captured["env_factory"] is _make_paper_env_v4_5
    assert captured["require_paper_evaluation_contract"] is True


def test_candidate_metadata_uses_target_only_learned_selector():
    value = subject._method_hyperparameters_v4_11(V411_FULL)
    assert value["deployment_decoder_candidates"] == [TARGET_DECODER]
    assert value["checkpoint_decoder_candidate_pairs"] == 2
    assert value["selector_mode"] == SELECTOR_MODE
    assert value["fixed_speed_grid_at_inference"] is False
    assert value["external_kinematic_projection"] is False
    assert value["action_postprocessing_override"] is False
    assert value["collision_return_loss"] == (
        "soft_target_binary_cross_entropy_with_logits"
    )
    assert value["collision_pairwise_hard_threshold"] is False
    assert value["collision_pairwise_fixed_margin"] is False


def test_version_payload_drops_v492_patch_segment():
    value = subject._version_payload(
        {
            "schema_version": (
                "topo-scene-v4.9.2.target-only-tie-replicated-selector/v1"
            )
        }
    )
    assert value["schema_version"] == (
        "topo-scene-v4.11.target-only-tie-replicated-selector/v1"
    )


def test_receipt_rewrite_records_target_only_model_deployment(tmp_path: Path):
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "topo-scene-v4.9.2.target-only-tie-replicated-selector/v1"
                ),
                "selected_deployment_decoder": TARGET_DECODER,
                "calibration_deterministic_lane_decoder": "parent_control",
            }
        ),
        encoding="utf-8",
    )
    subject._rewrite_v4_11(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema_version"] == (
        "topo-scene-v4.11.target-only-tie-replicated-selector/v1"
    )
    assert value["selector_mode"] == SELECTOR_MODE
    assert value["candidate_count"] == 2
    assert value["deployment_decoder_candidates"] == [TARGET_DECODER]
    assert value["calibration_deterministic_lane_decoder"] == TARGET_DECODER
    assert value["external_kinematic_projection"] is False
    assert value["action_postprocessing_override"] is False


def test_check_result_rewrite_records_exact_hybrid_lane(tmp_path: Path):
    path = tmp_path / "check_result.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.9.check-result/v1",
                "algorithm": V411_FULL,
                "action": [0.25, -1.0],
                "exact_hybrid_lane_code": None,
            }
        ),
        encoding="utf-8",
    )
    subject._rewrite_v4_11(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["exact_hybrid_lane_code"] is True


def test_argument_rewrite_removes_inherited_confidence_threshold(tmp_path: Path):
    path = tmp_path / "arguments.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.9.run-arguments/v1",
                "requested_raw_steps": {"algo": V411_FULL},
                "implementation_fidelity": {
                    "actor_non_keep_confidence_threshold": 0.9
                },
            }
        ),
        encoding="utf-8",
    )
    subject._rewrite_v4_11(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    fidelity = value["implementation_fidelity"]
    assert "actor_non_keep_confidence_threshold" not in fidelity
    assert fidelity["external_kinematic_projection"] is False
    assert fidelity["action_postprocessing_override"] is False
    assert fidelity["kinematic_safety_projection"] is False
    assert fidelity["traffic_risk_in_lane_mask"] is False
    assert fidelity["collision_pairwise_hard_threshold"] is False


def test_control_artifacts_seal_formal_access_and_no_projection(tmp_path: Path):
    metadata_path = tmp_path / "method_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.9.method-metadata/v1",
                "algorithm": TEMPORAL_GRAPH,
                "network_changed": False,
            }
        ),
        encoding="utf-8",
    )
    subject._rewrite_v4_11(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["network_changed"] is False
    assert metadata["inference_safety_rule_added"] is False
    assert metadata["external_kinematic_projection"] is False
    assert metadata["action_postprocessing_override"] is False

    detailed_path = tmp_path / "paper_evaluation_detailed.json"
    detailed_path.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.9.detailed-evaluation/v1",
                "algorithm": TEMPORAL_GRAPH,
                "evaluation_split": "validation",
            }
        ),
        encoding="utf-8",
    )
    subject._rewrite_v4_11(detailed_path)
    detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
    assert detailed["formal_test_accessed"] is False
    assert detailed["external_kinematic_projection"] is False
    assert detailed["action_postprocessing_override"] is False

    actions_path = tmp_path / "action_diagnostics.json"
    actions_path.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.9.action-diagnostics/v1",
                "algorithm": TEMPORAL_GRAPH,
            }
        ),
        encoding="utf-8",
    )
    subject._rewrite_v4_11(actions_path)
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    assert actions["learned_collision_critic_present"] is False
    assert actions["shared_risk_encoder_present"] is False
    assert actions["fixed_speed_grid_at_inference"] is False
    assert actions["inference_safety_rule_added"] is False
    assert actions["external_kinematic_projection"] is False
    assert actions["action_postprocessing_override"] is False


def test_main_installs_and_restores_target_only_selector(monkeypatch):
    observed = {}

    def fake_parent_main(*args, **kwargs):
        observed.update(
            {
                "v47_decoders": train_sb3_v4_7.CANDIDATE_DECODERS,
                "v48_decoders": train_sb3_v4_8.CANDIDATE_DECODERS,
                "v48_mode": train_sb3_v4_8.SELECTOR_MODE,
                "v48_tied": train_sb3_v4_8.tied_top_pairs,
                "v48_select": train_sb3_v4_8.select_deployment,
                "v49_decoders": train_sb3_v4_9.CANDIDATE_DECODERS,
                "v49_mode": train_sb3_v4_9.SELECTOR_MODE,
            }
        )
        return 23

    originals = {
        "v47_decoders": train_sb3_v4_7.CANDIDATE_DECODERS,
        "v48_decoders": train_sb3_v4_8.CANDIDATE_DECODERS,
        "v48_mode": train_sb3_v4_8.SELECTOR_MODE,
        "v48_tied": train_sb3_v4_8.tied_top_pairs,
        "v48_select": train_sb3_v4_8.select_deployment,
        "v49_decoders": train_sb3_v4_9.CANDIDATE_DECODERS,
        "v49_mode": train_sb3_v4_9.SELECTOR_MODE,
    }
    monkeypatch.setattr(train_sb3_v4_9, "main", fake_parent_main)
    assert subject.main(["--algo", V411_FULL]) == 23
    assert observed["v47_decoders"] == TARGET_ONLY_DECODERS
    assert observed["v48_decoders"] == TARGET_ONLY_DECODERS
    assert observed["v48_mode"] == SELECTOR_MODE
    assert observed["v48_tied"] is tied_top_pairs
    assert observed["v48_select"] is select_deployment
    assert observed["v49_decoders"] == TARGET_ONLY_DECODERS
    assert observed["v49_mode"] == SELECTOR_MODE
    assert train_sb3_v4_7.CANDIDATE_DECODERS == originals["v47_decoders"]
    assert train_sb3_v4_8.CANDIDATE_DECODERS == originals["v48_decoders"]
    assert train_sb3_v4_8.SELECTOR_MODE == originals["v48_mode"]
    assert train_sb3_v4_8.tied_top_pairs is originals["v48_tied"]
    assert train_sb3_v4_8.select_deployment is originals["v48_select"]
    assert train_sb3_v4_9.CANDIDATE_DECODERS == originals["v49_decoders"]
    assert train_sb3_v4_9.SELECTOR_MODE == originals["v49_mode"]
