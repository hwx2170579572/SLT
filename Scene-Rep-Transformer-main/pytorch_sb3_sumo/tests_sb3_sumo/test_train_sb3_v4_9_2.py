from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from configs.sb3_configs_v4_9 import TEMPORAL_GRAPH, V49_CANDIDATE
from tools import train_sb3_v4_7, train_sb3_v4_8, train_sb3_v4_9
from tools import train_sb3_v4_9_2 as subject
from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER
from tools.checkpoint_decoder_selector_v4_9_2 import SELECTOR_MODE


def test_candidate_metadata_is_strict_target_only():
    value = subject._method_hyperparameters_v4_9_2(V49_CANDIDATE)
    assert value["deployment_decoder_candidates"] == [TARGET_DECODER]
    assert value["checkpoint_decoder_candidate_pairs"] == 2
    assert value["selector_mode"] == SELECTOR_MODE
    assert value["fusion_candidate_present"] is False
    assert value["actor_confidence_threshold_present"] is False
    assert "actor_non_keep_confidence_threshold" not in value
    assert value["external_kinematic_projection"] is False
    assert value["action_postprocessing_override"] is False
    assert value["scientific_model_changed_from_v4_9"] is False


def test_control_metadata_keeps_parent_decoder():
    value = subject._method_hyperparameters_v4_9_2(TEMPORAL_GRAPH)
    assert value["deployment_decoder_candidates"] == ["parent_control"]
    assert value["checkpoint_decoder_candidate_pairs"] == 2
    assert value["selector_changed_from_v4_9"] is False
    assert value["fusion_candidate_present"] is False


def test_version_rewrite_removes_threshold_and_records_no_rule(tmp_path: Path):
    path = tmp_path / "arguments.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.9.run-arguments/v1",
                "requested_raw_steps": {"algo": V49_CANDIDATE},
                "implementation_fidelity": {
                    "actor_non_keep_confidence_threshold": 0.9
                },
            }
        ),
        encoding="utf-8",
    )
    subject._rewrite_v4_9_2(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema_version"] == "topo-scene-v4.9.2.run-arguments/v1"
    fidelity = value["implementation_fidelity"]
    assert "actor_non_keep_confidence_threshold" not in fidelity
    assert fidelity["deployment_decoder_candidates"] == [TARGET_DECODER]
    assert fidelity["checkpoint_decoder_candidate_pairs"] == 2
    assert fidelity["fusion_candidate_present"] is False
    assert fidelity["external_kinematic_projection"] is False


def test_receipt_rewrite_corrects_target_only_calibration_metadata(tmp_path: Path):
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.9.tie-only-replicated-selector/v1",
                "selected_deployment_decoder": TARGET_DECODER,
                "calibration_deterministic_lane_decoder": "parent_control",
            }
        ),
        encoding="utf-8",
    )
    subject._rewrite_v4_9_2(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["calibration_deterministic_lane_decoder"] == TARGET_DECODER
    assert value["calibration_uses_method_own_frozen_decoder"] is True
    assert value["fusion_candidate_present"] is False


def test_formal_unlock_is_locked_without_v492_receipt():
    args = argparse.Namespace(
        evaluation_split="test",
        algo=V49_CANDIDATE,
        experiment_contract_sha256="a" * 64,
        implementation_freeze_sha256="b" * 64,
        formal_unlock_receipt=None,
    )
    with pytest.raises(ValueError, match="missing v4.9.2 promotion"):
        subject._validate_formal_unlock_v4_9_2(args)


def test_main_restores_all_parent_monkeypatches(monkeypatch):
    observed = {}

    def fake_parent_main(*args, **kwargs):
        observed["v47_decoders"] = train_sb3_v4_7.CANDIDATE_DECODERS
        observed["v48_decoders"] = train_sb3_v4_8.CANDIDATE_DECODERS
        observed["v48_mode"] = train_sb3_v4_8.SELECTOR_MODE
        observed["v49_mode"] = train_sb3_v4_9.SELECTOR_MODE
        observed["v48_tied"] = train_sb3_v4_8.tied_top_pairs
        return 17

    originals = {
        "v47_decoders": train_sb3_v4_7.CANDIDATE_DECODERS,
        "v48_decoders": train_sb3_v4_8.CANDIDATE_DECODERS,
        "v48_mode": train_sb3_v4_8.SELECTOR_MODE,
        "v49_mode": train_sb3_v4_9.SELECTOR_MODE,
        "v48_tied": train_sb3_v4_8.tied_top_pairs,
    }
    monkeypatch.setattr(train_sb3_v4_9, "main", fake_parent_main)
    assert subject.main([]) == 17
    assert observed["v47_decoders"] == (TARGET_DECODER,)
    assert observed["v48_decoders"] == (TARGET_DECODER,)
    assert observed["v48_mode"] == SELECTOR_MODE
    assert observed["v49_mode"] == SELECTOR_MODE
    assert observed["v48_tied"] is subject.tied_top_pairs
    assert train_sb3_v4_7.CANDIDATE_DECODERS == originals["v47_decoders"]
    assert train_sb3_v4_8.CANDIDATE_DECODERS == originals["v48_decoders"]
    assert train_sb3_v4_8.SELECTOR_MODE == originals["v48_mode"]
    assert train_sb3_v4_9.SELECTOR_MODE == originals["v49_mode"]
    assert train_sb3_v4_8.tied_top_pairs is originals["v48_tied"]
