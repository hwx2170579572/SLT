from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from configs.sb3_configs_v4_13 import (
    TEMPORAL_GRAPH,
    V413_FULL,
    V413_ISOLATED_UNTEMPERED,
    V413_NO_CROSS_AUGMENTATION,
    V413_UNISOLATED_TEMPERED,
)
from tools import train_paper_sb3_sumo_v4_13 as paper_subject
from tools import train_sb3_v4_13 as subject
from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER
from tools.train_paper_sb3_sumo_v4_5 import _make_paper_env_v4_5


def test_paper_wrapper_reuses_frozen_environment_factory(monkeypatch) -> None:
    captured = {}

    def fake_main(argv, **kwargs):
        captured.update(kwargs)
        return 17

    monkeypatch.setattr(paper_subject.train_sb3_v4_13, "main", fake_main)
    assert paper_subject.main(["--check-only"]) == 17
    assert captured["env_factory"] is _make_paper_env_v4_5
    assert captured["require_paper_evaluation_contract"] is True


@pytest.mark.parametrize(
    ("algorithm", "isolated", "scale", "cross_augmentation"),
    [
        (V413_FULL, True, 0.25, True),
        (V413_UNISOLATED_TEMPERED, False, 0.25, True),
        (V413_ISOLATED_UNTEMPERED, True, 1.0, True),
        (V413_NO_CROSS_AUGMENTATION, True, 0.25, False),
    ],
)
def test_candidate_metadata_records_preregistered_variant(
    algorithm: str,
    isolated: bool,
    scale: float,
    cross_augmentation: bool,
) -> None:
    value = subject._method_hyperparameters_v4_13(algorithm)
    assert value["lane_support_gradient_isolated"] is isolated
    assert value["lane_support_scale"] == scale
    assert value["cross_rotation_augmentation"] is cross_augmentation
    assert value["same_deployed_lane_head_for_support"] is True
    assert value["new_inference_head"] is False
    assert value["inference_score_equation_changed_from_v4_12"] is False
    assert value["deployment_decoder_candidates"] == [TARGET_DECODER]
    for key in (
        "inference_safety_rule_added",
        "external_kinematic_projection",
        "kinematic_safety_projection",
        "traffic_risk_in_lane_mask",
        "ttc_or_headway_threshold",
        "lane_change_veto",
        "actor_confidence_gate",
        "action_postprocessing_override",
        "scenario_conditioned_inference_rule",
    ):
        assert value[key] is False


def test_argument_and_receipt_rewrites_bind_v413_contract(
    tmp_path: Path,
) -> None:
    arguments = tmp_path / "arguments.json"
    arguments.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.12.run-arguments/v1",
                "requested_raw_steps": {"algo": V413_FULL},
                "implementation_fidelity": {"v4_12_isolated_files": True},
            }
        ),
        encoding="utf-8",
    )
    subject._rewrite_v4_13(arguments)
    value = json.loads(arguments.read_text(encoding="utf-8"))
    fidelity = value["implementation_fidelity"]
    assert value["schema_version"] == "topo-scene-v4.13.run-arguments/v1"
    assert "v4_12_isolated_files" not in fidelity
    assert fidelity["v4_13_isolated_files"] is True
    assert fidelity["same_deployed_lane_head_for_support"] is True
    assert fidelity["new_inference_head"] is False
    assert fidelity["kinematic_safety_projection"] is False

    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": (
                    "topo-scene-v4.12.target-only-tie-replicated-selector/v1"
                ),
                "selected_deployment_decoder": TARGET_DECODER,
            }
        ),
        encoding="utf-8",
    )
    subject._rewrite_v4_13(receipt)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["schema_version"] == (
        "topo-scene-v4.13.target-only-tie-replicated-selector/v1"
    )
    assert value["kinematic_safety_projection"] is False


def test_formal_unlock_requires_passing_hash_bound_v413_receipt(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "promotion_gate.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.13.promotion-gate/v1",
                "decision": "pass",
                "experiment_contract_sha256": "contract",
                "implementation_freeze_sha256": "freeze",
                "formal_algorithms": [TEMPORAL_GRAPH, V413_FULL],
                "promotion_results_sha256": "results",
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        evaluation_split="test",
        algo=V413_FULL,
        experiment_contract_sha256="contract",
        implementation_freeze_sha256="freeze",
        formal_unlock_receipt=receipt,
    )
    value = subject._validate_formal_unlock_v4_13(args)
    assert value is not None
    assert value["promotion_results_sha256"] == "results"
    args.algo = V413_UNISOLATED_TEMPERED
    with pytest.raises(ValueError, match="formal test permits"):
        subject._validate_formal_unlock_v4_13(args)


def test_main_installs_and_restores_v413_parent_interfaces(monkeypatch) -> None:
    observed = {}
    names = (
        "V412_ALGORITHMS",
        "V412_CANDIDATES",
        "V412_FORMAL_ALGORITHMS",
        "V412_FULL",
        "V412_IMPLEMENTATION_IDS",
        "make_model_v4_12",
        "source_action_repeat_v4_12",
        "_method_hyperparameters_v4_12",
        "_validate_formal_unlock_v4_12",
        "_write_json_v4_12",
        "_write_json_plain_v4_12",
        "load_model_for_deployment_v4_12",
        "evaluate_with_action_diagnostics_v4_12_model",
        "decoder_integrity_passed_v4_12",
        "decoder_integrity_summary_v4_12",
    )
    originals = {name: getattr(subject.parent, name) for name in names}

    def fake_main(*args, **kwargs):
        observed["full"] = subject.parent.V412_FULL
        observed["factory"] = subject.parent.make_model_v4_12
        return 23

    monkeypatch.setattr(subject.parent, "main", fake_main)
    assert subject.main(["--algo", V413_FULL]) == 23
    assert observed["full"] == V413_FULL
    assert observed["factory"] is subject.make_model_v4_13
    for name, value in originals.items():
        assert getattr(subject.parent, name) is value
