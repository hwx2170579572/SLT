from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from configs.sb3_configs_v4_12 import (
    TEMPORAL_GRAPH,
    V412_AUGMENTATION_ONLY,
    V412_FULL,
    V412_NO_CROSS_AUGMENTATION,
)
from tools import train_sb3_v4_12 as subject
from tools import train_paper_sb3_sumo_v4_12 as paper_subject
from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER
from tools.checkpoint_decoder_selector_v4_9_2 import SELECTOR_MODE
from tools.train_paper_sb3_sumo_v4_5 import _make_paper_env_v4_5


def test_paper_wrapper_reuses_frozen_environment_factory(monkeypatch) -> None:
    captured = {}

    def fake_main(argv, **kwargs):
        captured.update(kwargs)
        return 17

    monkeypatch.setattr(paper_subject.train_sb3_v4_12, "main", fake_main)
    assert paper_subject.main(["--check-only"]) == 17
    assert captured["env_factory"] is _make_paper_env_v4_5
    assert captured["require_paper_evaluation_contract"] is True


@pytest.mark.parametrize(
    ("algorithm", "lane_scale", "lane_prior", "cross_augmentation"),
    [
        (V412_FULL, 1.0, 0.05, True),
        (V412_NO_CROSS_AUGMENTATION, 1.0, 0.05, False),
        (V412_AUGMENTATION_ONLY, 0.0, 0.0, True),
    ],
)
def test_candidate_metadata_records_preregistered_joint_support_variant(
    algorithm: str,
    lane_scale: float,
    lane_prior: float,
    cross_augmentation: bool,
) -> None:
    value = subject._method_hyperparameters_v4_12(algorithm)
    assert value["lane_support_scale"] == lane_scale
    assert value["lane_prior_coef"] == lane_prior
    assert value["cross_rotation_augmentation"] is cross_augmentation
    assert value["deployment_decoder_candidates"] == [TARGET_DECODER]
    assert value["selector_mode"] == SELECTOR_MODE
    assert value["replay_support_objective"] == (
        "masked_lane_categorical_nll_plus_conditional_speed_mixture_nll"
    )
    for key in (
        "inference_safety_rule_added",
        "external_kinematic_projection",
        "kinematic_safety_projection",
        "traffic_risk_in_lane_mask",
        "actor_confidence_gate",
        "action_postprocessing_override",
    ):
        assert value[key] is False


def test_version_payload_promotes_v411_schema_only() -> None:
    value = subject._version_payload(
        {"schema_version": "topo-scene-v4.11.run-arguments/v1"}
    )
    assert value["schema_version"] == "topo-scene-v4.12.run-arguments/v1"


def test_argument_and_receipt_rewrites_bind_v412_model_only_contract(
    tmp_path: Path,
) -> None:
    arguments = tmp_path / "arguments.json"
    arguments.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.11.run-arguments/v1",
                "requested_raw_steps": {"algo": V412_FULL},
                "implementation_fidelity": {
                    "v4_11_isolated_files": True,
                    "actor_non_keep_confidence_threshold": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )
    subject._rewrite_v4_12(arguments)
    fidelity = json.loads(arguments.read_text(encoding="utf-8"))[
        "implementation_fidelity"
    ]
    assert "v4_11_isolated_files" not in fidelity
    assert fidelity["v4_12_isolated_files"] is True
    assert fidelity["joint_replay_action_support"] is True
    assert fidelity["continuous_lane_log_prior"] is True
    assert fidelity["actor_confidence_threshold_present"] is False
    assert fidelity["kinematic_safety_projection"] is False

    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.11.target-only-tie-replicated-selector/v1",
                "selected_deployment_decoder": TARGET_DECODER,
            }
        ),
        encoding="utf-8",
    )
    subject._rewrite_v4_12(receipt)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["schema_version"] == (
        "topo-scene-v4.12.target-only-tie-replicated-selector/v1"
    )
    assert value["deployment_decoder_candidates"] == [TARGET_DECODER]
    assert value["actor_confidence_threshold_present"] is False
    assert value["kinematic_safety_projection"] is False


def test_formal_unlock_requires_passing_hash_bound_v412_receipt(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "promotion_gate.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.12.promotion-gate/v1",
                "decision": "pass",
                "experiment_contract_sha256": "contract",
                "implementation_freeze_sha256": "freeze",
                "formal_algorithms": [TEMPORAL_GRAPH, V412_FULL],
                "promotion_results_sha256": "results",
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        evaluation_split="test",
        algo=V412_FULL,
        experiment_contract_sha256="contract",
        implementation_freeze_sha256="freeze",
        formal_unlock_receipt=receipt,
    )
    value = subject._validate_formal_unlock_v4_12(args)
    assert value is not None
    assert value["promotion_results_sha256"] == "results"
    args.algo = V412_AUGMENTATION_ONLY
    with pytest.raises(ValueError, match="formal test permits"):
        subject._validate_formal_unlock_v4_12(args)


def test_main_installs_and_restores_v412_parent_interfaces(monkeypatch) -> None:
    observed = {}
    names = (
        "V411_ALGORITHMS",
        "V411_CANDIDATES",
        "V411_FORMAL_ALGORITHMS",
        "V411_FULL",
        "V411_IMPLEMENTATION_IDS",
        "make_model_v4_11",
        "source_action_repeat_v4_11",
        "_method_hyperparameters_v4_11",
        "_validate_formal_unlock_v4_11",
        "_write_json_v4_11",
        "_write_json_plain_v4_11",
        "load_model_for_deployment_v4_11",
        "evaluate_with_action_diagnostics_v4_11_model",
        "decoder_integrity_passed_v4_11",
        "decoder_integrity_summary_v4_11",
    )
    originals = {name: getattr(subject.parent, name) for name in names}

    def fake_main(*args, **kwargs):
        observed["full"] = subject.parent.V411_FULL
        observed["factory"] = subject.parent.make_model_v4_11
        observed["diagnostics"] = (
            subject.parent.evaluate_with_action_diagnostics_v4_11_model
        )
        return 23

    monkeypatch.setattr(subject.parent, "main", fake_main)
    assert subject.main(["--algo", V412_FULL]) == 23
    assert observed["full"] == V412_FULL
    assert observed["factory"] is subject.make_model_v4_12
    assert observed["diagnostics"] is subject.evaluate_with_action_diagnostics_v4_12_model
    for name, value in originals.items():
        assert getattr(subject.parent, name) is value
