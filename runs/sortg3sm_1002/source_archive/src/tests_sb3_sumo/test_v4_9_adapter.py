from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from configs import sb3_configs_v4_9 as config
from tools import train_sb3_v4_6 as deployment_parent
from tools import train_sb3_v4_8 as parent
from tools import train_sb3_v4_9 as train
from tools import action_diagnostics_v4_9_model as diagnostics
from tools.checkpoint_decoder_selector_v4_6 import PARENT_DECODER


def test_v4_9_writer_records_model_only_change(tmp_path: Path) -> None:
    output = tmp_path / "arguments.json"
    train._write_json_v4_9(
        output,
        {
            "schema_version": "topo-scene-v4.7.run-arguments/v1",
            "implementation_fidelity": {},
            "requested_raw_steps": {
                "algo": config.V49_CANDIDATE,
                "calibration_seed_start": 101000,
            },
        },
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    fidelity = payload["implementation_fidelity"]
    assert payload["schema_version"] == "topo-scene-v4.9.run-arguments/v1"
    assert fidelity["implementation_id"] == config.V49_IMPLEMENTATION_IDS[
        config.V49_CANDIDATE
    ]
    assert fidelity["single_change"] == (
        "learned_twin_collision_value_actor_constraint"
    )
    assert fidelity["training_changed_from_v4_8"] is True
    assert fidelity["return_estimator_changed_from_v4_8"] is False
    assert fidelity["learned_collision_critic"] is True
    assert fidelity["inference_safety_rule_added"] is False
    assert fidelity["selector_changed_from_v4_8"] is False
    assert fidelity["decoder_changed_from_v4_8"] is True
    assert fidelity["decoder_change_kind"] == (
        "learned_reward_minus_collision_value_scoring"
    )
    assert "v4_8_isolated_files" not in fidelity


def test_v4_9_plain_writer_versions_selector_receipt(tmp_path: Path) -> None:
    output = tmp_path / "selector" / "receipt.json"
    train._write_json_plain_v4_9(
        output,
        {
            "schema_version": "topo-scene-v4.6.checkpoint-decoder-selector-receipt/v1",
            "selector_mode": train.SELECTOR_MODE,
        },
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == (
        "topo-scene-v4.9.tie-only-replicated-selector/v1"
    )


def test_v4_9_temporal_graph_hyperparameters_do_not_recurse() -> None:
    payload = train._method_hyperparameters_v4_9(config.TEMPORAL_GRAPH)
    assert payload["single_change"] == "none_temporal_graph_control"
    assert payload["training_changed_from_v4_8"] is False
    assert payload["network_changed"] is False


def test_v4_9_formal_test_remains_locked_without_pass_receipt() -> None:
    args = type(
        "Args",
        (),
        {
            "evaluation_split": "test",
            "formal_unlock_receipt": None,
            "algo": config.V49_CANDIDATE,
            "experiment_contract_sha256": "contract",
            "implementation_freeze_sha256": "freeze",
        },
    )()
    with pytest.raises(ValueError, match="missing v4.9 promotion gate receipt"):
        train._validate_formal_unlock_v4_9(args)


def test_v4_9_runtime_patch_is_scoped_and_restored(monkeypatch) -> None:
    original_parent_candidate = parent.V48_CANDIDATE
    original_parent_method = parent._method_hyperparameters_v4_8
    original_loader = deployment_parent.load_model_for_deployment
    observed = {}

    def fake_main(argv, **kwargs):
        observed["candidate"] = parent.V48_CANDIDATE
        observed["method"] = parent._method_hyperparameters_v4_8(
            config.TEMPORAL_GRAPH
        )
        observed["loader"] = deployment_parent.load_model_for_deployment
        return 17

    monkeypatch.setattr(parent, "main", fake_main)
    assert train.main(["check", "--algo", config.V49_CANDIDATE]) == 17
    assert observed["candidate"] == config.V49_CANDIDATE
    assert observed["method"]["network_changed"] is False
    assert observed["loader"] is train.load_model_for_deployment_v4_9
    assert parent.V48_CANDIDATE == original_parent_candidate
    assert parent._method_hyperparameters_v4_8 is original_parent_method
    assert deployment_parent.load_model_for_deployment is original_loader


def test_v4_9_loader_preserves_temporal_graph_parent_path(tmp_path: Path) -> None:
    sentinel = object()
    with mock.patch.object(
        diagnostics,
        "load_parent_model_for_deployment",
        return_value=sentinel,
    ) as loader:
        restored = diagnostics.load_model_for_deployment_v4_9(
            object,
            tmp_path / "control.zip",
            decoder=PARENT_DECODER,
            env=None,
            device="cpu",
        )
    assert restored is sentinel
    loader.assert_called_once()
