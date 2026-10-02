from __future__ import annotations

from pathlib import Path

import pytest

from tools import train_sb3_v4_6 as deployment_parent
from tools import train_sb3_v4_7_1 as patch


def _selector() -> dict[str, str]:
    return {
        "selected_deployment_decoder": "target_critic",
        "selected_checkpoint_sha256": "a" * 64,
        "selected_model_policy_class": "TargetCriticDecisionAlignedSACPolicyV43",
        "selected_model_parameter_state_sha256": "b" * 64,
    }


def test_v4_7_1_binds_only_sealed_selector_fields(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def writer(path, payload):
        captured["path"] = path
        captured["payload"] = payload

    monkeypatch.setattr(deployment_parent, "_SELECTED_DEPLOYMENT", _selector())
    monkeypatch.setattr(patch, "_ORIGINAL_WRITER", writer)
    destination = tmp_path / "paper_evaluation_detailed.json"
    source = {"schema_version": "topo-scene-v4.7.detailed-evaluation/v1"}
    patch._write_json_v4_7_1(destination, source)
    value = captured["payload"]
    assert source == {"schema_version": "topo-scene-v4.7.detailed-evaluation/v1"}
    assert captured["path"] == destination
    assert value["selected_deployment_decoder"] == "target_critic"
    assert value["selected_source_checkpoint_sha256"] == "a" * 64
    assert value["selected_model_parameter_state_sha256"] == "b" * 64
    assert value["engineering_patch_id"] == patch.PATCH_ID
    assert value["scientific_protocol_changed_by_engineering_patch"] is False


def test_v4_7_1_refuses_unsealed_or_conflicting_binding(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "paper_evaluation_detailed.json"
    monkeypatch.setattr(deployment_parent, "_SELECTED_DEPLOYMENT", None)
    with pytest.raises(RuntimeError, match="not sealed"):
        patch._write_json_v4_7_1(path, {})

    monkeypatch.setattr(deployment_parent, "_SELECTED_DEPLOYMENT", _selector())
    with pytest.raises(ValueError, match="conflicts"):
        patch._write_json_v4_7_1(
            path, {"selected_deployment_decoder": "fusion_0_90"}
        )


def test_v4_7_1_passes_non_detailed_artifacts_through(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_WRITER",
        lambda path, payload: captured.update(path=path, payload=payload),
    )
    source = {"schema_version": "topo-scene-v4.7.training-diagnostics/v1"}
    path = tmp_path / "training_diagnostics.json"
    patch._write_json_v4_7_1(path, source)
    assert captured == {"path": path, "payload": source}
