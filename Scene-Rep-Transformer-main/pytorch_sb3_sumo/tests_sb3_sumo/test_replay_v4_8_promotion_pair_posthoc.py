from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import replay_v4_8_promotion_pair_posthoc as replay


def test_constants_bind_frozen_science_and_attribution_root() -> None:
    assert len(replay.EXPECTED_CONTRACT) == 64
    assert len(replay.EXPECTED_FREEZE) == 64
    assert replay.EXPECTED_ALGORITHM.startswith("topo_v4_8")
    assert "attribution" in replay.ATTRIBUTION_ROOT.parts
    assert "closed_loop" in replay.ATTRIBUTION_ROOT.parts


def test_parent_parser_forbids_test_partition() -> None:
    choices = next(
        action.choices
        for action in replay.parent.parser()._actions
        if action.dest == "traffic_partition"
    )
    assert tuple(choices) == ("train", "validation")
    assert "test" not in choices


def test_validate_source_rejects_output_outside_attribution(monkeypatch, tmp_path) -> None:
    run = tmp_path / "run"
    (run / "selector").mkdir(parents=True)
    model = run / "final_model.zip"
    model.write_bytes(b"model")
    arguments = {
        "experiment_contract_sha256": replay.EXPECTED_CONTRACT,
        "implementation_freeze_sha256": replay.EXPECTED_FREEZE,
        "requested_raw_steps": {
            "algo": replay.EXPECTED_ALGORITHM,
            "evaluation_split": "validation",
            "evaluation_seed_start": 10,
            "eval_episodes": 30,
        },
    }
    diagnostics = {"n_step": 16, "bootstrap_discount": "gamma_power_actual_horizon"}
    selector = {
        "selected_source_checkpoint_path": str(model.resolve()),
        "selected_checkpoint_kind": "exact_final",
    }
    (run / "arguments.json").write_text(__import__("json").dumps(arguments), encoding="utf-8")
    (run / "return_estimator_diagnostics.json").write_text(__import__("json").dumps(diagnostics), encoding="utf-8")
    (run / "selector" / "receipt.json").write_text(__import__("json").dumps(selector), encoding="utf-8")
    monkeypatch.setattr(replay, "DEVELOPMENT_DECISION", run / "arguments.json")
    args = SimpleNamespace(
        run_dir=run,
        output_dir=tmp_path / "outside",
        model_file="final_model.zip",
        checkpoint_sha256=replay._sha256(model),
        checkpoint_kind="exact_final",
        traffic_partition="validation",
        evaluation_seed_start=10,
        episodes=30,
        experiment_contract_sha256=replay.EXPECTED_CONTRACT,
        implementation_freeze_sha256=replay.EXPECTED_FREEZE,
        development_decision_sha256=replay._sha256(run / "arguments.json"),
    )
    with pytest.raises(ValueError, match="attribution root"):
        replay.validate_source(args)
