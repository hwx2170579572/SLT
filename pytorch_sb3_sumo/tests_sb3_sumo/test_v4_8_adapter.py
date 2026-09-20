from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

from configs import sb3_configs_v4_8 as config
from tools import train_sb3_v4_8 as train
from tools.checkpoint_decoder_selector_v4_6 import FUSION_DECODER, TARGET_DECODER
from tools.checkpoint_selector_v4_4 import sha256


def _checkpoint(path: Path) -> Path:
    with ZipFile(path, "w") as archive:
        archive.writestr("payload.txt", path.stem)
    return path


def _summary(success: int, collision: int) -> dict[str, float | int]:
    return {
        "episodes": 2,
        "mean_return": float(success - collision),
        "std_return": 0.0,
        "mean_decision_steps": 1.0,
        "mean_raw_steps": 3.0,
        "success_rate": success / 2,
        "collision_rate": collision / 2,
        "off_route_rate": 0.0,
        "timeout_rate": (2 - success - collision) / 2,
    }


def _records(seed: int, success: int, collision: int) -> list[dict[str, object]]:
    return [
        {
            "episode": episode,
            "seed": seed + episode,
            "traffic_variant": f"traffic_{episode}.rou.xml",
            "success": episode < success,
            "collision": success <= episode < success + collision,
            "off_route": False,
            "timeout": episode >= success + collision,
        }
        for episode in range(2)
    ]


def _candidate(
    checkpoint: Path,
    kind: str,
    decoder: str,
    summary: dict[str, float | int],
    *,
    seed: int = 100,
) -> dict[str, object]:
    return {
        "checkpoint_kind": kind,
        "deployment_decoder": decoder,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "traffic_partition": "train",
        "calibration_seed_start": seed,
        "summary": summary,
        "episode_records": _records(
            seed,
            round(float(summary["success_rate"]) * 2),
            round(float(summary["collision_rate"]) * 2),
        ),
        "calibration_result_sha256": ("a" if seed == 100 else "c") * 64,
        "calibration_action_diagnostics_sha256": ("b" if seed == 100 else "d") * 64,
        "decoder_integrity": {"selected_deployment_decoder": decoder},
        "decoder_integrity_passed": True,
    }


def _checkpoints(tmp_path: Path, summaries: list[dict[str, float | int]]) -> list[dict[str, object]]:
    best = _checkpoint(tmp_path / "best.zip")
    final = _checkpoint(tmp_path / "final.zip")
    pairs = [
        ("highest_training_success", TARGET_DECODER, best),
        ("highest_training_success", FUSION_DECODER, best),
        ("exact_final", TARGET_DECODER, final),
        ("exact_final", FUSION_DECODER, final),
    ]
    rows = [
        _candidate(path, kind, decoder, summaries[index])
        for index, (kind, decoder, path) in enumerate(pairs)
    ]
    return [
        {"checkpoint_kind": "highest_training_success", "deployment_candidates": rows[:2]},
        {"checkpoint_kind": "exact_final", "deployment_candidates": rows[2:]},
    ]


def test_v4_8_unique_primary_winner_does_not_construct_secondary(tmp_path: Path) -> None:
    checkpoints = _checkpoints(
        tmp_path,
        [_summary(1, 1), _summary(1, 1), _summary(2, 0), _summary(1, 1)],
    )
    old = train._ACTIVE_ALGORITHM
    train._ACTIVE_ALGORITHM = config.V48_CANDIDATE
    try:
        with mock.patch.object(train, "_evaluate_secondary_pair") as evaluate:
            receipt = train._select_deployment_adapter_v4_8(checkpoints)
    finally:
        train._ACTIVE_ALGORITHM = old
    evaluate.assert_not_called()
    assert receipt["secondary_calibration_triggered"] is False
    assert receipt["selected_checkpoint_kind"] == "exact_final"
    assert receipt["selected_deployment_decoder"] == TARGET_DECODER


def test_v4_8_adapter_evaluates_only_the_empirical_tied_top(tmp_path: Path) -> None:
    bad = _summary(1, 1)
    good = _summary(2, 0)
    checkpoints = _checkpoints(tmp_path, [bad, bad, good, good])
    initial = {
        (row["checkpoint_kind"], pair["deployment_decoder"]): pair
        for row in checkpoints
        for pair in row["deployment_candidates"]
    }

    def secondary(*, checkpoint_kind: str, deployment_decoder: str, context):
        primary = initial[(checkpoint_kind, deployment_decoder)]
        summary = good if deployment_decoder == FUSION_DECODER else bad
        row = _candidate(
            Path(str(primary["checkpoint_path"])),
            checkpoint_kind,
            deployment_decoder,
            summary,
            seed=200,
        )
        return row

    old_algorithm = train._ACTIVE_ALGORITHM
    old_contexts = dict(train._CONTEXTS)
    train._ACTIVE_ALGORITHM = config.V48_CANDIDATE
    train._CONTEXTS.clear()
    train._CONTEXTS.update({"exact_final": {"sentinel": True}})
    try:
        with mock.patch.object(train, "_evaluate_secondary_pair", side_effect=secondary) as evaluate:
            receipt = train._select_deployment_adapter_v4_8(checkpoints)
    finally:
        train._CONTEXTS.clear()
        train._CONTEXTS.update(old_contexts)
        train._ACTIVE_ALGORITHM = old_algorithm
    assert evaluate.call_count == 2
    assert {
        (call.kwargs["checkpoint_kind"], call.kwargs["deployment_decoder"])
        for call in evaluate.call_args_list
    } == {("exact_final", TARGET_DECODER), ("exact_final", FUSION_DECODER)}
    assert receipt["secondary_calibration_triggered"] is True
    assert receipt["selected_deployment_decoder"] == FUSION_DECODER


def test_v4_8_writer_records_secondary_seed_and_unchanged_training(tmp_path: Path) -> None:
    output = tmp_path / "arguments.json"
    train._write_json_v4_8(
        output,
        {
            "schema_version": "topo-scene-v4.7.run-arguments/v1",
            "implementation_fidelity": {},
            "requested_raw_steps": {
                "algo": config.V48_CANDIDATE,
                "calibration_seed_start": 77000,
            },
        },
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "topo-scene-v4.8.run-arguments/v1"
    assert payload["requested_raw_steps"]["secondary_calibration_seed_start"] == 77100
    assert payload["implementation_fidelity"]["single_change"] == "tie_only_replicated_train_calibration"
    assert payload["implementation_fidelity"]["training_changed_from_v4_7"] is False
    assert payload["implementation_fidelity"]["return_estimator_changed_from_v4_7"] is False


def test_v4_8_plain_writer_preserves_specialized_selector_schema(tmp_path: Path) -> None:
    output = tmp_path / "selector" / "receipt.json"
    train._write_json_plain_v4_8(
        output,
        {
            "schema_version": "topo-scene-v4.6.checkpoint-decoder-selector-receipt/v1",
            "selector_mode": train.SELECTOR_MODE,
        },
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == (
        "topo-scene-v4.8.tie-only-replicated-selector/v1"
    )


def test_v4_8_writer_versions_nested_return_diagnostics(tmp_path: Path) -> None:
    output = tmp_path / "training_diagnostics.json"
    diagnostics = {
        "schema_version": "topo-scene-v4.7.return-estimator-diagnostics/v1",
        "n_step": 4,
    }
    with mock.patch.object(
        train.frozen, "_return_estimator_diagnostics", return_value=diagnostics
    ):
        train._write_json_v4_8(
            output,
            {
                "schema_version": "topo-scene-v4.7.training-diagnostics/v1",
                "algorithm": config.TEMPORAL_GRAPH,
            },
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    standalone = json.loads(
        (tmp_path / "return_estimator_diagnostics.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "topo-scene-v4.8.training-diagnostics/v1"
    assert payload["return_estimator"]["schema_version"] == (
        "topo-scene-v4.8.return-estimator-diagnostics/v1"
    )
    assert payload["return_estimator"] == standalone


def test_v4_8_model_factory_reuses_v4_7_candidate_training() -> None:
    fake = mock.Mock()
    fake.v4_method_metadata = {"implementation_id": "v4.7-parent"}
    with mock.patch.object(config, "make_model_v4_7", return_value=fake) as make:
        result = config.make_model_v4_8(config.V48_CANDIDATE, object())
    assert result is fake
    make.assert_called_once()
    assert make.call_args.args[0] == config.V47_CANDIDATE
    assert result.v4_method_metadata["single_change"] == "tie_only_replicated_train_calibration"
    assert result.v4_method_metadata["training_changed_from_v4_7"] is False
