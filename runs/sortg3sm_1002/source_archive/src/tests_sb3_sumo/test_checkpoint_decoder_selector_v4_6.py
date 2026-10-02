from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.checkpoint_decoder_selector_v4_6 import (
    FUSION_DECODER,
    PARENT_DECODER,
    TARGET_DECODER,
    deployment_selection_key,
    select_deployment,
)
from tools.checkpoint_selector_v4_4 import sha256


def _summary(success: int, collision: int, timeout: int, *, episodes: int = 2):
    return {
        "episodes": episodes,
        "mean_return": (success - collision) / episodes,
        "std_return": 0.0,
        "mean_decision_steps": 1.0,
        "mean_raw_steps": 3.0,
        "success_rate": success / episodes,
        "collision_rate": collision / episodes,
        "off_route_rate": 0.0,
        "timeout_rate": timeout / episodes,
    }


def _records(success: int, collision: int, timeout: int, *, overlap: bool = False):
    rows = []
    for episode in range(2):
        rows.append(
            {
                "episode": episode,
                "seed": 100 + episode,
                "traffic_variant": f"traffic_{episode}.rou.xml",
                "success": episode < success,
                "collision": success <= episode < success + collision,
                "off_route": False,
                "timeout": episode >= 2 - timeout,
            }
        )
    if overlap:
        rows[0]["success"] = True
        rows[0]["timeout"] = True
    return rows


def _checkpoint(root: Path, name: str) -> Path:
    path = root / f"{name}.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("payload.txt", name)
    return path


def _candidate(path: Path, checkpoint: str, decoder: str, summary, records):
    return {
        "checkpoint_kind": checkpoint,
        "deployment_decoder": decoder,
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256(path),
        "traffic_partition": "train",
        "calibration_seed_start": 100,
        "summary": summary,
        "episode_records": records,
        "calibration_result_sha256": "a" * 64,
        "calibration_action_diagnostics_sha256": "b" * 64,
        "decoder_integrity": {"selected_deployment_decoder": decoder},
        "decoder_integrity_passed": True,
    }


def test_joint_selector_uses_outcomes_then_exact_final_then_target() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        best = _checkpoint(root, "best")
        final = _checkpoint(root, "final")
        target = _summary(2, 0, 0)
        fusion = _summary(1, 1, 0)
        candidates = [
            _candidate(best, "highest_training_success", TARGET_DECODER, target, _records(2, 0, 0)),
            _candidate(best, "highest_training_success", FUSION_DECODER, fusion, _records(1, 1, 0)),
            _candidate(final, "exact_final", TARGET_DECODER, target, _records(2, 0, 0)),
            _candidate(final, "exact_final", FUSION_DECODER, fusion, _records(1, 1, 0)),
        ]
        receipt = select_deployment(candidates)
    assert receipt["selector_mode"] == "joint_checkpoint_decoder"
    assert receipt["candidate_count"] == 4
    assert receipt["selected_checkpoint_kind"] == "exact_final"
    assert receipt["selected_deployment_decoder"] == TARGET_DECODER
    assert receipt["validation_used_for_selection"] is False


def test_target_is_only_final_tie_break_after_checkpoint_preference() -> None:
    summary = _summary(2, 0, 0)
    target_key = deployment_selection_key(
        summary, checkpoint_kind="exact_final", deployment_decoder=TARGET_DECODER
    )
    fusion_key = deployment_selection_key(
        summary, checkpoint_kind="exact_final", deployment_decoder=FUSION_DECODER
    )
    best_target_key = deployment_selection_key(
        summary,
        checkpoint_kind="highest_training_success",
        deployment_decoder=TARGET_DECODER,
    )
    assert target_key > fusion_key > best_target_key


def test_selector_preserves_overlapping_source_flags() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        best = _checkpoint(root, "best")
        final = _checkpoint(root, "final")
        overlapping_summary = _summary(1, 0, 2)
        overlapping_records = _records(1, 0, 2, overlap=True)
        candidates = [
            _candidate(path, checkpoint, decoder, overlapping_summary, overlapping_records)
            for checkpoint, path in (
                ("highest_training_success", best),
                ("exact_final", final),
            )
            for decoder in (TARGET_DECODER, FUSION_DECODER)
        ]
        receipt = select_deployment(candidates)
    assert all(
        row["terminal_event_evidence"]["overlap_episode_count"] == 1
        for row in receipt["candidates"]
    )
    assert receipt["event_precedence_added"] is False


def test_selector_rejects_incomplete_or_nonpaired_matrix() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        best = _checkpoint(root, "best")
        final = _checkpoint(root, "final")
        summary = _summary(2, 0, 0)
        candidates = [
            _candidate(path, checkpoint, decoder, summary, _records(2, 0, 0))
            for checkpoint, path in (
                ("highest_training_success", best),
                ("exact_final", final),
            )
            for decoder in (TARGET_DECODER, FUSION_DECODER)
        ]
        with pytest.raises(ValueError, match="complete v4.6"):
            select_deployment(candidates[:-1])
        tampered = copy.deepcopy(candidates)
        tampered[-1]["episode_records"][1]["traffic_variant"] = "different.xml"
        with pytest.raises(ValueError, match="non-paired"):
            select_deployment(tampered)


def test_parent_control_uses_checkpoint_only_selector() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        best = _checkpoint(root, "best")
        final = _checkpoint(root, "final")
        candidates = [
            _candidate(
                best,
                "highest_training_success",
                PARENT_DECODER,
                _summary(1, 1, 0),
                _records(1, 1, 0),
            ),
            _candidate(
                final,
                "exact_final",
                PARENT_DECODER,
                _summary(2, 0, 0),
                _records(2, 0, 0),
            ),
        ]
        receipt = select_deployment(candidates)
    assert receipt["selector_mode"] == "checkpoint_only_parent_control"
    assert receipt["selected_checkpoint_kind"] == "exact_final"
    assert receipt["selected_deployment_decoder"] == PARENT_DECODER

