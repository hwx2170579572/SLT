from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER
from tools.checkpoint_decoder_selector_v4_9_2 import (
    SELECTOR_MODE,
    select_deployment,
    tied_top_pairs,
)
from tools.checkpoint_selector_v4_4 import sha256


def _summary(*, success: int, collision: int, mean_return: float) -> dict:
    episodes = 2
    return {
        "episodes": episodes,
        "success_rate": success / episodes,
        "collision_rate": collision / episodes,
        "off_route_rate": 0.0,
        "timeout_rate": (episodes - success - collision) / episodes,
        "mean_return": mean_return,
    }


def _records(seed: int, summary: dict) -> list[dict]:
    success_count = round(summary["success_rate"] * summary["episodes"])
    collision_count = round(summary["collision_rate"] * summary["episodes"])
    rows = []
    for index in range(summary["episodes"]):
        rows.append(
            {
                "episode": index,
                "seed": seed + index,
                "traffic_variant": f"v{index}",
                "success": index < success_count,
                "collision": success_count <= index < success_count + collision_count,
                "off_route": False,
                "timeout": index >= success_count + collision_count,
            }
        )
    return rows


def _candidate(
    checkpoint: str,
    path: Path,
    summary: dict,
    *,
    seed: int,
) -> dict:
    return {
        "checkpoint_kind": checkpoint,
        "deployment_decoder": TARGET_DECODER,
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256(path),
        "traffic_partition": "train",
        "calibration_seed_start": seed,
        "summary": summary,
        "episode_records": _records(seed, summary),
        "calibration_result_sha256": "a" * 64,
        "calibration_action_diagnostics_sha256": "b" * 64,
        "decoder_integrity": {"passed": True},
        "decoder_integrity_passed": True,
    }


@pytest.fixture
def checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    import zipfile

    paths = (tmp_path / "best.zip", tmp_path / "final.zip")
    for path in paths:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data", "{}")
            archive.writestr("policy.pth", b"weights")
            archive.writestr("pytorch_variables.pth", b"variables")
            archive.writestr("_stable_baselines3_version", "2")
            archive.writestr("system_info.txt", "test")
    return paths


def test_target_only_selector_rejects_non_target_candidate(checkpoints):
    best, final = checkpoints
    rows = [
        _candidate("highest_training_success", best, _summary(success=2, collision=0, mean_return=1.0), seed=10),
        _candidate("exact_final", final, _summary(success=1, collision=1, mean_return=0.0), seed=10),
    ]
    rows[1]["deployment_decoder"] = "fusion_0_90"
    with pytest.raises(ValueError, match="exactly the two"):
        tied_top_pairs(rows)


def test_target_only_selector_uses_primary_empirical_winner(checkpoints):
    best, final = checkpoints
    rows = [
        _candidate("highest_training_success", best, _summary(success=2, collision=0, mean_return=1.0), seed=10),
        _candidate("exact_final", final, _summary(success=1, collision=1, mean_return=0.0), seed=10),
    ]
    receipt = select_deployment(rows)
    assert receipt["selector_mode"] == SELECTOR_MODE
    assert receipt["candidate_count"] == 2
    assert receipt["deployment_decoder_candidates"] == [TARGET_DECODER]
    assert receipt["selected_checkpoint_kind"] == "highest_training_success"
    assert receipt["selected_deployment_decoder"] == TARGET_DECODER
    assert receipt["secondary_calibration_triggered"] is False
    assert receipt["fusion_candidate_present"] is False


def test_target_only_selector_replicates_tie_and_prefers_exact_final(checkpoints):
    best, final = checkpoints
    rows = [
        _candidate("highest_training_success", best, _summary(success=2, collision=0, mean_return=1.0), seed=10),
        _candidate("exact_final", final, _summary(success=2, collision=0, mean_return=1.0), seed=10),
    ]
    secondary = []
    for row in rows:
        copied = copy.deepcopy(row)
        copied["calibration_seed_start"] = 110
        copied["episode_records"] = _records(110, copied["summary"])
        copied["calibration_result_sha256"] = "c" * 64
        secondary.append(copied)
    receipt = select_deployment(rows, secondary, secondary_seed_offset=100)
    assert receipt["secondary_calibration_triggered"] is True
    assert receipt["secondary_calibration_seed_start"] == 110
    assert receipt["selected_checkpoint_kind"] == "exact_final"
    assert receipt["selected_deployment_decoder"] == TARGET_DECODER


def test_target_only_selector_rejects_unpaired_primary(checkpoints):
    best, final = checkpoints
    rows = [
        _candidate("highest_training_success", best, _summary(success=2, collision=0, mean_return=1.0), seed=10),
        _candidate("exact_final", final, _summary(success=2, collision=0, mean_return=1.0), seed=10),
    ]
    rows[1]["episode_records"][1]["traffic_variant"] = "drift"
    with pytest.raises(ValueError, match="not paired"):
        select_deployment(rows)
