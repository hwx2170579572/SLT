from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.checkpoint_decoder_selector_v4_6 import FUSION_DECODER, TARGET_DECODER
from tools.checkpoint_decoder_selector_v4_8 import (
    SELECTOR_MODE,
    select_deployment,
    tied_top_pairs,
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


def _records(success: int, collision: int, timeout: int, *, seed: int):
    rows = []
    for episode in range(2):
        rows.append(
            {
                "episode": episode,
                "seed": seed + episode,
                "traffic_variant": f"traffic_{episode}.rou.xml",
                "success": episode < success,
                "collision": success <= episode < success + collision,
                "off_route": False,
                "timeout": episode >= 2 - timeout,
            }
        )
    return rows


def _checkpoint(root: Path, name: str) -> Path:
    path = root / f"{name}.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("payload.txt", name)
    return path


def _candidate(
    path: Path,
    checkpoint: str,
    decoder: str,
    summary,
    records,
    *,
    seed: int,
):
    return {
        "checkpoint_kind": checkpoint,
        "deployment_decoder": decoder,
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256(path),
        "traffic_partition": "train",
        "calibration_seed_start": seed,
        "summary": summary,
        "episode_records": records,
        "calibration_result_sha256": ("a" if seed == 100 else "c") * 64,
        "calibration_action_diagnostics_sha256": ("b" if seed == 100 else "d") * 64,
        "decoder_integrity": {"selected_deployment_decoder": decoder},
        "decoder_integrity_passed": True,
    }


def _matrix(root: Path, summaries, *, seed: int):
    best = _checkpoint(root, "best") if not (root / "best.zip").exists() else root / "best.zip"
    final = _checkpoint(root, "final") if not (root / "final.zip").exists() else root / "final.zip"
    pairs = [
        ("highest_training_success", TARGET_DECODER, best),
        ("highest_training_success", FUSION_DECODER, best),
        ("exact_final", TARGET_DECODER, final),
        ("exact_final", FUSION_DECODER, final),
    ]
    return [
        _candidate(
            path,
            checkpoint,
            decoder,
            summaries[index],
            _records(
                round(summaries[index]["success_rate"] * 2),
                round(summaries[index]["collision_rate"] * 2),
                round(summaries[index]["timeout_rate"] * 2),
                seed=seed,
            ),
            seed=seed,
        )
        for index, (checkpoint, decoder, path) in enumerate(pairs)
    ]


def test_unique_empirical_winner_forbids_secondary() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        initial = _matrix(
            root,
            [_summary(1, 1, 0), _summary(1, 1, 0), _summary(2, 0, 0), _summary(1, 1, 0)],
            seed=100,
        )
        receipt = select_deployment(initial)
        with pytest.raises(ValueError, match="forbidden"):
            select_deployment(initial, [copy.deepcopy(initial[2])])
    assert receipt["selector_mode"] == SELECTOR_MODE
    assert receipt["secondary_calibration_triggered"] is False
    assert receipt["selected_checkpoint_kind"] == "exact_final"
    assert receipt["selected_deployment_decoder"] == TARGET_DECODER


def test_tie_runs_only_tied_set_and_combined_evidence_selects_fusion() -> None:
    tied = _summary(1, 1, 0)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        initial = _matrix(root, [tied, tied, tied, tied], seed=100)
        secondary = _matrix(
            root,
            [tied, tied, tied, _summary(2, 0, 0)],
            seed=200,
        )
        assert len(tied_top_pairs(initial)) == 4
        receipt = select_deployment(initial, secondary)
    assert receipt["secondary_calibration_triggered"] is True
    assert receipt["initial_empirical_tied_top_count"] == 4
    assert len(receipt["secondary_candidates"]) == 4
    assert receipt["selected_checkpoint_kind"] == "exact_final"
    assert receipt["selected_deployment_decoder"] == FUSION_DECODER
    assert receipt["combined_tied_candidates"]["exact_final__fusion_0_90"]["success_count"] == 3
    assert receipt["validation_used_for_selection"] is False


def test_secondary_subset_must_exactly_match_top_tie() -> None:
    good = _summary(2, 0, 0)
    bad = _summary(1, 1, 0)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        initial = _matrix(root, [bad, bad, good, good], seed=100)
        secondary = _matrix(root, [bad, bad, good, good], seed=200)
        tied_secondary = secondary[2:]
        receipt = select_deployment(initial, tied_secondary)
        with pytest.raises(ValueError, match="equal"):
            select_deployment(initial, secondary)
    assert receipt["initial_empirical_tied_top_pairs"] == [
        "exact_final__target_critic",
        "exact_final__fusion_0_90",
    ]


def test_secondary_rejects_seed_overlap_variant_drift_and_unpaired_rows() -> None:
    tied = _summary(1, 1, 0)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        initial = _matrix(root, [tied] * 4, seed=100)
        overlap = _matrix(root, [tied] * 4, seed=100)
        with pytest.raises(ValueError, match="offset"):
            select_deployment(initial, overlap)
        secondary = _matrix(root, [tied] * 4, seed=200)
        secondary[0]["episode_records"][0]["traffic_variant"] = "different.xml"
        with pytest.raises(ValueError, match="variant support|not paired"):
            select_deployment(initial, secondary)


def test_still_tied_retains_parent_exact_final_target_fallback() -> None:
    tied = _summary(1, 1, 0)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        initial = _matrix(root, [tied] * 4, seed=100)
        secondary = _matrix(root, [tied] * 4, seed=200)
        receipt = select_deployment(initial, secondary)
    assert receipt["parent_v4_7_counterfactual_selected_pair"] == "exact_final__target_critic"
    assert receipt["selected_checkpoint_kind"] == "exact_final"
    assert receipt["selected_deployment_decoder"] == TARGET_DECODER
