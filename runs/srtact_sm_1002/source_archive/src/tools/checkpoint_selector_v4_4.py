"""Train-only paired checkpoint selection for the v4.4 protocol."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile


CHECKPOINT_KINDS = ("highest_training_success", "exact_final")
SELECTION_ORDER = (
    "maximize_success_count",
    "minimize_collision_count",
    "minimize_off_route_count",
    "minimize_timeout_count",
    "maximize_mean_return",
    "prefer_exact_final_on_complete_tie",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def verify_checkpoint_zip(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with ZipFile(path, "r") as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise ValueError(f"checkpoint ZIP CRC failed at {bad_member}")


def _count(summary: dict[str, Any], rate_name: str) -> int:
    episodes = int(summary["episodes"])
    value = float(summary[rate_name]) * episodes
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        raise ValueError(f"{rate_name} does not imply an integral count: {value}")
    return int(rounded)


def outcome_counts(summary: dict[str, Any]) -> dict[str, int]:
    episodes = int(summary["episodes"])
    if episodes <= 0:
        raise ValueError("calibration must contain at least one episode")
    counts = {
        "episodes": episodes,
        "success_count": _count(summary, "success_rate"),
        "collision_count": _count(summary, "collision_rate"),
        "off_route_count": _count(summary, "off_route_rate"),
        "timeout_count": _count(summary, "timeout_rate"),
    }
    for key, value in counts.items():
        if key != "episodes" and not 0 <= value <= episodes:
            raise ValueError(f"{key} lies outside the episode population")
    if sum(value for key, value in counts.items() if key != "episodes") != episodes:
        raise ValueError("terminal outcome counts do not partition calibration episodes")
    mean_return = float(summary["mean_return"])
    if not math.isfinite(mean_return):
        raise ValueError("calibration mean return must be finite")
    return counts


def selection_key(
    summary: dict[str, Any], *, checkpoint_kind: str
) -> tuple[int, int, int, int, float, int]:
    if checkpoint_kind not in CHECKPOINT_KINDS:
        raise ValueError(f"unsupported checkpoint kind {checkpoint_kind!r}")
    counts = outcome_counts(summary)
    return (
        counts["success_count"],
        -counts["collision_count"],
        -counts["off_route_count"],
        -counts["timeout_count"],
        float(summary["mean_return"]),
        int(checkpoint_kind == "exact_final"),
    )


def _episode_pair_signature(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "episode": int(row["episode"]),
            "seed": int(row["seed"]),
            "traffic_variant": row.get("traffic_variant"),
        }
        for row in records
    ]


def select_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("checkpoint selector received no candidates")
    kinds = [str(row["checkpoint_kind"]) for row in candidates]
    if len(kinds) != len(set(kinds)) or any(kind not in CHECKPOINT_KINDS for kind in kinds):
        raise ValueError(f"invalid checkpoint candidate kinds: {kinds}")
    if "exact_final" not in kinds:
        raise ValueError("exact-final checkpoint is mandatory")

    reference = candidates[0]
    reference_signature = _episode_pair_signature(reference["episode_records"])
    reference_episodes = int(reference["summary"]["episodes"])
    reference_seed_start = int(reference["calibration_seed_start"])
    for candidate in candidates:
        path = Path(candidate["checkpoint_path"])
        verify_checkpoint_zip(path)
        if sha256(path) != str(candidate["checkpoint_sha256"]).lower():
            raise ValueError("checkpoint hash changed during selection")
        if candidate.get("traffic_partition") != "train":
            raise ValueError("checkpoint selection may only use the train partition")
        if int(candidate["summary"]["episodes"]) != reference_episodes:
            raise ValueError("checkpoint candidates used different episode counts")
        if int(candidate["calibration_seed_start"]) != reference_seed_start:
            raise ValueError("checkpoint candidates used different calibration seeds")
        if _episode_pair_signature(candidate["episode_records"]) != reference_signature:
            raise ValueError("checkpoint candidates used non-paired traffic episodes")

    ranked = sorted(
        candidates,
        key=lambda row: selection_key(
            row["summary"], checkpoint_kind=str(row["checkpoint_kind"])
        ),
        reverse=True,
    )
    selected = ranked[0]
    receipt_candidates = []
    for candidate in candidates:
        counts = outcome_counts(candidate["summary"])
        receipt_candidates.append(
            {
                "checkpoint_kind": candidate["checkpoint_kind"],
                "checkpoint_path": str(Path(candidate["checkpoint_path"]).resolve()),
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "calibration_seed_start": int(candidate["calibration_seed_start"]),
                "traffic_partition": candidate["traffic_partition"],
                "summary": candidate["summary"],
                "outcome_counts": counts,
                "selection_key": list(
                    selection_key(
                        candidate["summary"],
                        checkpoint_kind=str(candidate["checkpoint_kind"]),
                    )
                ),
                "episode_pair_signature": _episode_pair_signature(
                    candidate["episode_records"]
                ),
                "calibration_result_sha256": candidate[
                    "calibration_result_sha256"
                ],
            }
        )
    return {
        "schema_version": "topo-scene-v4.4.checkpoint-selector/v1",
        "selection_partition": "train",
        "paired_calibration": True,
        "calibration_episodes": reference_episodes,
        "calibration_seed_start": reference_seed_start,
        "selection_order": list(SELECTION_ORDER),
        "candidate_count": len(candidates),
        "candidates": receipt_candidates,
        "selected_checkpoint_kind": selected["checkpoint_kind"],
        "selected_checkpoint_path": str(
            Path(selected["checkpoint_path"]).resolve()
        ),
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "validation_used_for_selection": False,
        "formal_test_used_for_selection": False,
    }


__all__ = [
    "CHECKPOINT_KINDS",
    "SELECTION_ORDER",
    "outcome_counts",
    "select_checkpoint",
    "selection_key",
    "sha256",
    "verify_checkpoint_zip",
]
