"""v4.5.1 train-only selector for source event flags that may overlap.

The source-compatible environment can report arrival and max-time on the same
raw simulator tick.  This selector preserves those raw flags and the v4.5
selection tuple; it removes only the invalid assumption that event-flag counts
must sum to the number of episodes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

from tools.checkpoint_selector_v4_4 import (
    CHECKPOINT_KINDS,
    SELECTION_ORDER,
    sha256,
    verify_checkpoint_zip,
)


EVENT_FIELDS = ("success", "collision", "off_route", "timeout")
RATE_FIELDS = tuple(f"{name}_rate" for name in EVENT_FIELDS)


def _count(summary: dict[str, Any], rate_name: str) -> int:
    episodes = int(summary["episodes"])
    value = float(summary[rate_name]) * episodes
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        raise ValueError(f"{rate_name} does not imply an integral count: {value}")
    return int(rounded)


def event_flag_counts(summary: dict[str, Any]) -> dict[str, int]:
    """Return bounded integral event counts without imposing exclusivity."""

    episodes = int(summary["episodes"])
    if episodes <= 0:
        raise ValueError("calibration must contain at least one episode")
    counts = {
        "episodes": episodes,
        **{
            f"{event}_count": _count(summary, rate)
            for event, rate in zip(EVENT_FIELDS, RATE_FIELDS)
        },
    }
    for key, value in counts.items():
        if key != "episodes" and not 0 <= value <= episodes:
            raise ValueError(f"{key} lies outside the episode population")
    mean_return = float(summary["mean_return"])
    if not math.isfinite(mean_return):
        raise ValueError("calibration mean return must be finite")
    return counts


def _episode_pair_signature(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "episode": int(row["episode"]),
            "seed": int(row["seed"]),
            "traffic_variant": row.get("traffic_variant"),
        }
        for row in records
    ]


def terminal_event_evidence(
    summary: dict[str, Any], records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Cross-check summary counts against immutable per-episode event flags."""

    rows = list(records)
    counts = event_flag_counts(summary)
    episodes = counts["episodes"]
    if len(rows) != episodes:
        raise ValueError(
            f"episode record count does not match summary: {len(rows)} != {episodes}"
        )
    episode_ids = [int(row["episode"]) for row in rows]
    if len(set(episode_ids)) != episodes:
        raise ValueError("calibration episode ids are duplicated")

    observed = {event: 0 for event in EVENT_FIELDS}
    overlaps: list[dict[str, Any]] = []
    for row in rows:
        active = [event for event in EVENT_FIELDS if bool(row.get(event, False))]
        if not active:
            raise ValueError(
                f"episode {int(row['episode'])} has no terminal event flag"
            )
        for event in active:
            observed[event] += 1
        if len(active) > 1:
            overlaps.append(
                {
                    "episode": int(row["episode"]),
                    "seed": int(row["seed"]),
                    "traffic_variant": row.get("traffic_variant"),
                    "active_flags": active,
                }
            )

    for event in EVENT_FIELDS:
        expected = counts[f"{event}_count"]
        if observed[event] != expected:
            raise ValueError(
                f"{event} summary count disagrees with episode records: "
                f"{expected} != {observed[event]}"
            )
    return {
        "event_flag_semantics": "source_flags_may_overlap",
        "every_episode_has_terminal_event_flag": True,
        "event_flag_counts": counts,
        "event_flag_memberships": sum(observed.values()),
        "overlap_episode_count": len(overlaps),
        "overlap_episodes": overlaps,
    }


def selection_key(
    summary: dict[str, Any], *, checkpoint_kind: str
) -> tuple[int, int, int, int, float, int]:
    if checkpoint_kind not in CHECKPOINT_KINDS:
        raise ValueError(f"unsupported checkpoint kind {checkpoint_kind!r}")
    counts = event_flag_counts(summary)
    return (
        counts["success_count"],
        -counts["collision_count"],
        -counts["off_route_count"],
        -counts["timeout_count"],
        float(summary["mean_return"]),
        int(checkpoint_kind == "exact_final"),
    )


def select_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("checkpoint selector received no candidates")
    kinds = [str(row["checkpoint_kind"]) for row in candidates]
    if len(kinds) != len(set(kinds)) or any(
        kind not in CHECKPOINT_KINDS for kind in kinds
    ):
        raise ValueError(f"invalid checkpoint candidate kinds: {kinds}")
    if "exact_final" not in kinds:
        raise ValueError("exact-final checkpoint is mandatory")

    reference = candidates[0]
    reference_signature = _episode_pair_signature(reference["episode_records"])
    reference_episodes = int(reference["summary"]["episodes"])
    reference_seed_start = int(reference["calibration_seed_start"])
    evidence: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        kind = str(candidate["checkpoint_kind"])
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
        signature = _episode_pair_signature(candidate["episode_records"])
        if signature != reference_signature:
            raise ValueError("checkpoint candidates used non-paired traffic episodes")
        evidence[kind] = terminal_event_evidence(
            candidate["summary"], candidate["episode_records"]
        )

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
        kind = str(candidate["checkpoint_kind"])
        counts = event_flag_counts(candidate["summary"])
        receipt_candidates.append(
            {
                "checkpoint_kind": kind,
                "checkpoint_path": str(Path(candidate["checkpoint_path"]).resolve()),
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "calibration_seed_start": int(candidate["calibration_seed_start"]),
                "traffic_partition": candidate["traffic_partition"],
                "summary": candidate["summary"],
                "outcome_counts": counts,
                "terminal_event_evidence": evidence[kind],
                "selection_key": list(
                    selection_key(
                        candidate["summary"], checkpoint_kind=kind
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
        "schema_version": "topo-scene-v4.5.1.checkpoint-selector/v1",
        "engineering_patch": "v4.5.1_overlapping_terminal_event_flags",
        "selection_partition": "train",
        "paired_calibration": True,
        "calibration_episodes": reference_episodes,
        "calibration_seed_start": reference_seed_start,
        "selection_order": list(SELECTION_ORDER),
        "terminal_event_semantics": "source_flags_may_overlap",
        "summary_and_episode_records_preserved": True,
        "event_precedence_added": False,
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
    "EVENT_FIELDS",
    "event_flag_counts",
    "select_checkpoint",
    "selection_key",
    "terminal_event_evidence",
]
