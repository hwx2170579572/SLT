"""Train-only checkpoint selection for the strict target-only v4.9.2 protocol.

The selector never changes or post-processes an action.  It compares two
preregistered checkpoints using only real train-partition rollouts produced by
the learned ``target_critic`` deployment policy.  A second paired train block
is evaluated only when the primary empirical outcome key ties.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER
from tools.checkpoint_selector_v4_4 import (
    CHECKPOINT_KINDS,
    sha256,
    verify_checkpoint_zip,
)
from tools.checkpoint_selector_v4_5_1 import (
    event_flag_counts,
    terminal_event_evidence,
)


TARGET_ONLY_DECODERS = (TARGET_DECODER,)
EMPIRICAL_SELECTION_ORDER = (
    "maximize_success_count",
    "minimize_collision_count",
    "minimize_off_route_count",
    "minimize_timeout_count",
    "maximize_mean_return",
)
FINAL_FALLBACK_ORDER = ("prefer_exact_final_if_still_tied",)
SELECTOR_MODE = "tie_only_replicated_target_only_checkpoint_selector"


def _pair(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["checkpoint_kind"]), str(row["deployment_decoder"])


def _pair_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__{pair[1]}"


def _signature(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "episode": int(row["episode"]),
            "seed": int(row["seed"]),
            "traffic_variant": row.get("traffic_variant"),
        }
        for row in records
    ]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def empirical_selection_key(
    summary: dict[str, Any],
) -> tuple[int, int, int, int, float]:
    counts = event_flag_counts(summary)
    return (
        counts["success_count"],
        -counts["collision_count"],
        -counts["off_route_count"],
        -counts["timeout_count"],
        float(summary["mean_return"]),
    )


def final_fallback_key(pair: tuple[str, str]) -> tuple[int]:
    checkpoint, decoder = pair
    if decoder != TARGET_DECODER:
        raise ValueError("v4.9.2 fallback received a non-target decoder")
    return (int(checkpoint == "exact_final"),)


def _validate_primary(
    candidates: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    if not candidates:
        raise ValueError("target-only selector received no candidates")
    observed = [_pair(row) for row in candidates]
    expected = {(checkpoint, TARGET_DECODER) for checkpoint in CHECKPOINT_KINDS}
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(
            "v4.9.2 candidates must be exactly the two checkpoint x "
            "target_critic pairs"
        )

    reference = candidates[0]
    reference_signature = _signature(reference["episode_records"])
    reference_episodes = int(reference["summary"]["episodes"])
    reference_seed = int(reference["calibration_seed_start"])
    if reference_episodes <= 0:
        raise ValueError("primary calibration must contain episodes")
    if str(reference["traffic_partition"]) != "train":
        raise ValueError("deployment selection may only use train traffic")

    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidates:
        pair = _pair(row)
        checkpoint, decoder = pair
        if checkpoint not in CHECKPOINT_KINDS or decoder != TARGET_DECODER:
            raise ValueError(f"unsupported target-only pair {pair}")
        path = Path(row["checkpoint_path"])
        verify_checkpoint_zip(path)
        digest = str(row["checkpoint_sha256"]).lower()
        if sha256(path) != digest:
            raise ValueError("primary checkpoint hash changed")
        if str(row["traffic_partition"]) != "train":
            raise ValueError("primary calibration may only use train traffic")
        if int(row["summary"]["episodes"]) != reference_episodes:
            raise ValueError("primary candidates used different episode counts")
        if int(row["calibration_seed_start"]) != reference_seed:
            raise ValueError("primary candidates used different calibration seeds")
        if _signature(row["episode_records"]) != reference_signature:
            raise ValueError("primary candidates were not paired")
        if row.get("decoder_integrity_passed") is not True:
            raise ValueError(f"primary decoder integrity failed for {pair}")
        by_pair[pair] = row
    return by_pair


def tied_top_pairs(
    initial_candidates: list[dict[str, Any]],
) -> tuple[tuple[str, str], ...]:
    """Validate the two target candidates and return empirical first ties."""

    _validate_primary(initial_candidates)
    best = max(
        empirical_selection_key(row["summary"]) for row in initial_candidates
    )
    return tuple(
        _pair(row)
        for row in initial_candidates
        if empirical_selection_key(row["summary"]) == best
    )


def _validate_secondary(
    initial_candidates: list[dict[str, Any]],
    secondary_candidates: list[dict[str, Any]],
    tied: tuple[tuple[str, str], ...],
    *,
    secondary_seed_offset: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    initial_by_pair = _validate_primary(initial_candidates)
    observed = [_pair(row) for row in secondary_candidates]
    if len(observed) != len(set(observed)) or set(observed) != set(tied):
        raise ValueError(
            "secondary candidates must equal the target-only empirical tied top"
        )
    reference = secondary_candidates[0]
    reference_signature = _signature(reference["episode_records"])
    reference_seed = int(reference["calibration_seed_start"])
    reference_episodes = int(reference["summary"]["episodes"])
    if reference_episodes <= 0:
        raise ValueError("secondary calibration must contain episodes")
    if str(reference["traffic_partition"]) != "train":
        raise ValueError("secondary calibration may only use train traffic")
    primary_seed = int(initial_candidates[0]["calibration_seed_start"])
    if reference_seed != primary_seed + int(secondary_seed_offset):
        raise ValueError("secondary calibration seed does not match frozen offset")
    primary_signature = _signature(initial_candidates[0]["episode_records"])
    if {row["seed"] for row in primary_signature} & {
        row["seed"] for row in reference_signature
    }:
        raise ValueError("primary and secondary calibration seeds overlap")
    if [row["traffic_variant"] for row in primary_signature] != [
        row["traffic_variant"] for row in reference_signature
    ]:
        raise ValueError("secondary calibration did not replicate variant support")

    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in secondary_candidates:
        pair = _pair(row)
        checkpoint, decoder = pair
        if checkpoint not in CHECKPOINT_KINDS or decoder != TARGET_DECODER:
            raise ValueError(f"unsupported secondary pair {pair}")
        path = Path(row["checkpoint_path"])
        verify_checkpoint_zip(path)
        digest = str(row["checkpoint_sha256"]).lower()
        if sha256(path) != digest:
            raise ValueError("secondary checkpoint hash changed")
        primary = initial_by_pair[pair]
        if str(Path(primary["checkpoint_path"]).resolve()) != str(path.resolve()):
            raise ValueError("secondary calibration used a different checkpoint")
        if str(primary["checkpoint_sha256"]).lower() != digest:
            raise ValueError("secondary calibration used different weights")
        if str(row["traffic_partition"]) != "train":
            raise ValueError("secondary calibration may only use train traffic")
        if int(row["summary"]["episodes"]) != reference_episodes:
            raise ValueError("secondary candidates used different episode counts")
        if int(row["calibration_seed_start"]) != reference_seed:
            raise ValueError("secondary candidates used different seeds")
        if _signature(row["episode_records"]) != reference_signature:
            raise ValueError("secondary candidates were not paired")
        if row.get("decoder_integrity_passed") is not True:
            raise ValueError(f"secondary decoder integrity failed for {pair}")
        by_pair[pair] = row
    return by_pair


def _combined_evidence(
    primary: dict[str, Any], secondary: dict[str, Any]
) -> dict[str, Any]:
    left = event_flag_counts(primary["summary"])
    right = event_flag_counts(secondary["summary"])
    episodes = left["episodes"] + right["episodes"]
    counts = {
        "episodes": episodes,
        "success_count": left["success_count"] + right["success_count"],
        "collision_count": left["collision_count"] + right["collision_count"],
        "off_route_count": left["off_route_count"] + right["off_route_count"],
        "timeout_count": left["timeout_count"] + right["timeout_count"],
    }
    mean_return = (
        float(primary["summary"]["mean_return"]) * left["episodes"]
        + float(secondary["summary"]["mean_return"]) * right["episodes"]
    ) / episodes
    empirical_key = (
        counts["success_count"],
        -counts["collision_count"],
        -counts["off_route_count"],
        -counts["timeout_count"],
        mean_return,
    )
    return {
        **counts,
        "mean_return": mean_return,
        "empirical_selection_key": list(empirical_key),
        "primary_calibration_result_sha256": primary[
            "calibration_result_sha256"
        ],
        "secondary_calibration_result_sha256": secondary[
            "calibration_result_sha256"
        ],
        "combined_calibration_evidence_sha256": _canonical_sha256(
            {
                "primary": primary["calibration_result_sha256"],
                "secondary": secondary["calibration_result_sha256"],
                "counts": counts,
                "mean_return": mean_return,
            }
        ),
    }


def _receipt_candidate(row: dict[str, Any]) -> dict[str, Any]:
    checkpoint, decoder = _pair(row)
    return {
        "checkpoint_kind": checkpoint,
        "deployment_decoder": decoder,
        "checkpoint_path": str(Path(row["checkpoint_path"]).resolve()),
        "checkpoint_sha256": str(row["checkpoint_sha256"]).lower(),
        "calibration_seed_start": int(row["calibration_seed_start"]),
        "traffic_partition": str(row["traffic_partition"]),
        "summary": row["summary"],
        "outcome_counts": event_flag_counts(row["summary"]),
        "terminal_event_evidence": terminal_event_evidence(
            row["summary"], row["episode_records"]
        ),
        "empirical_selection_key": list(
            empirical_selection_key(row["summary"])
        ),
        "episode_pair_signature": _signature(row["episode_records"]),
        "calibration_result_sha256": row["calibration_result_sha256"],
        "calibration_action_diagnostics_sha256": row[
            "calibration_action_diagnostics_sha256"
        ],
        "decoder_integrity": row["decoder_integrity"],
        "decoder_integrity_passed": True,
    }


def select_deployment(
    initial_candidates: list[dict[str, Any]],
    secondary_candidates: list[dict[str, Any]] | None = None,
    *,
    secondary_seed_offset: int = 100,
) -> dict[str, Any]:
    """Select a target-only checkpoint with tie-triggered train replication."""

    initial_by_pair = _validate_primary(initial_candidates)
    tied = tied_top_pairs(initial_candidates)
    secondary_triggered = len(tied) > 1
    combined: dict[tuple[str, str], dict[str, Any]] = {}
    secondary_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    if not secondary_triggered:
        if secondary_candidates:
            raise ValueError("secondary calibration is forbidden without a tie")
        selected_pair = tied[0]
    else:
        if not secondary_candidates:
            raise ValueError("secondary calibration is required for tied checkpoints")
        secondary_by_pair = _validate_secondary(
            initial_candidates,
            secondary_candidates,
            tied,
            secondary_seed_offset=secondary_seed_offset,
        )
        combined = {
            pair: _combined_evidence(
                initial_by_pair[pair], secondary_by_pair[pair]
            )
            for pair in tied
        }
        selected_pair = max(
            tied,
            key=lambda pair: (
                tuple(combined[pair]["empirical_selection_key"]),
                final_fallback_key(pair),
            ),
        )

    selected = initial_by_pair[selected_pair]
    receipt: dict[str, Any] = {
        "schema_version": (
            "topo-scene-v4.9.2.target-only-tie-replicated-selector/v1"
        ),
        "selection_partition": "train",
        "selector_mode": SELECTOR_MODE,
        "paired_calibration": True,
        "identical_unique_traffic_block_for_all_pairs": True,
        "initial_calibration_episodes": int(
            initial_candidates[0]["summary"]["episodes"]
        ),
        "initial_calibration_seed_start": int(
            initial_candidates[0]["calibration_seed_start"]
        ),
        "selection_order": [
            *EMPIRICAL_SELECTION_ORDER,
            "if_empirical_target_checkpoints_tied_run_second_paired_train_seed_block",
            "rerank_tied_target_checkpoints_by_combined_empirical_fields",
            *FINAL_FALLBACK_ORDER,
        ],
        "terminal_event_semantics": "source_flags_may_overlap",
        "summary_and_episode_records_preserved": True,
        "event_precedence_added": False,
        "candidate_count": len(initial_candidates),
        "checkpoint_candidates": list(CHECKPOINT_KINDS),
        "deployment_decoder_candidates": [TARGET_DECODER],
        "fusion_candidate_present": False,
        "actor_confidence_threshold_present": False,
        "external_kinematic_projection": False,
        "action_postprocessing_override": False,
        "candidates": [_receipt_candidate(row) for row in initial_candidates],
        "initial_empirical_tied_top_count": len(tied),
        "initial_empirical_tied_top_pairs": [_pair_name(pair) for pair in tied],
        "secondary_calibration_triggered": secondary_triggered,
        "secondary_calibration_seed_offset": int(secondary_seed_offset),
        "secondary_candidate_scope": "empirical_tied_target_checkpoints_only",
        "non_top_candidate_reentry_forbidden": True,
        "selected_checkpoint_kind": selected_pair[0],
        "selected_checkpoint_path": str(
            Path(selected["checkpoint_path"]).resolve()
        ),
        "selected_checkpoint_sha256": str(
            selected["checkpoint_sha256"]
        ).lower(),
        "selected_deployment_decoder": TARGET_DECODER,
        "validation_used_for_selection": False,
        "formal_test_used_for_selection": False,
    }
    if secondary_triggered:
        reference = secondary_candidates[0]
        receipt.update(
            {
                "secondary_calibration_episodes": int(
                    reference["summary"]["episodes"]
                ),
                "secondary_calibration_seed_start": int(
                    reference["calibration_seed_start"]
                ),
                "secondary_same_train_variant_support_replicated": True,
                "secondary_paired_calibration": True,
                "secondary_candidates": [
                    _receipt_candidate(secondary_by_pair[pair]) for pair in tied
                ],
                "combined_tied_candidates": {
                    _pair_name(pair): combined[pair] for pair in tied
                },
                "selected_calibration_result_sha256": combined[selected_pair][
                    "combined_calibration_evidence_sha256"
                ],
            }
        )
    else:
        receipt.update(
            {
                "secondary_calibration_episodes": 0,
                "secondary_calibration_seed_start": None,
                "secondary_same_train_variant_support_replicated": None,
                "secondary_paired_calibration": None,
                "secondary_candidates": [],
                "combined_tied_candidates": {},
                "selected_calibration_result_sha256": selected[
                    "calibration_result_sha256"
                ],
            }
        )
    return receipt


__all__ = [
    "EMPIRICAL_SELECTION_ORDER",
    "FINAL_FALLBACK_ORDER",
    "SELECTOR_MODE",
    "TARGET_ONLY_DECODERS",
    "empirical_selection_key",
    "final_fallback_key",
    "select_deployment",
    "tied_top_pairs",
]
