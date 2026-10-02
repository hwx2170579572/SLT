"""Tie-only replicated train calibration selector for v4.8."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from tools.checkpoint_decoder_selector_v4_6 import (
    CANDIDATE_DECODERS,
    FUSION_DECODER,
    TARGET_DECODER,
    select_deployment as select_parent_deployment,
)
from tools.checkpoint_selector_v4_4 import CHECKPOINT_KINDS, sha256, verify_checkpoint_zip
from tools.checkpoint_selector_v4_5_1 import event_flag_counts, terminal_event_evidence


EMPIRICAL_SELECTION_ORDER = (
    "maximize_success_count",
    "minimize_collision_count",
    "minimize_off_route_count",
    "minimize_timeout_count",
    "maximize_mean_return",
)
FINAL_FALLBACK_ORDER = (
    "prefer_exact_final_if_still_tied",
    "prefer_target_critic_if_still_tied",
)
SELECTOR_MODE = "tie_only_replicated_joint_checkpoint_decoder"


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


def empirical_selection_key(summary: dict[str, Any]) -> tuple[int, int, int, int, float]:
    counts = event_flag_counts(summary)
    return (
        counts["success_count"],
        -counts["collision_count"],
        -counts["off_route_count"],
        -counts["timeout_count"],
        float(summary["mean_return"]),
    )


def final_fallback_key(pair: tuple[str, str]) -> tuple[int, int]:
    checkpoint, decoder = pair
    return (
        int(checkpoint == "exact_final"),
        int(decoder == TARGET_DECODER),
    )


def tied_top_pairs(initial_candidates: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    """Validate the primary matrix and return empirical first-place ties."""

    select_parent_deployment(initial_candidates)
    best = max(empirical_selection_key(row["summary"]) for row in initial_candidates)
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
) -> None:
    expected = set(tied)
    observed = [_pair(row) for row in secondary_candidates]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("secondary candidate set must equal the empirical tied-top set")

    initial_by_pair = {_pair(row): row for row in initial_candidates}
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
        raise ValueError("secondary calibration seed does not match the frozen offset")
    primary_signature = _signature(initial_candidates[0]["episode_records"])
    if {row["seed"] for row in primary_signature} & {row["seed"] for row in reference_signature}:
        raise ValueError("primary and secondary calibration seeds overlap")
    if [row["traffic_variant"] for row in primary_signature] != [
        row["traffic_variant"] for row in reference_signature
    ]:
        raise ValueError("secondary calibration did not replicate train variant support")

    for row in secondary_candidates:
        pair = _pair(row)
        checkpoint, decoder = pair
        if checkpoint not in CHECKPOINT_KINDS or decoder not in CANDIDATE_DECODERS:
            raise ValueError(f"unsupported secondary pair {pair}")
        path = Path(row["checkpoint_path"])
        verify_checkpoint_zip(path)
        digest = str(row["checkpoint_sha256"]).lower()
        if sha256(path) != digest:
            raise ValueError("secondary checkpoint hash changed")
        primary = initial_by_pair[pair]
        if str(Path(primary["checkpoint_path"]).resolve()) != str(path.resolve()):
            raise ValueError("secondary calibration used a different checkpoint path")
        if str(primary["checkpoint_sha256"]).lower() != digest:
            raise ValueError("secondary calibration used different checkpoint weights")
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
        "episodes": episodes,
        **{key: value for key, value in counts.items() if key != "episodes"},
        "mean_return": mean_return,
        "empirical_selection_key": list(empirical_key),
        "primary_calibration_result_sha256": primary["calibration_result_sha256"],
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


def select_deployment(
    initial_candidates: list[dict[str, Any]],
    secondary_candidates: list[dict[str, Any]] | None = None,
    *,
    secondary_seed_offset: int = 100,
) -> dict[str, Any]:
    """Select using a second train-only round iff empirical first place ties."""

    parent_receipt = select_parent_deployment(initial_candidates)
    if parent_receipt["selector_mode"] != "joint_checkpoint_decoder":
        raise ValueError("v4.8 tie-only selector requires the four-pair candidate matrix")
    initial_by_pair = {_pair(row): row for row in initial_candidates}
    tied = tied_top_pairs(initial_candidates)
    parent_pair = (
        str(parent_receipt["selected_checkpoint_kind"]),
        str(parent_receipt["selected_deployment_decoder"]),
    )

    secondary_triggered = len(tied) > 1
    combined: dict[tuple[str, str], dict[str, Any]] = {}
    secondary_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    if not secondary_triggered:
        if secondary_candidates:
            raise ValueError("secondary calibration is forbidden without an empirical tie")
        selected_pair = tied[0]
    else:
        if not secondary_candidates:
            raise ValueError("secondary calibration is required for tied top candidates")
        _validate_secondary(
            initial_candidates,
            secondary_candidates,
            tied,
            secondary_seed_offset=secondary_seed_offset,
        )
        secondary_by_pair = {_pair(row): row for row in secondary_candidates}
        combined = {
            pair: _combined_evidence(initial_by_pair[pair], secondary_by_pair[pair])
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
    receipt = dict(parent_receipt)
    receipt.update(
        {
            "schema_version": "topo-scene-v4.8.tie-only-replicated-selector/v1",
            "selector_mode": SELECTOR_MODE,
            "selection_order": [
                *EMPIRICAL_SELECTION_ORDER,
                "if_empirical_top_tied_run_second_paired_train_seed_block",
                "rerank_tied_top_by_combined_24_episode_empirical_fields",
                *FINAL_FALLBACK_ORDER,
            ],
            "initial_calibration_episodes": int(
                initial_candidates[0]["summary"]["episodes"]
            ),
            "initial_calibration_seed_start": int(
                initial_candidates[0]["calibration_seed_start"]
            ),
            "initial_empirical_tied_top_count": len(tied),
            "initial_empirical_tied_top_pairs": [_pair_name(pair) for pair in tied],
            "secondary_calibration_triggered": secondary_triggered,
            "secondary_calibration_seed_offset": int(secondary_seed_offset),
            "secondary_candidate_scope": "empirical_tied_top_only",
            "non_top_candidate_reentry_forbidden": True,
            "parent_v4_7_counterfactual_selected_pair": _pair_name(parent_pair),
            "selected_checkpoint_kind": selected_pair[0],
            "selected_checkpoint_path": str(Path(selected["checkpoint_path"]).resolve()),
            "selected_checkpoint_sha256": str(selected["checkpoint_sha256"]).lower(),
            "selected_deployment_decoder": selected_pair[1],
            "validation_used_for_selection": False,
            "formal_test_used_for_selection": False,
        }
    )
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
                    {
                        "checkpoint_kind": pair[0],
                        "deployment_decoder": pair[1],
                        "checkpoint_path": str(
                            Path(secondary_by_pair[pair]["checkpoint_path"]).resolve()
                        ),
                        "checkpoint_sha256": str(
                            secondary_by_pair[pair]["checkpoint_sha256"]
                        ).lower(),
                        "traffic_partition": "train",
                        "calibration_seed_start": int(
                            secondary_by_pair[pair]["calibration_seed_start"]
                        ),
                        "summary": secondary_by_pair[pair]["summary"],
                        "outcome_counts": event_flag_counts(
                            secondary_by_pair[pair]["summary"]
                        ),
                        "terminal_event_evidence": terminal_event_evidence(
                            secondary_by_pair[pair]["summary"],
                            secondary_by_pair[pair]["episode_records"],
                        ),
                        "episode_pair_signature": _signature(
                            secondary_by_pair[pair]["episode_records"]
                        ),
                        "calibration_result_sha256": secondary_by_pair[pair][
                            "calibration_result_sha256"
                        ],
                        "calibration_action_diagnostics_sha256": secondary_by_pair[
                            pair
                        ]["calibration_action_diagnostics_sha256"],
                        "decoder_integrity": secondary_by_pair[pair][
                            "decoder_integrity"
                        ],
                        "decoder_integrity_passed": True,
                    }
                    for pair in tied
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
    "empirical_selection_key",
    "final_fallback_key",
    "select_deployment",
    "tied_top_pairs",
]

