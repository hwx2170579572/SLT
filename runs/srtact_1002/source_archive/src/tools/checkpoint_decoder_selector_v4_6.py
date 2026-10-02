"""Paired train-only joint checkpoint/decoder selection for v4.6."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from tools.checkpoint_selector_v4_4 import CHECKPOINT_KINDS, sha256, verify_checkpoint_zip
from tools.checkpoint_selector_v4_5_1 import event_flag_counts, terminal_event_evidence


TARGET_DECODER = "target_critic"
FUSION_DECODER = "fusion_0_90"
PARENT_DECODER = "parent_control"
CANDIDATE_DECODERS = (TARGET_DECODER, FUSION_DECODER)
SELECTION_ORDER = (
    "maximize_success_count",
    "minimize_collision_count",
    "minimize_off_route_count",
    "minimize_timeout_count",
    "maximize_mean_return",
    "prefer_exact_final_on_complete_outcome_tie",
    "prefer_target_critic_on_complete_checkpoint_decoder_tie",
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


def deployment_selection_key(
    summary: dict[str, Any], *, checkpoint_kind: str, deployment_decoder: str
) -> tuple[int, int, int, int, float, int, int]:
    """Return the fully preregistered v4.6 lexicographic key."""

    if checkpoint_kind not in CHECKPOINT_KINDS:
        raise ValueError(f"unsupported checkpoint kind {checkpoint_kind!r}")
    if deployment_decoder not in (*CANDIDATE_DECODERS, PARENT_DECODER):
        raise ValueError(f"unsupported deployment decoder {deployment_decoder!r}")
    counts = event_flag_counts(summary)
    return (
        counts["success_count"],
        -counts["collision_count"],
        -counts["off_route_count"],
        -counts["timeout_count"],
        float(summary["mean_return"]),
        int(checkpoint_kind == "exact_final"),
        int(deployment_decoder == TARGET_DECODER),
    )


def _validate_candidate_set(candidates: list[dict[str, Any]]) -> str:
    pairs = [
        (str(row["checkpoint_kind"]), str(row["deployment_decoder"]))
        for row in candidates
    ]
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"duplicate deployment candidates: {pairs}")
    observed = set(pairs)
    joint = {
        (checkpoint, decoder)
        for checkpoint in CHECKPOINT_KINDS
        for decoder in CANDIDATE_DECODERS
    }
    parent = {(checkpoint, PARENT_DECODER) for checkpoint in CHECKPOINT_KINDS}
    if observed == joint:
        return "joint_checkpoint_decoder"
    if observed == parent:
        return "checkpoint_only_parent_control"
    raise ValueError(
        "deployment candidates must be the complete v4.6 candidate Cartesian "
        "product or the two parent-control checkpoint candidates"
    )


def select_deployment(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate paired evidence and select a checkpoint/decoder deployment."""

    if not candidates:
        raise ValueError("joint selector received no candidates")
    selector_mode = _validate_candidate_set(candidates)
    reference = candidates[0]
    reference_signature = _episode_pair_signature(reference["episode_records"])
    reference_episodes = int(reference["summary"]["episodes"])
    reference_seed_start = int(reference["calibration_seed_start"])
    reference_partition = str(reference["traffic_partition"])
    if reference_partition != "train":
        raise ValueError("deployment selection may only use the train partition")

    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    checkpoint_identity: dict[str, tuple[str, str]] = {}
    for candidate in candidates:
        checkpoint = str(candidate["checkpoint_kind"])
        decoder = str(candidate["deployment_decoder"])
        path = Path(candidate["checkpoint_path"])
        verify_checkpoint_zip(path)
        digest = str(candidate["checkpoint_sha256"]).lower()
        if sha256(path) != digest:
            raise ValueError("checkpoint hash changed during joint selection")
        identity = (str(path.resolve()), digest)
        previous = checkpoint_identity.setdefault(checkpoint, identity)
        if previous != identity:
            raise ValueError(f"{checkpoint} uses different weights across decoders")
        if str(candidate["traffic_partition"]) != reference_partition:
            raise ValueError("deployment candidates used different traffic partitions")
        if int(candidate["summary"]["episodes"]) != reference_episodes:
            raise ValueError("deployment candidates used different episode counts")
        if int(candidate["calibration_seed_start"]) != reference_seed_start:
            raise ValueError("deployment candidates used different calibration seeds")
        signature = _episode_pair_signature(candidate["episode_records"])
        if signature != reference_signature:
            raise ValueError("deployment candidates used non-paired traffic episodes")
        if candidate.get("decoder_integrity_passed") is not True:
            raise ValueError(f"decoder integrity failed for {checkpoint} x {decoder}")
        evidence[(checkpoint, decoder)] = terminal_event_evidence(
            candidate["summary"], candidate["episode_records"]
        )

    ranked = sorted(
        candidates,
        key=lambda row: deployment_selection_key(
            row["summary"],
            checkpoint_kind=str(row["checkpoint_kind"]),
            deployment_decoder=str(row["deployment_decoder"]),
        ),
        reverse=True,
    )
    selected = ranked[0]
    receipt_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        checkpoint = str(candidate["checkpoint_kind"])
        decoder = str(candidate["deployment_decoder"])
        receipt_candidates.append(
            {
                "checkpoint_kind": checkpoint,
                "deployment_decoder": decoder,
                "checkpoint_path": str(Path(candidate["checkpoint_path"]).resolve()),
                "checkpoint_sha256": str(candidate["checkpoint_sha256"]).lower(),
                "calibration_seed_start": int(candidate["calibration_seed_start"]),
                "traffic_partition": str(candidate["traffic_partition"]),
                "summary": candidate["summary"],
                "outcome_counts": event_flag_counts(candidate["summary"]),
                "terminal_event_evidence": evidence[(checkpoint, decoder)],
                "selection_key": list(
                    deployment_selection_key(
                        candidate["summary"],
                        checkpoint_kind=checkpoint,
                        deployment_decoder=decoder,
                    )
                ),
                "episode_pair_signature": _episode_pair_signature(
                    candidate["episode_records"]
                ),
                "calibration_result_sha256": candidate["calibration_result_sha256"],
                "calibration_action_diagnostics_sha256": candidate[
                    "calibration_action_diagnostics_sha256"
                ],
                "decoder_integrity": candidate["decoder_integrity"],
                "decoder_integrity_passed": True,
            }
        )

    return {
        "schema_version": "topo-scene-v4.6.checkpoint-decoder-selector/v1",
        "selection_partition": "train",
        "selector_mode": selector_mode,
        "paired_calibration": True,
        "identical_unique_traffic_block_for_all_pairs": True,
        "calibration_episodes": reference_episodes,
        "calibration_seed_start": reference_seed_start,
        "selection_order": list(SELECTION_ORDER),
        "terminal_event_semantics": "source_flags_may_overlap",
        "summary_and_episode_records_preserved": True,
        "event_precedence_added": False,
        "candidate_count": len(candidates),
        "candidates": receipt_candidates,
        "selected_checkpoint_kind": str(selected["checkpoint_kind"]),
        "selected_checkpoint_path": str(Path(selected["checkpoint_path"]).resolve()),
        "selected_checkpoint_sha256": str(selected["checkpoint_sha256"]).lower(),
        "selected_deployment_decoder": str(selected["deployment_decoder"]),
        "selected_calibration_result_sha256": selected["calibration_result_sha256"],
        "validation_used_for_selection": False,
        "formal_test_used_for_selection": False,
    }


__all__ = [
    "CANDIDATE_DECODERS",
    "FUSION_DECODER",
    "PARENT_DECODER",
    "SELECTION_ORDER",
    "TARGET_DECODER",
    "deployment_selection_key",
    "select_deployment",
]
