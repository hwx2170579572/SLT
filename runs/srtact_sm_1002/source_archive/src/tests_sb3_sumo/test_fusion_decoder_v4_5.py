from __future__ import annotations

import copy

import pytest
import torch

from algos.sb3_torch.hybrid_policy_v4_5 import select_fusion_lane_indices
from tools.action_diagnostics_v4_5 import fusion_decoder_metrics


def test_fusion_selection_covers_override_fallback_tie_and_mask() -> None:
    probabilities = torch.tensor(
        [
            [0.95, 0.04, 0.01],
            [0.01, 0.98, 0.01],
            [0.90, 0.09, 0.01],
            [0.99, 0.01, 0.00],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [False, True, True],
        ]
    )
    minimum_q = torch.tensor(
        [
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
            [0.0, 3.0, 1.0],
            [9.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    actor, confidence, target, override, selected, keep_tied = (
        select_fusion_lane_indices(
            probabilities,
            mask,
            minimum_q,
            actor_non_keep_confidence_threshold=0.90,
        )
    )
    assert actor.tolist() == [0, 1, 0, 1]
    assert confidence.tolist() == pytest.approx([0.95, 0.98, 0.90, 0.01])
    assert target.tolist() == [1, 2, 1, 1]
    assert override.tolist() == [True, False, True, False]
    assert selected.tolist() == [0, 2, 0, 1]
    assert keep_tied.squeeze(1).tolist() == [True, False, True, True]


def test_fusion_selection_rejects_invalid_contracts() -> None:
    probabilities = torch.ones((1, 3), dtype=torch.float32) / 3.0
    q_values = torch.zeros((1, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="feasible"):
        select_fusion_lane_indices(
            probabilities,
            torch.zeros((1, 3), dtype=torch.bool),
            q_values,
            actor_non_keep_confidence_threshold=0.90,
        )
    with pytest.raises(ValueError, match="threshold"):
        select_fusion_lane_indices(
            probabilities,
            torch.ones((1, 3), dtype=torch.bool),
            q_values,
            actor_non_keep_confidence_threshold=1.01,
        )


def _trace_record(
    *,
    episode: int,
    probabilities: list[float],
    q_values: list[float],
    actor_index: int,
    target_index: int,
    selected_index: int,
    override: bool,
) -> dict:
    lanes = (-1, 0, 1)
    return {
        "episode": episode,
        "lane_command": lanes[selected_index],
        "pre_action_route_intent_valid": True,
        "pre_action_route_intent": lanes[actor_index],
        "fusion_decoder": {
            "lane_probabilities": probabilities,
            "valid_lane_actions": [True, True, True],
            "minimum_target_twin_q": q_values,
            "actor_selected_lane_index": actor_index,
            "actor_selected_lane": lanes[actor_index],
            "actor_selected_confidence": probabilities[actor_index],
            "actor_non_keep_confidence_threshold": 0.90,
            "actor_override": override,
            "target_selected_lane_index": target_index,
            "target_selected_lane": lanes[target_index],
            "target_selected_q_minus_keep_q": (
                q_values[target_index] - q_values[1]
            ),
            "keep_was_exact_tied_target_maximum": (
                q_values[1] == max(q_values)
            ),
            "selected_lane_index": selected_index,
            "selected_lane": lanes[selected_index],
            "selected_source": "actor" if override else "target_critic",
        },
    }


def test_fusion_trace_metrics_recompute_both_branches() -> None:
    records = [
        _trace_record(
            episode=0,
            probabilities=[0.95, 0.04, 0.01],
            q_values=[0.0, 2.0, 1.0],
            actor_index=0,
            target_index=1,
            selected_index=0,
            override=True,
        ),
        _trace_record(
            episode=1,
            probabilities=[0.01, 0.98, 0.01],
            q_values=[0.0, 1.0, 2.0],
            actor_index=1,
            target_index=2,
            selected_index=2,
            override=False,
        ),
    ]
    metrics = fusion_decoder_metrics(records)
    assert metrics["fusion_decoder_records"] == 2
    assert metrics["exact_fusion_rule_match_rate"] == 1.0
    assert metrics["exact_fusion_action_match_rate"] == 1.0
    assert metrics["selected_action_mask_feasible_rate"] == 1.0
    assert metrics["actor_override_records"] == 1
    assert metrics["target_fallback_records"] == 1
    assert metrics["actor_override_predicate_valid_rate"] == 1.0
    assert metrics["target_fallback_rule_match_rate"] == 1.0
    assert metrics["selected_source_counts"] == {"actor": 1, "target_critic": 1}

    tampered = copy.deepcopy(records)
    tampered[0]["fusion_decoder"]["selected_lane_index"] = 1
    assert fusion_decoder_metrics(tampered)["exact_fusion_rule_match_rate"] == 0.5
