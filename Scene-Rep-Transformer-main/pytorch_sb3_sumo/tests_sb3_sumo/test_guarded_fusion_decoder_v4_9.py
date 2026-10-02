from __future__ import annotations

import copy

import pytest
import torch

from algos.sb3_torch.hybrid_policy_v4_9 import (
    select_keep_tie_veto_fusion_lane_indices,
)
from tools.action_diagnostics_v4_9 import (
    DECODER,
    guarded_decoder_integrity_passed,
    keep_tie_veto_decoder_metrics,
)


def test_guarded_selection_vetoes_only_confident_override_on_keep_tie() -> None:
    probabilities = torch.tensor(
        [
            [0.95, 0.04, 0.01],
            [0.95, 0.04, 0.01],
            [0.01, 0.98, 0.01],
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
    q_values = torch.tensor(
        [
            [0.0, 2.0, 2.0],
            [3.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
            [9.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    actor, confidence, target, base, veto, override, selected, tied = (
        select_keep_tie_veto_fusion_lane_indices(
            probabilities,
            mask,
            q_values,
            actor_non_keep_confidence_threshold=0.90,
        )
    )
    assert actor.tolist() == [0, 0, 1, 1]
    assert confidence.tolist() == pytest.approx([0.95, 0.95, 0.98, 0.01])
    assert target.tolist() == [1, 0, 2, 1]
    assert base.tolist() == [True, True, False, False]
    assert veto.tolist() == [True, False, False, False]
    assert override.tolist() == [False, True, False, False]
    assert selected.tolist() == [1, 0, 2, 1]
    assert tied.squeeze(1).tolist() == [True, False, False, True]


def test_guarded_selection_rejects_invalid_contracts() -> None:
    probabilities = torch.ones((1, 3), dtype=torch.float32) / 3.0
    q_values = torch.zeros((1, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="feasible"):
        select_keep_tie_veto_fusion_lane_indices(
            probabilities,
            torch.zeros((1, 3), dtype=torch.bool),
            q_values,
            actor_non_keep_confidence_threshold=0.90,
        )
    with pytest.raises(ValueError, match="threshold"):
        select_keep_tie_veto_fusion_lane_indices(
            probabilities,
            torch.ones((1, 3), dtype=torch.bool),
            q_values,
            actor_non_keep_confidence_threshold=-0.01,
        )


def _record(*, keep_tied: bool, override: bool) -> dict:
    q_values = [1.0, 2.0, 2.0] if keep_tied else [3.0, 2.0, 1.0]
    selected_index = 0 if override else 1
    veto = keep_tied
    return {
        "episode": 0,
        "lane_command": (-1, 0, 1)[selected_index],
        "guarded_fusion_decoder": {
            "lane_probabilities": [0.95, 0.04, 0.01],
            "valid_lane_actions": [True, True, True],
            "minimum_target_twin_q": q_values,
            "actor_selected_lane_index": 0,
            "actor_selected_lane": -1,
            "actor_selected_confidence": 0.95,
            "actor_non_keep_confidence_threshold": 0.90,
            "base_actor_override": True,
            "keep_tie_veto_applied": veto,
            "actor_override": override,
            "target_selected_lane_index": 1 if keep_tied else 0,
            "target_selected_lane": 0 if keep_tied else -1,
            "target_selected_q_minus_keep_q": 0.0 if keep_tied else 1.0,
            "keep_was_exact_tied_target_maximum": keep_tied,
            "selected_lane_index": selected_index,
            "selected_lane": (-1, 0, 1)[selected_index],
            "selected_source": "actor" if override else "target_critic",
            "fallback_reason": "keep_tie_veto" if veto else None,
        },
    }


def test_guarded_trace_metrics_recompute_veto_and_override() -> None:
    records = [_record(keep_tied=True, override=False), _record(keep_tied=False, override=True)]
    metrics = keep_tie_veto_decoder_metrics(records)
    assert metrics["deterministic_lane_decoder"] == DECODER
    assert metrics["exact_guarded_fusion_rule_match_rate"] == 1.0
    assert metrics["exact_guarded_fusion_action_match_rate"] == 1.0
    assert metrics["keep_tie_veto_records"] == 1
    assert metrics["actor_override_records"] == 1
    assert metrics["selected_source_counts"] == {"actor": 1, "target_critic": 1}

    complete = {
        **metrics,
        "hybrid_exact_lateral_code_rate": 1.0,
    }
    assert guarded_decoder_integrity_passed(complete)
    tampered = copy.deepcopy(records)
    tampered[0]["guarded_fusion_decoder"]["keep_tie_veto_applied"] = False
    assert (
        keep_tie_veto_decoder_metrics(tampered)[
            "exact_guarded_fusion_rule_match_rate"
        ]
        == 0.5
    )
