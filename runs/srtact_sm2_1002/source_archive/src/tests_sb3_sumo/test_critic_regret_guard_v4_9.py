from __future__ import annotations

import copy

import pytest
import torch

from algos.sb3_torch.hybrid_policy_v4_9_qguard import (
    DECODER,
    select_critic_regret_guard_fusion_lane_indices,
)
from tools.action_diagnostics_v4_9_qguard import (
    q_guard_decoder_integrity_passed,
    q_guard_decoder_metrics,
)


def test_q_guard_selection_covers_pass_veto_fallback_and_mask() -> None:
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
            [1.96, 2.0, 1.0],
            [1.94, 2.0, 1.0],
            [0.0, 1.0, 2.0],
            [9.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    actor, confidence, target, regret, base, passed, override, selected, tied = (
        select_critic_regret_guard_fusion_lane_indices(
            probabilities,
            mask,
            q_values,
            actor_non_keep_confidence_threshold=0.90,
            maximum_actor_target_q_regret=0.05,
        )
    )
    assert actor.tolist() == [0, 0, 1, 1]
    assert confidence.tolist() == pytest.approx([0.95, 0.95, 0.98, 0.01])
    assert target.tolist() == [1, 1, 2, 1]
    assert regret.tolist() == pytest.approx([0.04, 0.06, 1.0, 0.0])
    assert base.tolist() == [True, True, False, False]
    assert passed.tolist() == [True, False, False, True]
    assert override.tolist() == [True, False, False, False]
    assert selected.tolist() == [0, 1, 2, 1]
    assert tied.squeeze(1).tolist() == [True, True, False, True]


def test_q_guard_selection_rejects_negative_regret_limit() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        select_critic_regret_guard_fusion_lane_indices(
            torch.ones((1, 3)) / 3.0,
            torch.ones((1, 3), dtype=torch.bool),
            torch.zeros((1, 3)),
            actor_non_keep_confidence_threshold=0.90,
            maximum_actor_target_q_regret=-0.01,
        )


def _record(*, actor_q: float, override: bool) -> dict:
    selected = 0 if override else 1
    veto = not override
    regret = 2.0 - actor_q
    return {
        "episode": 0,
        "lane_command": (-1, 0, 1)[selected],
        "q_guard_fusion_decoder": {
            "lane_probabilities": [0.95, 0.04, 0.01],
            "valid_lane_actions": [True, True, True],
            "minimum_target_twin_q": [actor_q, 2.0, 1.0],
            "actor_selected_lane_index": 0,
            "actor_selected_lane": -1,
            "actor_selected_confidence": 0.95,
            "actor_non_keep_confidence_threshold": 0.90,
            "actor_target_q_regret": regret,
            "maximum_actor_target_q_regret": 0.05,
            "base_actor_override": True,
            "q_regret_guard_passed": override,
            "q_regret_veto_applied": veto,
            "actor_override": override,
            "target_selected_lane_index": 1,
            "target_selected_lane": 0,
            "target_selected_q_minus_keep_q": 0.0,
            "keep_was_exact_tied_target_maximum": True,
            "selected_lane_index": selected,
            "selected_lane": (-1, 0, 1)[selected],
            "selected_source": "actor" if override else "target_critic",
            "fallback_reason": "q_regret_veto" if veto else None,
        },
    }


def test_q_guard_trace_metrics_recompute_pass_and_veto() -> None:
    records = [_record(actor_q=1.96, override=True), _record(actor_q=1.94, override=False)]
    metrics = q_guard_decoder_metrics(records)
    assert metrics["deterministic_lane_decoder"] == DECODER
    assert metrics["exact_q_guard_fusion_rule_match_rate"] == 1.0
    assert metrics["exact_q_guard_fusion_action_match_rate"] == 1.0
    assert metrics["q_regret_veto_records"] == 1
    assert metrics["actor_override_records"] == 1
    assert q_guard_decoder_integrity_passed(
        {**metrics, "hybrid_exact_lateral_code_rate": 1.0}
    )
    tampered = copy.deepcopy(records)
    tampered[1]["q_guard_fusion_decoder"]["q_regret_veto_applied"] = False
    assert q_guard_decoder_metrics(tampered)["exact_q_guard_fusion_rule_match_rate"] == 0.5
