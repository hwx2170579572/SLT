from __future__ import annotations

import torch

from algos.sb3_torch.hybrid_policy_v4_11_model import (
    ProperCalibratedRankedRiskSACPolicyV411,
)
from algos.sb3_torch.sac_v4_11_model import (
    continuous_pairwise_ranking_loss,
    soft_target_bernoulli_loss,
    temporal_risk_consistency_loss,
)


def test_soft_target_bernoulli_loss_is_finite_and_has_expected_gradient() -> None:
    logits = torch.tensor([[-2.0], [0.0], [2.0]], requires_grad=True)
    targets = torch.tensor([[1.0], [0.25], [0.0]])
    loss = soft_target_bernoulli_loss((logits,), targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad[0].item() < 0.0
    assert logits.grad[2].item() > 0.0


def test_continuous_ranking_prefers_target_order_without_a_class_threshold() -> None:
    targets = torch.tensor([[0.0], [0.2], [0.9]])
    wrong = torch.tensor([[2.0], [0.0], [-2.0]], requires_grad=True)
    right = torch.tensor([[-2.0], [0.0], [2.0]], requires_grad=True)
    wrong_loss, wrong_active, wrong_weight = continuous_pairwise_ranking_loss(
        (wrong,), targets
    )
    right_loss, right_active, right_weight = continuous_pairwise_ranking_loss(
        (right,), targets
    )
    wrong_loss.backward()
    assert right_loss < wrong_loss
    assert wrong_active == 1.0
    assert right_active == 1.0
    assert wrong_weight == right_weight
    assert wrong.grad is not None and bool(torch.isfinite(wrong.grad).all())


def test_equal_soft_targets_disable_ranking_continuously() -> None:
    logits = torch.tensor([[0.1], [0.2]], requires_grad=True)
    targets = torch.tensor([[0.4], [0.4]])
    loss, active, mean_weight = continuous_pairwise_ranking_loss((logits,), targets)
    loss.backward()
    assert loss.item() == 0.0
    assert active == 0.0
    assert mean_weight == 0.0
    assert logits.grad is not None


def test_temporal_consistency_detaches_teacher() -> None:
    student = torch.tensor([[0.1], [0.8]], requires_grad=True)
    teacher = torch.tensor([[0.2], [0.6]], requires_grad=True)
    loss = temporal_risk_consistency_loss((student,), (teacher,))
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_v4_11_policy_declares_model_only_boundary() -> None:
    assert ProperCalibratedRankedRiskSACPolicyV411.inference_safety_rule_added is False
    assert ProperCalibratedRankedRiskSACPolicyV411.external_kinematic_projection is False
    assert ProperCalibratedRankedRiskSACPolicyV411.action_postprocessing_override is False
