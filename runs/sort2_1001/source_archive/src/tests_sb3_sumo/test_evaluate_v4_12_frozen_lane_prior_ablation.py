from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tools import evaluate_v4_12_frozen_lane_prior_ablation as subject


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.lane_prior_coef = 0.05


def _model() -> SimpleNamespace:
    return SimpleNamespace(policy=_Policy())


def test_continuous_intervention_preserves_learned_tensors() -> None:
    model = _model()
    receipt = subject.apply_lane_prior_intervention(model, 0.0)
    assert receipt["original_value"] == pytest.approx(0.05)
    assert receipt["intervened_value"] == 0.0
    assert receipt["learned_tensor_state_unchanged"] is True
    assert receipt["threshold_or_veto"] is False
    assert receipt["kinematic_projection"] is False
    assert receipt["parameter_state_sha256_before"] == receipt[
        "parameter_state_sha256_after"
    ]


def test_intervention_cannot_increase_coefficient() -> None:
    with pytest.raises(ValueError, match="may not increase"):
        subject.apply_lane_prior_intervention(_model(), 0.1)


def test_explicit_sensitivity_can_increase_to_preregistered_bound() -> None:
    model = _model()
    receipt = subject.apply_lane_prior_intervention(
        model, 0.1, allow_increase=True
    )
    assert receipt["intervened_value"] == pytest.approx(0.1)
    assert receipt["kind"].endswith("sensitivity")
    assert receipt["learned_tensor_state_unchanged"] is True


def test_intervention_rejects_negative_coefficient() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        subject.apply_lane_prior_intervention(_model(), -0.01)
