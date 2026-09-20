from __future__ import annotations

import pytest
import torch

from tools import diagnose_v4_12_lane_speed_gradient_coupling as subject


def test_gradient_vector_preserves_parameter_layout_and_conflict() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    positive = parameter.sum()
    negative = -parameter.sum()
    left = subject.gradient_vector(
        positive, (parameter,), retain_graph=True
    )
    right = subject.gradient_vector(
        negative, (parameter,), retain_graph=False
    )
    assert left.tolist() == pytest.approx([1.0, 1.0])
    assert right.tolist() == pytest.approx([-1.0, -1.0])
    assert subject._cosine(left, right) == pytest.approx(-1.0)
    assert subject._opposing_sign_fraction(left, right) == pytest.approx(1.0)
