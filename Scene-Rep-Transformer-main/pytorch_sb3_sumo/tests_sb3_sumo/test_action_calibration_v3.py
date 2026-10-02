from __future__ import annotations

import numpy as np
import pytest

from tools.evaluate_lateral_calibration_v3 import (
    LateralScalePolicyAdapter,
    calibrate_lateral_action,
    gate_decision,
)


class _PredictModel:
    def __init__(self, action: np.ndarray) -> None:
        self.action = action
        self.policy = object()

    def predict(self, observation, deterministic=True):
        return self.action.copy(), {"state": 1}


def _actions(*, keep: float, applied: float) -> dict:
    return {
        "lane_command_rates": {
            "negative": (1.0 - keep) / 2.0,
            "keep": keep,
            "positive": (1.0 - keep) / 2.0,
        },
        "lane_change_applied_rate": applied,
    }


def test_calibrate_lateral_action_preserves_longitudinal_and_clips() -> None:
    action = np.asarray([[0.25, 0.1], [-0.5, -0.3]], dtype=np.float32)
    calibrated = calibrate_lateral_action(action, 5.0)
    np.testing.assert_allclose(calibrated[:, 0], action[:, 0])
    np.testing.assert_allclose(calibrated[:, 1], np.asarray([0.5, -1.0]))
    np.testing.assert_allclose(action[:, 1], np.asarray([0.1, -0.3]))


def test_calibrate_lateral_action_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        calibrate_lateral_action(np.zeros(2), 0.0)
    with pytest.raises(ValueError, match="action shape"):
        calibrate_lateral_action(np.zeros(3), 5.0)


def test_policy_adapter_uses_only_locked_scale() -> None:
    model = _PredictModel(np.asarray([0.2, 0.1], dtype=np.float32))
    adapter = LateralScalePolicyAdapter(model, scale=5.0)
    action, state = adapter.predict(None)
    np.testing.assert_allclose(action, np.asarray([0.2, 0.5]))
    assert state == {"state": 1}
    assert adapter.policy is model.policy
    with pytest.raises(ValueError, match="locked"):
        LateralScalePolicyAdapter(model, scale=4.0)


def test_carla_gate_requires_every_preregistered_check() -> None:
    passing = {
        "success_rate": 0.3,
        "collision_rate": 0.1,
        "off_route_rate": 0.0,
        "timeout_rate": 0.7,
    }
    decision = gate_decision(
        "carla", passing, _actions(keep=0.9, applied=0.02)
    )
    assert decision["decision"] == "pass"
    assert all(decision["checks"].values())
    assert decision["formal_test_unlocked"] is False

    failing = dict(passing, success_rate=0.2)
    decision = gate_decision(
        "carla", failing, _actions(keep=0.9, applied=0.02)
    )
    assert decision["decision"] == "fail"
    assert decision["checks"]["success_rate_min"] is False


def test_cross_gate_rejects_nonfinite_values() -> None:
    outcomes = {
        "success_rate": float("nan"),
        "collision_rate": 0.0,
        "off_route_rate": 0.0,
        "timeout_rate": 0.0,
    }
    decision = gate_decision(
        "cross", outcomes, _actions(keep=0.5, applied=0.1)
    )
    assert decision["decision"] == "fail"
    assert decision["checks"]["finite"] is False

