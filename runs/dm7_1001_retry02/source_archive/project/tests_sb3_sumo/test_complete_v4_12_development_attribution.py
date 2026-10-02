from __future__ import annotations

from tools import complete_v4_12_development_attribution as subject


def test_complete_attribution_selects_model_change_not_rule() -> None:
    payload = subject.build_complete_attribution()
    method = payload["selected_next_method"]
    assert method["version"] == "v4.13"
    assert method["lane_support_scale"] == 0.25
    assert method["lane_prior_coef"] == 0.05
    assert method["scenario_conditioned"] is False
    assert payload["formal_test_accessed"] is False
    assert all(payload["forbidden_mechanisms_for_next_method"].values())
