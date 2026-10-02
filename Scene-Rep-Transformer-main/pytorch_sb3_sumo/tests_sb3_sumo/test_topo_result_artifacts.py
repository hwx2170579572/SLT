from __future__ import annotations

from pathlib import Path

import pytest
from matplotlib import pyplot as plt

from tools.build_topo_final_artifacts import _choose_decision
from tools.plot_topo_results import METHOD_ORDER, _build_figure, _timeline_index, _timeline_key


def _outcome(success: float, collision: float, mean_return: float) -> dict[str, float]:
    return {
        "success_rate": success,
        "collision_rate": collision,
        "mean_return": mean_return,
    }


def test_plot_timeline_names_are_explicit_raw_steps() -> None:
    assert _timeline_key("full_10k") == ("full", 10_000)
    assert _timeline_key("topo_scene_50000") == ("full", 50_000)
    assert _timeline_key("balanced_40k") == ("balanced", 40_000)


def test_plot_timeline_requires_matched_checkpoint_grid() -> None:
    models = {"full_10k": {}, "full_20k": {}, "balanced_10k": {}, "balanced_20k": {}}
    index = _timeline_index(models)
    assert sorted(index["full"]) == [10_000, 20_000]

    with pytest.raises(ValueError, match="must match exactly"):
        _timeline_index({"full_10k": {}, "full_20k": {}, "balanced_10k": {}})


def test_final_decision_retains_safe_temporal_ablation_without_topology_gain() -> None:
    results = {
        "scene_rep": _outcome(1.0, 0.0, 1.0),
        "temporal_graph": _outcome(1.0, 0.0, 1.0),
        "topo_scene": _outcome(0.6, 0.4, 0.2),
        "topo_scene_balanced": _outcome(1.0, 0.0, 1.0),
    }
    decision, reason = _choose_decision(results)
    assert decision == "retain_temporal_only"
    assert "MST+SLT" in reason


def test_final_decision_only_confirms_a_safe_primary_gain() -> None:
    results = {
        "scene_rep": _outcome(0.7, 0.1, 0.4),
        "temporal_graph": _outcome(0.8, 0.1, 0.6),
        "topo_scene": _outcome(0.6, 0.3, 0.0),
        "topo_scene_balanced": _outcome(0.9, 0.1, 0.8),
    }
    assert _choose_decision(results)[0] == "confirm"

    results["topo_scene_balanced"] = _outcome(0.9, 0.2, 0.8)
    assert _choose_decision(results)[0] != "confirm"


def test_dashboard_renderer_uses_only_supplied_rows(tmp_path: Path) -> None:
    final_rows = {
        method: {
            "success_rate": 0.5,
            "collision_rate": 0.1,
            "mean_return": 0.3,
            "raw_steps": 50_000,
        }
        for method in METHOD_ORDER
    }
    curves = {
        method: {
            "points": [
                {
                    "raw_steps": 10_000,
                    "success_rate": 0.5,
                    "collision_rate": 0.1,
                    "mean_return": 0.3,
                }
            ]
        }
        for method in METHOD_ORDER
    }
    probe_slot = {"latent_std_mean": 0.5}
    probes = {
        "results": {
            "full_10k": {slot: dict(probe_slot) for slot in ("ego", "social", "route")},
            "balanced_10k": {slot: dict(probe_slot) for slot in ("ego", "social", "route")},
        }
    }
    action_row = {
        "action_summary_by_risk": {
            "critical_ttc_le_3s": {"target_speed_mps_mean": 4.0}
        },
        "slot_mean_ablation": {
            "route": {
                "critical_ttc_le_3s": {
                    "mean_absolute_target_speed_delta_mps": 0.2
                }
            }
        },
    }
    actions = {"models": {"full_10k": action_row, "balanced_10k": action_row}}
    timeline = _timeline_index({"full_10k": {}, "balanced_10k": {}})
    source_rows: list[dict[str, object]] = []
    figure = _build_figure(final_rows, curves, probes, actions, timeline, source_rows)
    target = tmp_path / "dashboard.svg"
    figure.savefig(target)
    plt.close(figure)
    assert target.is_file()
    assert {row["panel"] for row in source_rows} == {"a", "b", "c", "d"}
