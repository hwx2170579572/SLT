"""v4.2 route-window probability-margin diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from envs.sumo.decision_alignment_v4 import lane_command_index
from tools.action_diagnostics_v4 import evaluate_with_action_diagnostics_v4


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "population_std": None,
            "minimum": None,
            "p10": None,
            "median": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "population_std": float(array.std()),
        "minimum": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def _route_margins(records: list[dict[str, Any]]) -> list[float]:
    hybrid_trace = any(record.get("hybrid_policy") is not None for record in records)
    margins: list[float] = []
    for record in records:
        route_intent = int(record["pre_action_route_intent"])
        if not bool(record["pre_action_route_intent_valid"]) or route_intent == 0:
            continue
        hybrid = record.get("hybrid_policy")
        if hybrid is None and hybrid_trace:
            raise ValueError("v4.2 candidate trace is missing hybrid-policy probabilities")
        if hybrid is None:
            continue
        probabilities = hybrid["lane_probabilities"]
        margins.append(
            float(probabilities[lane_command_index(route_intent)])
            - float(probabilities[lane_command_index(0)])
        )
    return margins


def evaluate_with_action_diagnostics_v4_2(
    model: Any,
    env: Any,
    *,
    episodes: int,
    seed: int,
    trace_path: Path,
):
    report, diagnostics = evaluate_with_action_diagnostics_v4(
        model,
        env,
        episodes=episodes,
        seed=seed,
        trace_path=trace_path,
    )
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    margins = _route_margins(records)
    diagnostics = dict(diagnostics)
    diagnostics["schema_version"] = "topo-scene-v4.2.action-diagnostics/v1"
    diagnostics["route_window_required_minus_keep_probability_margin"] = _summary(
        margins
    )
    return report, diagnostics


__all__ = ["evaluate_with_action_diagnostics_v4_2"]
