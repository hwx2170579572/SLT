"""Read existing evidence for the method-choice assessment; never run training."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_method_candidate_assessment_20260912_v1"
SCENES = (
    "left_turn", "cross", "roundabout_easy", "roundabout_medium",
    "roundabout", "carla",
)
SOURCES: dict[str, str] = {}


def read_json(path: str | Path):
    path = ROOT / path
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8-sig"))


def flag(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError(f"Unknown event flag: {value!r}")


def verify_rates(detail):
    episodes = detail["episode_records"]
    assert len(episodes) == detail["summary"]["episodes"]
    for metric, field in (
        ("success_rate", "success"), ("collision_rate", "collision"),
        ("timeout_rate", "timeout"),
    ):
        value = mean(flag(row[field]) for row in episodes)
        assert abs(value - detail["summary"][metric]) < 1e-12, metric
    assert abs(mean(row["episode_return"] for row in episodes)
               - detail["summary"]["mean_return"]) < 1e-12


def main():
    complete = read_json("results_phase2_reuse_v3/analysis/completion.json")
    assert complete["complete"] and complete["resolved"] == 66
    source_map = read_json("results_phase2_reuse_v3/analysis/evidence_sources.json")["sources"]
    decision = read_json("results_phase2_reuse_v3/analysis/screen_decision.json")
    curve_map = read_json("results_phase2_training_curves_v2_episode20_ema999/curve_values.json")
    curve_sources = read_json("results_phase2_training_curves_v2_episode20_ema999/source_manifest.json")
    cells = {}
    for unit, source in source_map.items():
        detail = read_json(source["path"])
        assert SOURCES[str(Path(source["path"]))] == source["sha256"], unit
        verify_rates(detail)
        cells[unit] = detail["summary"]

    ranking = []
    for row in decision["rows"]:
        method, candidate = row["method"], row["candidate"]
        per_scene = {scene: cells[f"{method}__{scene}__{candidate}__seed0"]
                     for scene in SCENES}
        metrics = {key: mean(values[key] for values in per_scene.values())
                   for key in ("success_rate", "collision_rate", "timeout_rate", "mean_return")}
        for key, value in row["metrics"].items():
            assert abs(metrics[key] - value) < 1e-12, (method, candidate, key)
        reasons = (row.get("guards") or {}).get("reasons", [])
        removed = [r for r in reasons if "efficiency" in r or "insufficient_paired_success" in r]
        remaining = [r for r in reasons if r not in removed]
        ranking.append({
            "method": method, "candidate": candidate, "metrics": metrics,
            "per_scene": per_scene, "original_failure_reasons": reasons,
            "efficiency_related_reasons_excluded_for_current_assessment": removed,
            "remaining_failure_reasons": remaining,
            "passes_remaining_guards": None if candidate == "control" else not remaining,
        })

    training = {}
    for unit, curve in curve_map.items():
        src = curve_sources[unit]
        path = ROOT / src["training_log"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == src["training_log_sha256"], unit
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
        assert len(rows) >= 20
        training[unit] = {
            "source": src,
            "raw_last_20_episode_mean_return": mean(float(r["r"]) for r in rows[-20:]),
            "raw_last_20_episode_success_rate": mean(flag(r["is_success"]) for r in rows[-20:]),
            "reference_style_points": {
                metric: {str(step): min(points, key=lambda p: abs(p[0] - step))[:2]
                         for step in (10000, 20000, 30000, 40000, 50000)}
                for metric, points in curve.items()
            },
        }

    early_evidence = []
    for relative in (
        "results_topo_v4_13_dev/development/summary.json",
        "results_topo_v4_13_dev/ablation/summary.json",
        "results_topo_v4_13_promotion/summary.json",
    ):
        stage = read_json(relative)
        for row in stage["per_run"]:
            if row["method"] == "temporal_graph":
                continue
            folder = Path(row["evidence_directory"])
            detail = read_json(folder / "paper_evaluation_detailed.json")
            verify_rates(detail)
            for metric in ("success_rate", "collision_rate", "timeout_rate", "mean_return"):
                assert abs(detail["summary"][metric] - row[metric]) < 1e-12
            early_evidence.append({
                key: row[key] for key in (
                    "method", "scenario", "seed", "raw_steps", "evaluation_episodes",
                    "success_rate", "collision_rate", "timeout_rate", "mean_return",
                    "selected_checkpoint_kind", "evidence_directory",
                )
            })

    code_files = (
        "configs/sb3_configs.py", "configs/sb3_configs_v4_8.py",
        "configs/sb3_configs_v4_13.py", "algos/sb3_torch/callbacks.py",
        "algos/sb3_torch/hybrid_policy_v4_13_model.py",
        "algos/sb3_torch/hybrid_policy_v4_12_model.py",
        "algos/sb3_torch/sac_v4_11_model.py",
        "tools/checkpoint_decoder_selector_v4_8.py",
        "tools/checkpoint_decoder_selector_v4_9_2.py",
        "results_phase1_checkpoint_diagnostics_v1/summary/REPORT.md",
    )
    for relative in code_files:
        SOURCES[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    output = {
        "assessment_date": "2026-09-12",
        "scope": "Method-choice development assessment after user removed efficiency veto; not a new preregistered screen or final-test claim.",
        "original_frozen_decision_modified": False,
        "training_or_evaluation_launched": False,
        "completed_units": complete,
        "ranking": ranking, "training_curves": training,
        "early_mechanism_and_promotion_evidence": early_evidence,
        "source_sha256": SOURCES,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "evidence_audit.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"verified_evaluation_cells": len(cells),
                      "verified_training_logs": len(training),
                      "verified_early_evidence_runs": len(early_evidence),
                      "candidates_passing_without_efficiency_veto": [
                          [r["method"], r["candidate"]] for r in ranking
                          if r["passes_remaining_guards"] is True],
                      "output": str(OUT / "evidence_audit.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
