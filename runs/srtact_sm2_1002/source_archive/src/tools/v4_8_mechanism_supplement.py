"""v4_8/lr_half mechanism supplement: confidence trace (B1) + decoder ablation (B2).

Two evidence-bound experiments, both on the frozen ``v4_8__{cross,carla}__lr_half``
checkpoint and the frozen legacy (training) traffic, so the fusion decoder's causal
contribution is isolated without any selector mixing:

* B1 -- confidence trace.  Replay the fusion decoder with decision tracing and
  characterise ``actor_selected_confidence`` (and its margin to the 0.90 gate) split
  by episode outcome, plus the actor-vs-target Q regret of the override population.
* B2 -- decoder ablation.  Three deterministic decoders on the same checkpoint and
  the same legacy traffic: ``native`` (confidence-gated fusion), ``target_only``
  (always target critic) and ``actor_deterministic`` (always actor).

The decoder ablation reuses the frozen phase-3 ``evaluate`` machinery (sealing,
resume, traffic replay).  For CARLA the phase-3 selector chose ``target_critic`` as
the native decoder, so the fusion arm must be forced by loading with the
``fusion_0_90`` policy class and stripping the reused legacy-evaluation binding;
this is done explicitly and declared in the output, never silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase3_mechanism_v1.common import (  # noqa: E402
    OUT,
    digest,
    read,
    seal,
    source_registry,
)


SCENARIOS = ("cross", "carla")
DECODERS = ("native", "target_only", "actor_deterministic")
STEP = "50000"
MODE = "legacy"
EPISODES = 100
SEED_START = 10000
FUSION = "fusion_0_90"


def _forced_fusion_source(source: dict[str, Any]) -> dict[str, Any]:
    """CARLA's selected native decoder is target_critic; force the fusion loader."""
    forced = dict(source)
    forced["decoder"] = FUSION
    forced.pop("legacy_evaluation", None)
    forced.pop("legacy_evaluation_sha256", None)
    return forced


def _summary_from_evaluate(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": result.get("summary"),
        "origin": result.get("origin"),
        "checkpoint_sha256": result.get("identity", {}).get("checkpoint_sha256"),
        "native_decoder": result.get("identity", {}).get("native_decoder"),
        "decoder": result.get("identity", {}).get("decoder"),
    }


def run_decoder_ablation(
    source: dict[str, Any], scenario: str, force_fusion: bool, device: str
) -> dict[str, Any]:
    from tools.phase3_mechanism_v1.evaluation import evaluate

    src = _forced_fusion_source(source) if force_fusion else source
    arms: dict[str, Any] = {}
    for decoder in DECODERS:
        rel = evaluate(src, STEP, MODE, decoder, device, smoke=False)
        result = read(ROOT / rel)
        arms[decoder] = {
            **_summary_from_evaluate(result),
            "result_path": rel,
        }
    return {
        "scenario": scenario,
        "forced_fusion_loader": force_fusion,
        "loader_decoder": src["decoder"],
        "checkpoint_path": src["checkpoints"][STEP]["path"],
        "mode": MODE,
        "episodes": result.get("summary", {}).get("episodes"),
        "arms": arms,
    }


def run_fusion_trace(
    source: dict[str, Any],
    scenario: str,
    force_fusion: bool,
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    from tools.paper_evaluation_contract import validate_model_environment_spaces
    from tools.phase3_mechanism_v1.traffic import make_env
    from tools.action_diagnostics_v4_5 import evaluate_with_action_diagnostics_v4_5
    from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45
    from tools.action_diagnostics_v4_6 import load_model_for_deployment

    src = _forced_fusion_source(source) if force_fusion else source
    checkpoint = ROOT / src["checkpoints"][STEP]["path"]
    namespace = digest({"fusion_trace": scenario, "step": STEP})[:16]
    env = make_env(src, MODE, namespace)
    trace_path = out_dir / f"{scenario}_fusion_trace.jsonl"
    try:
        model = load_model_for_deployment(
            ConfidentActorFusionSACV45, checkpoint,
            decoder=FUSION, env=env, device=device,
        )
        validate_model_environment_spaces(model, env)
        report, diagnostics = evaluate_with_action_diagnostics_v4_5(
            model, env, episodes=EPISODES, seed=SEED_START, trace_path=trace_path
        )
        summary = report.summary.to_dict()
    finally:
        env.close()

    # Cross-check the fusion replay against the frozen legacy evaluation when one
    # exists for the fusion decoder (cross only; carla's frozen eval is target_critic).
    frozen_check = None
    if not force_fusion and src.get("legacy_evaluation"):
        frozen = read(ROOT / src["legacy_evaluation"])
        frozen_summary = frozen["summary"]
        frozen_check = {
            "success_rate_match": summary["success_rate"] == frozen_summary["success_rate"],
            "collision_rate_match": summary["collision_rate"] == frozen_summary["collision_rate"],
            "timeout_rate_match": summary["timeout_rate"] == frozen_summary["timeout_rate"],
            "frozen": frozen_summary,
            "replayed": summary,
        }

    analysis = analyze_fusion_trace(trace_path, report.episode_records)
    payload = {
        "scenario": scenario,
        "forced_fusion_loader": force_fusion,
        "decoder": FUSION,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": src["checkpoints"][STEP]["sha256"],
        "mode": MODE,
        "episodes": EPISODES,
        "seed_start": SEED_START,
        "summary": summary,
        "frozen_legacy_cross_check": frozen_check,
        "diagnostics": {
            k: v for k, v in diagnostics.items()
            if k in (
                "fusion_decoder_records", "actor_non_keep_confidence_threshold",
                "exact_fusion_rule_match_rate", "exact_fusion_action_match_rate",
                "actor_override_rate", "target_fallback_rate", "selected_source_counts",
                "outcomes",
            )
        },
        "confidence_analysis": analysis,
    }
    seal(out_dir / f"{scenario}_fusion_trace.json", payload)
    return payload


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    import statistics
    finite = [float(v) for v in values]
    ordered = sorted(finite)

    def q(fraction: float) -> float:
        loc = fraction * (len(ordered) - 1)
        lo = int(loc)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] * (1 - (loc - lo)) + ordered[hi] * (loc - lo)

    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "population_std": statistics.pstdev(finite),
        "minimum": ordered[0],
        "p10": q(0.10),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "maximum": ordered[-1],
    }


def _outcome_of(record: Any) -> str:
    if getattr(record, "collision", False):
        return "collision"
    if getattr(record, "off_route", False):
        return "off_route"
    if getattr(record, "timeout", False):
        return "timeout"
    if getattr(record, "success", False):
        return "success"
    return "unknown"


def analyze_fusion_trace(trace_path: Path, episode_records: list[Any]) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("fusion trace is empty")

    outcome_by_episode: dict[int, str] = {
        int(rec.episode): _outcome_of(rec) for rec in episode_records
    }

    confidences: list[float] = []
    margins: list[float] = []
    override_confidences: list[float] = []
    override_margins: list[float] = []
    fallback_confidences: list[float] = []
    q_regret_override: list[float] = []
    near_threshold_override: list[float] = []

    by_outcome: dict[str, dict[str, list[float]]] = {}
    override_rate_by_outcome: dict[str, int] = {}
    total_by_outcome: dict[str, int] = {}

    for row in rows:
        decoder = row["fusion_decoder"]
        confidence = float(decoder["actor_selected_confidence"])
        override = bool(decoder["actor_override"])
        margin = confidence - 0.90
        confidences.append(confidence)
        margins.append(margin)
        q_values = decoder["minimum_target_twin_q"]
        actor_index = int(decoder["actor_selected_lane_index"])
        target_index = int(decoder["target_selected_lane_index"])
        actor_q = q_values[actor_index]
        target_q = q_values[target_index]
        outcome = outcome_by_episode[int(row["episode"])]

        total_by_outcome[outcome] = total_by_outcome.get(outcome, 0) + 1
        override_rate_by_outcome[outcome] = override_rate_by_outcome.get(outcome, 0) + int(override)
        by_outcome.setdefault(outcome, {"confidence": [], "margin": [], "override": []})
        by_outcome[outcome]["confidence"].append(confidence)
        by_outcome[outcome]["margin"].append(margin)
        by_outcome[outcome]["override"].append(int(override))

        if override:
            override_confidences.append(confidence)
            override_margins.append(margin)
            if actor_q is not None and target_q is not None:
                q_regret_override.append(float(actor_q) - float(target_q))
            if margin <= 0.05:
                near_threshold_override.append(confidence)
        else:
            fallback_confidences.append(confidence)

    return {
        "decision_records": len(rows),
        "episode_outcome_counts": {
            outcome: sum(1 for v in outcome_by_episode.values() if v == outcome)
            for outcome in sorted(set(outcome_by_episode.values()))
        },
        "all_decisions_confidence": _distribution(confidences),
        "all_decisions_margin_to_0_90": _distribution(margins),
        "actor_override": {
            "confidence": _distribution(override_confidences),
            "margin_to_0_90": _distribution(override_margins),
            "near_threshold_override_count": len(near_threshold_override),
            "near_threshold_margin_lt_0_05_rate": (
                len(near_threshold_override) / len(override_confidences)
                if override_confidences else None
            ),
            "actor_q_minus_target_q": _distribution(q_regret_override),
        },
        "target_fallback_confidence": _distribution(fallback_confidences),
        "override_rate_by_outcome": {
            outcome: (
                override_rate_by_outcome.get(outcome, 0) / total
                if total else None
            )
            for outcome, total in total_by_outcome.items()
        },
        "by_outcome": {
            outcome: {
                "decisions": total_by_outcome[outcome],
                "confidence": _distribution(values["confidence"]),
                "margin_to_0_90": _distribution(values["margin"]),
                "override_rate": (
                    sum(values["override"]) / len(values["override"])
                    if values["override"] else None
                ),
            }
            for outcome, values in by_outcome.items()
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--scenario", choices=SCENARIOS, action="append")
    value.add_argument("--device", default="cuda")
    value.add_argument("--skip-ablation", action="store_true")
    value.add_argument("--skip-trace", action="store_true")
    value.add_argument("--output", type=Path,
                       default=OUT / "v4_8_mechanism_supplement" / "summary.json")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    scenarios = tuple(args.scenario) if args.scenario else SCENARIOS
    sources = source_registry()
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation: dict[str, Any] = {}
    traces: dict[str, Any] = {}
    for scenario in scenarios:
        source = sources[f"v4_8__{scenario}__lr_half"]
        force_fusion = scenario == "carla"  # selector chose target_critic for carla
        if not args.skip_ablation:
            ablation[scenario] = run_decoder_ablation(
                source, scenario, force_fusion, args.device
            )
        if not args.skip_trace:
            traces[scenario] = run_fusion_trace(
                source, scenario, force_fusion, args.device, out_dir
            )

    payload = {
        "schema_version": "topo-scene-v4.8.mechanism-supplement/v1",
        "computed_from_real_rollout": True,
        "fabricated_values": False,
        "no_training": True,
        "no_new_network_structure": True,
        "checkpoint": "v4_8/{scenario}/lr_half seed0 final_model.zip",
        "traffic": "frozen legacy (training) replay, 100 episodes, seeds 10000..10099",
        "decoder_ablation": ablation,
        "confidence_trace": traces,
    }
    seal(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
