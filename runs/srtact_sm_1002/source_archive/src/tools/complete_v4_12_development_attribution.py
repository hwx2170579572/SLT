"""Seal the complete v4.12 development-failure attribution package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ATTRIBUTION_ROOT = (
    ROOT
    / "results_topo_v4_12_dev"
    / "development"
    / "attribution"
    / "d3_roundabout_regression"
)
BASE_ATTRIBUTION = ATTRIBUTION_ROOT / "attribution.json"
ZERO_PRIOR_MATRIX = ATTRIBUTION_ROOT / "closed_loop_lane_prior_matrix.json"
HIGH_PRIOR_EVALUATION = (
    ATTRIBUTION_ROOT
    / "closed_loop_lane_prior_0_10_D3"
    / "closed_loop_evaluation.json"
)
GRADIENT_DIAGNOSTIC = ATTRIBUTION_ROOT / "lane_speed_gradient_coupling.json"
SENSITIVITY_PLAN = (
    ROOT / "experiments" / "topo_scene_v4" / "V4_12_LANE_PRIOR_SENSITIVITY_PLAN.md"
)
DEFAULT_OUTPUT = ATTRIBUTION_ROOT / "complete_attribution.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _moment(
    attribution: Mapping[str, Any], key: str, moment: str
) -> tuple[float, float, float]:
    value = attribution["training_diagnostics"][key]
    return (
        float(value["parent"][moment]),
        float(value["candidate"][moment]),
        float(value["candidate_minus_parent"][moment]),
    )


def _matrix_effects(matrix: Mapping[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for row in matrix["rows"]:
        _require(row["status"] == "accepted", f"matrix cell failed: {row['cell']}")
        rows[str(row["cell"])] = {
            "scenario": row["scenario"],
            "source_success_rate": row["source_summary"]["success_rate"],
            "source_collision_rate": row["source_summary"]["collision_rate"],
            "lane_prior_0_success_rate": row["ablation_summary"]["success_rate"],
            "lane_prior_0_collision_rate": row["ablation_summary"]["collision_rate"],
            "success_rate_delta": row["ablation_minus_source"]["success_rate"],
            "collision_rate_delta": row["ablation_minus_source"]["collision_rate"],
            "evaluation_sha256": row["evaluation_sha256"],
        }
    _require(set(rows) == {"D1", "D2", "D3", "D4"}, "matrix cell set drifted")
    return rows


def build_complete_attribution() -> dict[str, Any]:
    paths = {
        "base_attribution": BASE_ATTRIBUTION,
        "zero_prior_matrix": ZERO_PRIOR_MATRIX,
        "high_prior_evaluation": HIGH_PRIOR_EVALUATION,
        "gradient_diagnostic": GRADIENT_DIAGNOSTIC,
        "sensitivity_plan": SENSITIVITY_PLAN,
    }
    for name, path in paths.items():
        _require(path.is_file(), f"missing {name}: {path}")
    base = _load_json(BASE_ATTRIBUTION)
    matrix = _load_json(ZERO_PRIOR_MATRIX)
    high = _load_json(HIGH_PRIOR_EVALUATION)
    gradient = _load_json(GRADIENT_DIAGNOSTIC)
    _require(matrix["complete"] is True, "zero-prior matrix is incomplete")
    _require(matrix["accepted_cells"] == 4, "zero-prior matrix is not 4/4")
    _require(matrix["formal_test_accessed"] is False, "matrix accessed formal")
    _require(high["formal_test_accessed"] is False, "sensitivity accessed formal")
    _require(high["intervention"]["intervened_value"] == 0.10, "high prior drift")
    _require(high["intervention"]["learned_tensor_state_unchanged"] is True, "weights changed")
    _require(gradient["formal_test_accessed"] is False, "gradient used formal")
    _require(gradient["optimizer_step_performed"] is False, "gradient diagnostic trained")
    zero_effects = _matrix_effects(matrix)
    high_summary = high["ablation_summary"]
    high_pass = bool(
        float(high_summary["success_rate"]) >= 0.85
        and float(high_summary["collision_rate"]) <= 0.15
        and (
            float(high["ablation_minus_source"]["success_rate"]) >= 0.05
            or float(high["ablation_minus_source"]["collision_rate"]) <= -0.05
        )
    )
    _require(high_pass is False, "high-prior sensitivity unexpectedly passed")
    behavior = base["behavior"]
    spread_mean = _moment(base, "support/component_speed_spread", "mean")
    spread_last = _moment(base, "support/component_speed_spread", "last")
    lane_entropy_last = _moment(base, "hybrid/lane_entropy", "last")
    risk_error_last = _moment(
        base, "risk/collision_probability_absolute_error", "last"
    )
    trunk = gradient["aggregate"]["latent_trunk"]
    lane_norm = float(trunk["lane_nll_gradient_norm"]["mean"])
    speed_norm = float(trunk["speed_nll_gradient_norm"]["mean"])
    _require(speed_norm > 0.0, "speed gradient norm is zero")
    hypotheses = {
        "H1_remove_or_weaken_deployment_lane_prior": {
            "decision": "rejected",
            "evidence": (
                "coef=0 changed D2 by -0.15/+0.15 and D3 by -0.20/+0.20; "
                "coef=0.10 left D3 at 0.80/0.20"
            ),
        },
        "H2_increase_global_deployment_lane_prior": {
            "decision": "rejected",
            "evidence": "precommitted 0.10 D3 gate failed with zero net outcome change",
        },
        "H3_cross_rotation_augmentation_caused_D3_regression": {
            "decision": "rejected",
            "evidence": (
                "roundabout_medium already enabled the same random rotation in "
                "v4.11 and v4.12; the new augmentation scope was Cross only"
            ),
        },
        "H4_shared_scene_encoder_support_gradient_interference": {
            "decision": "rejected",
            "evidence": "real-state gradient norms are exactly zero in the scene encoder",
        },
        "H5_lane_speed_support_coupling_in_actor_latent_trunk": {
            "decision": "supported_for_next_iteration",
            "evidence": (
                "both losses update latent_pi; lane/speed norm ratio is "
                f"{lane_norm / speed_norm:.6f}, and active-coordinate opposing-sign "
                f"fraction is {float(trunk['opposing_sign_fraction']['mean']):.6f}"
            ),
        },
    }
    selected_method = {
        "version": "v4.13",
        "name": "gradient_isolated_tempered_joint_support_prcr",
        "parent": "v4.12_augmented_joint_support_prcr",
        "global_model_change": True,
        "scenario_conditioned": False,
        "lane_support_gradient_path": (
            "same_deployed_lane_head_from_detached_actor_latent"
        ),
        "lane_support_scale": 0.25,
        "lane_prior_coef": 0.05,
        "replay_support_outer_coef": 0.05,
        "conditional_speed_support_unchanged": True,
        "cross_rotation_augmentation_unchanged": True,
        "inference_score_equation_unchanged_from_v4_12": True,
        "intended_effect": (
            "preserve lane-head calibration while preventing lane imitation from "
            "dominating the latent trunk used by learned speed components"
        ),
        "not_claimed_before_fresh_training": (
            "the diagnostic supports a mechanism hypothesis, not an outcome claim"
        ),
    }
    return {
        "schema_version": "topo-scene-v4.12.complete-development-attribution/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_kind": "complete_multi_intervention_and_gradient_attribution",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "development_gate_decision": "fail",
        "failed_cell": "D3",
        "formal_test_accessed": False,
        "formal_test_unlocked": False,
        "cross_version_parent_episode_comparison_exact_seed_paired": False,
        "v4_12_development_outcomes": {
            cell: {
                "scenario": value["scenario"],
                "success_rate": value["source_success_rate"],
                "collision_rate": value["source_collision_rate"],
            }
            for cell, value in zero_effects.items()
        },
        "same_checkpoint_same_seed_lane_prior_zero_matrix": zero_effects,
        "same_checkpoint_same_seed_lane_prior_0_10_D3": {
            "source_summary": high["source_summary"],
            "sensitivity_summary": high_summary,
            "sensitivity_minus_source": high["ablation_minus_source"],
            "precommitted_gate_passed": high_pass,
        },
        "observed_behavior_shift_candidate_minus_v4_11_D3_aggregate_unpaired": {
            "non_keep_command_rate": behavior["candidate_minus_parent"][
                "non_keep_command_rate"
            ],
            "lane_change_applied_rate": behavior["candidate_minus_parent"][
                "lane_change_applied_rate"
            ],
            "valid_route_intent_match_rate": behavior["candidate_minus_parent"][
                "valid_route_intent_match_rate"
            ],
            "mean_actual_speed_mps": behavior["candidate_minus_parent"][
                "mean_actual_speed_mps"
            ],
            "comparison_is_aggregate_not_exact_seed_paired": True,
        },
        "training_evidence": {
            "component_speed_spread_mean_parent_candidate_delta": spread_mean,
            "component_speed_spread_last_parent_candidate_delta": spread_last,
            "candidate_over_parent_spread_ratio_mean": spread_mean[1] / spread_mean[0],
            "candidate_over_parent_spread_ratio_last": spread_last[1] / spread_last[0],
            "lane_entropy_last_parent_candidate_delta": lane_entropy_last,
            "collision_probability_error_last_parent_candidate_delta": risk_error_last,
            "collision_label_definition_unchanged": base[
                "collision_label_diagnostics"
            ]["same_observed_label_definition"],
        },
        "real_state_gradient_evidence": {
            "samples": gradient["samples"],
            "minibatches": gradient["minibatches"],
            "lane_nll_latent_trunk_gradient_norm_mean": lane_norm,
            "speed_nll_latent_trunk_gradient_norm_mean": speed_norm,
            "lane_to_speed_gradient_norm_ratio": lane_norm / speed_norm,
            "latent_trunk_gradient_cosine_mean": trunk["cosine"]["mean"],
            "latent_trunk_opposing_sign_fraction_mean": trunk[
                "opposing_sign_fraction"
            ]["mean"],
            "shared_scene_encoder_detached": gradient[
                "architecture_finding"
            ]["shared_scene_encoder_detached_from_both_support_losses"],
            "coupling_location": gradient["architecture_finding"][
                "coupling_location"
            ],
        },
        "hypothesis_triage": hypotheses,
        "selected_next_method": selected_method,
        "forbidden_mechanisms_for_next_method": {
            "kinematic_safety_projection": True,
            "ttc_or_headway_threshold": True,
            "geometry_unsafe_label": True,
            "lane_change_veto": True,
            "actor_confidence_gate": True,
            "safety_shield_or_rule_fallback": True,
            "semantic_tie_override": True,
            "post_decoder_action_rewrite": True,
            "scenario_conditioned_inference_rule": True,
        },
        "source_artifacts": {
            name: {"path": _relative(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    gradient = payload["real_state_gradient_evidence"]
    training = payload["training_evidence"]
    zero = payload["same_checkpoint_same_seed_lane_prior_zero_matrix"]
    return "\n".join(
        [
            "# Complete v4.12 development-failure attribution",
            "",
            "All evidence is from validation/development artifacts. Formal test remains locked and untouched.",
            "",
            "## Closed-loop interventions",
            "",
            f"- Removing the global lane prior: D1 `{zero['D1']['success_rate_delta']:+.2f}/{zero['D1']['collision_rate_delta']:+.2f}`, D2 `{zero['D2']['success_rate_delta']:+.2f}/{zero['D2']['collision_rate_delta']:+.2f}`, D3 `{zero['D3']['success_rate_delta']:+.2f}/{zero['D3']['collision_rate_delta']:+.2f}`, D4 `{zero['D4']['success_rate_delta']:+.2f}/{zero['D4']['collision_rate_delta']:+.2f}` (success/collision deltas).",
            "- Raising the same global prior from `0.05` to `0.10` leaves D3 at `0.80/0.20`; its precommitted gate fails.",
            "- Therefore neither deleting nor strengthening the deployment prior explains or fixes the regression.",
            "",
            "## Gradient localization",
            "",
            f"- Lane-NLL latent-trunk gradient norm mean: `{gradient['lane_nll_latent_trunk_gradient_norm_mean']:.6f}`.",
            f"- Speed-NLL latent-trunk gradient norm mean: `{gradient['speed_nll_latent_trunk_gradient_norm_mean']:.6f}`.",
            f"- Lane/speed norm ratio: `{gradient['lane_to_speed_gradient_norm_ratio']:.6f}`; opposing active-coordinate fraction: `{gradient['latent_trunk_opposing_sign_fraction_mean']:.6f}`.",
            "- Both support losses are already detached from the scene encoder. Their coupling is specifically in `actor.latent_pi`.",
            "",
            "## Corroborating training evidence",
            "",
            f"- Candidate/parent speed-component spread ratio: mean `{training['candidate_over_parent_spread_ratio_mean']:.6f}`, last `{training['candidate_over_parent_spread_ratio_last']:.6f}`.",
            "- v4.12 also shifts toward more lane changes and lower route-intent agreement; cross-version episode outcomes are not called paired because evaluation seeds differ.",
            "",
            "## v4.13 route",
            "",
            "Use the same deployed lane head, but compute the auxiliary replay-lane NLL from a detached actor latent so that this loss updates the lane head without updating the latent trunk shared with speed components. Temper its global scale from `1.0` to `0.25`; retain the learned lane prior at `0.05`. This is a training/model change only—no rule, threshold, veto, safety shield, kinematic projection, or action rewrite.",
            "",
        ]
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = build_complete_attribution()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = output.with_name("COMPLETE_ATTRIBUTION_V4_12.md")
    markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": _sha256(output),
                "markdown": str(markdown),
                "markdown_sha256": _sha256(markdown),
                "selected_next_method": payload["selected_next_method"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_complete_attribution"]
