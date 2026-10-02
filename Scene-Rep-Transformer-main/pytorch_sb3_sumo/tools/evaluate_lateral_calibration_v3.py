"""Bounded validation-only diagnostic for the rejected v2 candidate.

This tool performs no training and cannot access the formal-test partition. It
applies one preregistered lateral-action scale to an existing v2 checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch.sac_v2 import SceneRepresentationSACV2
from envs.sumo.paper_env_v2 import PaperSumoSceneEnvV2
from tools.action_diagnostics_v2 import evaluate_with_action_diagnostics_v2
from tools.paper_evaluation_contract import validate_model_environment_spaces
from tools.paper_evaluation_contract_v2 import build_evaluation_provenance_v2


LOCKED_LATERAL_SCALE = 5.0
LOCKED_EPISODES = 10
LOCKED_EVALUATION_SEED_START = 20_000
ALLOWED_SCENARIOS = ("carla", "cross")

GATE_THRESHOLDS: dict[str, dict[str, float]] = {
    "carla": {
        "success_rate_min": 0.30,
        "collision_rate_max": 0.10,
        "off_route_rate_max": 0.00,
        "timeout_rate_max": 0.70,
        "lane_command_keep_rate_max": 0.90,
        "lane_change_applied_rate_min": 0.02,
    },
    "cross": {
        "success_rate_min": 0.50,
        "collision_rate_max": 0.45,
        "off_route_rate_max": 0.00,
        "timeout_rate_max": 0.30,
        "lane_command_keep_rate_max": 0.80,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrate_lateral_action(action: Any, scale: float) -> np.ndarray:
    """Scale only the last action coordinate and preserve batch shape."""

    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("lateral scale must be finite and positive")
    calibrated = np.asarray(action, dtype=np.float32).copy()
    if calibrated.ndim == 0 or calibrated.shape[-1] != 2:
        raise ValueError(
            f"Expected action shape (..., 2), got {calibrated.shape!r}"
        )
    calibrated[..., 1] = np.clip(
        calibrated[..., 1] * float(scale), -1.0, 1.0
    )
    if not np.all(np.isfinite(calibrated)):
        raise ValueError("calibrated action contains non-finite values")
    return calibrated


class LateralScalePolicyAdapter:
    """Transparent predict adapter; the frozen policy is otherwise unchanged."""

    def __init__(self, model: Any, *, scale: float) -> None:
        self.model = model
        self.scale = float(scale)
        if not math.isclose(
            self.scale, LOCKED_LATERAL_SCALE, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError(
                f"v3 diagnostic scale is locked to {LOCKED_LATERAL_SCALE}"
            )

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[Any, Any]:
        action, state = self.model.predict(
            observation, deterministic=deterministic
        )
        return calibrate_lateral_action(action, self.scale), state

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def gate_decision(
    scenario: str,
    outcomes: dict[str, Any],
    action_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    if scenario not in GATE_THRESHOLDS:
        raise ValueError(f"No v3 diagnostic gate for scenario {scenario!r}")
    rates = action_diagnostics["lane_command_rates"]
    observed = {
        "success_rate": float(outcomes["success_rate"]),
        "collision_rate": float(outcomes["collision_rate"]),
        "off_route_rate": float(outcomes["off_route_rate"]),
        "timeout_rate": float(outcomes["timeout_rate"]),
        "lane_command_keep_rate": float(rates["keep"]),
        "lane_change_applied_rate": float(
            action_diagnostics["lane_change_applied_rate"]
        ),
    }
    finite = all(math.isfinite(value) for value in observed.values())
    thresholds = GATE_THRESHOLDS[scenario]
    checks: dict[str, bool] = {"finite": finite}
    for name, threshold in thresholds.items():
        metric, direction = name.rsplit("_", 1)
        if direction == "min":
            checks[name] = finite and observed[metric] >= threshold
        elif direction == "max":
            checks[name] = finite and observed[metric] <= threshold
        else:  # pragma: no cover - constants above make this unreachable
            raise AssertionError(f"Unexpected threshold name {name!r}")
    passed = all(checks.values())
    return {
        "schema_version": "topology-temporal-v3-action-gate/v1",
        "scenario": scenario,
        "decision": "pass" if passed else "fail",
        "passed": passed,
        "thresholds": thresholds,
        "observed": observed,
        "checks": checks,
        "formal_test_unlocked": False,
        "on_fail": (
            "stop_without_cross_or_training"
            if scenario == "carla"
            else "reject_v3_calibration"
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=ALLOWED_SCENARIOS)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lateral-scale", type=float, default=LOCKED_LATERAL_SCALE)
    parser.add_argument("--episodes", type=int, default=LOCKED_EPISODES)
    parser.add_argument(
        "--evaluation-seed-start",
        type=int,
        default=LOCKED_EVALUATION_SEED_START,
    )
    parser.add_argument("--contract-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.episodes != LOCKED_EPISODES:
        raise ValueError(f"episodes are locked to {LOCKED_EPISODES}")
    if args.evaluation_seed_start != LOCKED_EVALUATION_SEED_START:
        raise ValueError(
            f"evaluation seed start is locked to {LOCKED_EVALUATION_SEED_START}"
        )
    if not math.isclose(
        args.lateral_scale, LOCKED_LATERAL_SCALE, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(f"lateral scale is locked to {LOCKED_LATERAL_SCALE}")

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    actual_model_sha256 = sha256_file(model_path)
    if actual_model_sha256 != args.expected_model_sha256.lower():
        raise ValueError(
            "source model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, got {actual_model_sha256}"
        )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = PaperSumoSceneEnvV2(
        scenario=args.scenario,
        history_steps=10,
        neighbors=5,
        path_length=10,
        action_repeat=3,
        reward_discount=0.99,
        ego_control_profile="direct",
        include_state_lstm=False,
        state_lstm_only=False,
        traffic_partition="validation",
        episode_limit_profile="source",
        render_mode=None,
    )
    started = time.perf_counter()
    try:
        model = SceneRepresentationSACV2.load(
            model_path, env=env, device=args.device
        )
        space_contract = validate_model_environment_spaces(model, env)
        adapter = LateralScalePolicyAdapter(model, scale=args.lateral_scale)
        report, action_diagnostics = evaluate_with_action_diagnostics_v2(
            adapter,
            env,
            episodes=args.episodes,
            seed=args.evaluation_seed_start,
            trace_path=output_dir / "action_decisions.jsonl",
        )
        provenance = build_evaluation_provenance_v2(
            model=adapter,
            env=env,
            report=report,
            algorithm="topo_v3_lateral_scale5",
            scenario=args.scenario,
            traffic_protocol="frozen_60_20_20",
            evaluation_split="validation",
            episode_limit_profile="source",
            evaluation_seed_start=args.evaluation_seed_start,
            environment_factory=(
                "tools.evaluate_lateral_calibration_v3.PaperSumoSceneEnvV2"
            ),
            space_contract=space_contract,
        )
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    evaluation = {
        "schema_version": "topology-temporal-v3-calibrated-evaluation/v1",
        "computed_from_real_rollout": True,
        "fabricated_values": False,
        "evidence_role": "development_diagnostic_only",
        "scenario": args.scenario,
        "source_model": str(model_path),
        "source_model_sha256": actual_model_sha256,
        "contract_sha256": args.contract_sha256.lower(),
        "intervention": {
            "lateral_scale": LOCKED_LATERAL_SCALE,
            "lateral_clip": [-1.0, 1.0],
            "longitudinal_unchanged": True,
            "training_performed": False,
        },
        "evaluation_seed_start": args.evaluation_seed_start,
        "episodes": args.episodes,
        "elapsed_wall_seconds": elapsed,
        "formal_test_accessed": False,
        "provenance": provenance,
        **report.to_dict(),
    }
    gate = gate_decision(
        args.scenario, report.summary.to_dict(), action_diagnostics
    )
    evaluation_path = output_dir / "evaluation.json"
    diagnostics_path = output_dir / "action_diagnostics.json"
    gate_path = output_dir / "gate.json"
    _write_json(evaluation_path, evaluation)
    _write_json(diagnostics_path, action_diagnostics)
    _write_json(gate_path, gate)

    trace_path = output_dir / "action_decisions.jsonl"
    receipt = {
        "schema_version": "topology-temporal-v3-diagnostic-receipt/v1",
        "computed_from_real_rollout": True,
        "fabricated_values": False,
        "scenario": args.scenario,
        "decision": gate["decision"],
        "formal_test_unlocked": False,
        "training_jobs": 0,
        "online_scales_evaluated": [LOCKED_LATERAL_SCALE],
        "artifacts_sha256": {
            "evaluation_json": sha256_file(evaluation_path),
            "action_diagnostics_json": sha256_file(diagnostics_path),
            "action_decisions_jsonl": sha256_file(trace_path),
            "gate_json": sha256_file(gate_path),
        },
    }
    _write_json(output_dir / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "scenario": args.scenario,
                "decision": gate["decision"],
                "observed": gate["observed"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

