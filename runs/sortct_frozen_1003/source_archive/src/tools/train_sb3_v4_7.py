"""Train v4.7 while reusing the frozen v4.6 deployment workflow.

The candidate changes only the replay return estimator.  Checkpoint/decoder
calibration, selection, sealing, and final evaluation remain the audited v4.6
implementations and are adapted here without editing their source files.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch.replay_buffer_v4_7 import (
    HorizonCorrectDictNStepReplayBufferV47,
)
from configs.sb3_configs_v4_7 import (
    PARENT_CONTROL,
    TEMPORAL_GRAPH,
    V47_ALGORITHMS,
    V47_CANDIDATE,
    V47_FORMAL_ALGORITHMS,
    V47_IMPLEMENTATION_IDS,
    make_model_v4_7,
    source_action_repeat_v4_7,
)
from tools import train_sb3_v4_6 as parent
from tools.checkpoint_decoder_selector_v4_6 import (
    CANDIDATE_DECODERS,
    PARENT_DECODER,
)
from tools.checkpoint_selector_v4_4 import sha256


_ACTIVE_MODEL: Any | None = None


def _write_json_plain(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _version_payload(payload: Any) -> Any:
    value = copy.deepcopy(payload)
    if isinstance(value, dict):
        schema = value.get("schema_version")
        if isinstance(schema, str):
            for old in ("topo-scene-v4.5.", "topo-scene-v4.6."):
                if schema.startswith(old):
                    value["schema_version"] = schema.replace(
                        old, "topo-scene-v4.7.", 1
                    )
                    break
    return value


def _return_estimator_diagnostics(algorithm: str) -> dict[str, Any]:
    if _ACTIVE_MODEL is None:
        raise RuntimeError("v4.7 model was not captured")
    replay = _ACTIVE_MODEL.replay_buffer
    if algorithm == V47_CANDIDATE:
        if not isinstance(replay, HorizonCorrectDictNStepReplayBufferV47):
            raise TypeError("v4.7 candidate did not use its sealed replay buffer")
        value = replay.return_estimator_diagnostics()
        value.update(
            {
                "algorithm": algorithm,
                "implementation_id": V47_IMPLEMENTATION_IDS[algorithm],
                "return_estimator_changed": True,
                "diagnostics_computed_from_runtime_buffer": True,
            }
        )
        return value

    return {
        "schema_version": "topo-scene-v4.7.return-estimator-diagnostics/v1",
        "algorithm": algorithm,
        "implementation_id": V47_IMPLEMENTATION_IDS[algorithm],
        "return_estimator_changed": False,
        "n_step": int(getattr(replay, "n_steps", 4)),
        "gamma": float(getattr(replay, "gamma", 0.99)),
        "horizon_correct_bootstrap": False,
        "bootstrap_discount": "single_gamma_source_equivalent",
        "reward_accumulation": "sum_gamma_power_i_reward",
        "sampled_horizon_count": None,
        "mean_sampled_actual_horizon": None,
        "short_horizon_sample_rate": None,
        "diagnostics_computed_from_runtime_buffer": True,
        "parent_buffer_instrumentation_changed": False,
        "source_timeout_semantics_changed": False,
        "one_step_next_observation_changed": False,
        "episode_end_transition_duplication_changed": False,
    }


def _write_json_v4_7(path: Path, payload: Any) -> None:
    value = _version_payload(payload)
    if isinstance(value, dict):
        if path.name == "arguments.json":
            algorithm = value.get("requested_raw_steps", {}).get("algo")
            fidelity = dict(value.get("implementation_fidelity", {}))
            fidelity.pop("v4_5_isolated_files", None)
            fidelity.pop("v4_6_isolated_files", None)
            fidelity.update(
                {
                    "implementation_id": V47_IMPLEMENTATION_IDS[algorithm],
                    "v4_7_isolated_files": True,
                    "single_change": (
                        "horizon_correct_16_step_terminal_credit"
                        if algorithm == V47_CANDIDATE
                        else "none_frozen_control"
                    ),
                    "return_estimator_changed": algorithm == V47_CANDIDATE,
                    "n_step": 16 if algorithm == V47_CANDIDATE else 4,
                    "bootstrap_discount": (
                        "gamma_power_actual_horizon"
                        if algorithm == V47_CANDIDATE
                        else "single_gamma_source_equivalent"
                    ),
                    "checkpoint_decoder_candidate_pairs": (
                        2 if algorithm == TEMPORAL_GRAPH else 4
                    ),
                    "selector_changed": False,
                    "decoder_changed": False,
                }
            )
            value["implementation_fidelity"] = fidelity
        elif path.name == "check_result.json":
            algorithm = value.get("algorithm")
            if algorithm != TEMPORAL_GRAPH:
                action = np.asarray(value.get("action", [])).reshape(-1)
                value["exact_hybrid_lane_code"] = bool(
                    len(action) >= 2
                    and float(action[1]) in (-1.0, 0.0, 1.0)
                )
                value["checkpoint_decoder_candidate_pairs"] = 4
            value["return_estimator"] = _return_estimator_diagnostics(algorithm)
        elif path.name == "training_diagnostics.json":
            algorithm = value.get("algorithm")
            diagnostics = _return_estimator_diagnostics(algorithm)
            value["return_estimator"] = diagnostics
            _write_json_plain(path.parent / "return_estimator_diagnostics.json", diagnostics)
    _write_json_plain(path, value)


def _write_json_plain_v4_7(path: Path, payload: Any) -> None:
    _write_json_plain(path, _version_payload(payload))


def _method_hyperparameters_v4_7(algorithm: str) -> dict[str, Any]:
    common = {
        "reward_unchanged": True,
        "checkpoint_candidates": ["highest_training_success", "exact_final"],
        "checkpoint_calibration_episodes": 12,
        "checkpoint_calibration_partition": "train",
        "validation_used_for_checkpoint_selection": False,
        "selector_changed": False,
        "decoder_changed": False,
    }
    if algorithm == V47_CANDIDATE:
        return {
            **common,
            "single_change": "horizon_correct_16_step_terminal_credit",
            "return_estimator_changed": True,
            "parent_n_step": 4,
            "n_step": 16,
            "reward_accumulation": "sum_gamma_power_i_reward",
            "actual_horizon": "min_16_or_first_episode_boundary",
            "bootstrap_discount": "gamma_power_actual_horizon",
            "deployment_decoder_candidates": list(CANDIDATE_DECODERS),
            "checkpoint_decoder_candidate_pairs": 4,
            "actor_encoder_detach": True,
            "slot_balance_coef": 0.01,
            "hybrid_lane_actions": [-1, 0, 1],
            "lane_action_mask": True,
            "lane_conditioned_speed": True,
            "critic_lane_one_hot": True,
            "speed_target_entropy": -1.0,
            "lane_entropy_scale": 0.0,
            "actor_non_keep_confidence_threshold": 0.90,
        }
    if algorithm == PARENT_CONTROL:
        return {
            **common,
            "single_change": "none_matched_v4_6_parent_control",
            "return_estimator_changed": False,
            "n_step": 4,
            "bootstrap_discount": "single_gamma_source_equivalent",
            "deployment_decoder_candidates": list(CANDIDATE_DECODERS),
            "checkpoint_decoder_candidate_pairs": 4,
            "actor_encoder_detach": True,
            "slot_balance_coef": 0.01,
            "lane_entropy_scale": 0.0,
            "speed_target_entropy": -1.0,
        }
    if algorithm == TEMPORAL_GRAPH:
        return {
            **common,
            "single_change": "none_temporal_graph_control",
            "return_estimator_changed": False,
            "n_step": 4,
            "bootstrap_discount": "single_gamma_source_equivalent",
            "deployment_decoder_candidates": [PARENT_DECODER],
            "checkpoint_decoder_candidate_pairs": 2,
            "comparator_implementation_unchanged": True,
            "actor_encoder_detach": True,
            "slot_balance_coef": 0.0,
        }
    raise ValueError(f"unsupported v4.7 algorithm {algorithm!r}")


def _validate_formal_unlock_v4_7(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.evaluation_split != "test":
        if args.formal_unlock_receipt is not None:
            raise ValueError("formal unlock receipt may only be used with the test split")
        return None
    if args.algo not in V47_FORMAL_ALGORITHMS:
        raise ValueError("formal test permits only the frozen v4.7 method pair")
    if not args.experiment_contract_sha256 or not args.implementation_freeze_sha256:
        raise ValueError("formal test requires experiment and implementation hashes")
    if args.formal_unlock_receipt is None:
        raise ValueError("formal test locked: missing promotion gate receipt")
    receipt_path = args.formal_unlock_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "topo-scene-v4.7.promotion-gate/v1",
        "decision": "pass",
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"formal test locked: receipt {key!r} does not match")
    if receipt.get("formal_algorithms") != list(V47_FORMAL_ALGORITHMS):
        raise ValueError("formal test locked: receipt method pair drifted")
    return {
        "path": str(receipt_path),
        "sha256": sha256(receipt_path),
        "promotion_results_sha256": receipt.get("promotion_results_sha256"),
    }


def _calibration_decoders_v4_7(algorithm: str) -> tuple[str, ...]:
    return (
        tuple(CANDIDATE_DECODERS)
        if algorithm in (PARENT_CONTROL, V47_CANDIDATE)
        else (PARENT_DECODER,)
    )


def _make_model_and_capture(*args: Any, **kwargs: Any):
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = make_model_v4_7(*args, **kwargs)
    return _ACTIVE_MODEL


def main(
    argv: list[str] | None = None,
    *,
    env_factory: Callable[..., Any] | None = None,
    default_output_dir: Path | None = None,
    require_paper_evaluation_contract: bool = False,
) -> int:
    """Adapt the frozen v4.6 orchestration and restore it after every call."""

    global _ACTIVE_MODEL
    _ACTIVE_MODEL = None
    originals = {
        "V46_ALGORITHMS": parent.V46_ALGORITHMS,
        "V46_IMPLEMENTATION_IDS": parent.V46_IMPLEMENTATION_IDS,
        "make_model_v4_6": parent.make_model_v4_6,
        "source_action_repeat_v4_6": parent.source_action_repeat_v4_6,
        "_method_hyperparameters_v4_6": parent._method_hyperparameters_v4_6,
        "_validate_formal_unlock_v4_6": parent._validate_formal_unlock_v4_6,
        "_calibration_decoders": parent._calibration_decoders,
        "_write_json_v4_6": parent._write_json_v4_6,
        "_write_json_plain": parent._write_json_plain,
    }
    parent.V46_ALGORITHMS = V47_ALGORITHMS
    parent.V46_IMPLEMENTATION_IDS = V47_IMPLEMENTATION_IDS
    parent.make_model_v4_6 = _make_model_and_capture
    parent.source_action_repeat_v4_6 = source_action_repeat_v4_7
    parent._method_hyperparameters_v4_6 = _method_hyperparameters_v4_7
    parent._validate_formal_unlock_v4_6 = _validate_formal_unlock_v4_7
    parent._calibration_decoders = _calibration_decoders_v4_7
    parent._write_json_v4_6 = _write_json_v4_7
    parent._write_json_plain = _write_json_plain_v4_7
    try:
        return parent.main(
            argv,
            env_factory=env_factory,
            default_output_dir=default_output_dir,
            require_paper_evaluation_contract=require_paper_evaluation_contract,
        )
    finally:
        for name, value in originals.items():
            setattr(parent, name, value)
        _ACTIVE_MODEL = None


if __name__ == "__main__":
    raise SystemExit(main())
