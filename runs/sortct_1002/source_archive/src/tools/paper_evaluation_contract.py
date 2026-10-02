"""Strict environment and provenance contracts for paper evaluations."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Any, Callable

import gymnasium as gym
import numpy as np

from envs.sumo.paper_env import PaperSumoSceneEnv
from envs.sumo.paper_scenario_registry import get_paper_scenario_spec
from envs.sumo.ppo_env import PaperPpoCarlaEnv, PaperPpoRgbEnv

if TYPE_CHECKING:
    from algos.sb3_torch.evaluation import DetailedEvaluationReport


CONTRACT_VERSION = 1


def expected_environment_asset_provenance(scenario: str) -> dict[str, Any]:
    """Describe the audited asset class without mislabelling CARLA as released SUMO."""

    specification = get_paper_scenario_spec(scenario)
    if specification.released_assets:
        return {
            "uses_released_assets": True,
            "scenario_asset_source": "authors_release_v1.0.0",
            "source_observation_contract": specification.source_observation_contract,
            "scenario_evidence_class": "released_sumo_source_scenario",
        }
    if scenario != "carla" or specification.source_observation_contract != "carla":
        raise ValueError(
            f"No controlled reconstructed-asset contract exists for {scenario!r}"
        )
    return {
        "uses_released_assets": False,
        "scenario_asset_source": "carla_source_waypoint_reconstruction",
        "source_observation_contract": "carla",
        "scenario_evidence_class": "controlled_extension_not_reported_in_paper",
    }


def validate_environment_asset_provenance(
    env: gym.Env, scenario: str
) -> dict[str, Any]:
    """Require the exact released/reconstructed asset class declared per scenario."""

    expected = expected_environment_asset_provenance(scenario)
    actual_released = bool(getattr(env, "uses_released_assets", False))
    if actual_released != expected["uses_released_assets"]:
        raise ValueError(
            f"Environment uses_released_assets={actual_released}, expected "
            f"{expected['uses_released_assets']} for {scenario!r}"
        )
    environment_specification = getattr(env, "_paper_specification", None)
    actual_source_contract = getattr(
        environment_specification, "source_observation_contract", None
    )
    if actual_source_contract != expected["source_observation_contract"]:
        raise ValueError(
            f"Environment source_observation_contract={actual_source_contract!r}, "
            f"expected {expected['source_observation_contract']!r} for {scenario!r}"
        )
    return expected


def callable_name(value: Any) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


def make_injected_evaluation_env(
    env_factory: Callable[..., Any], args: Any
) -> Any:
    """Use the injected scene factory for every evaluation phase."""

    return env_factory(args, evaluation=True)


def expected_traffic_partition(traffic_protocol: str) -> str:
    if traffic_protocol == "source_all":
        return "all"
    if traffic_protocol == "frozen_80_20":
        return "evaluation"
    raise ValueError(f"Unsupported traffic protocol {traffic_protocol!r}")


def expected_environment_class(algorithm: str, scenario: str) -> type[gym.Env]:
    if algorithm == "ppo":
        return PaperPpoCarlaEnv if scenario == "carla" else PaperPpoRgbEnv
    return PaperSumoSceneEnv


def make_paper_evaluation_env(
    arguments: dict[str, Any], *, traffic_protocol: str
) -> PaperSumoSceneEnv:
    """Construct the source-compatible evaluation environment from run arguments."""

    algorithm = str(arguments["algo"])
    scenario = str(arguments["scenario"])
    partition = expected_traffic_partition(traffic_protocol)
    common = {
        "scenario": scenario,
        "action_repeat": int(arguments["action_repeat"]),
        "reward_discount": float(arguments["discount"]),
        "ego_control_profile": str(
            arguments.get("ego_control_profile", "direct")
        ),
        "traffic_partition": partition,
        "episode_limit_profile": str(
            arguments.get("episode_limit_profile", "source")
        ),
    }
    if algorithm == "ppo":
        env_class = PaperPpoCarlaEnv if scenario == "carla" else PaperPpoRgbEnv
        return env_class(training=False, **common)
    return PaperSumoSceneEnv(
        neighbors=int(arguments["neighbors"]),
        history_steps=int(arguments["history_steps"]),
        path_length=int(arguments["path_length"]),
        include_state_lstm=algorithm == "sac",
        state_lstm_only=algorithm == "sac",
        **common,
    )


def _array_digest(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest().upper()


def _finite_extreme(value: np.ndarray, reducer: Any) -> float | None:
    finite = value[np.isfinite(value)]
    return float(reducer(finite)) if finite.size else None


def space_signature(space: gym.spaces.Space[Any]) -> dict[str, Any]:
    """Return a compact, JSON-safe signature for a Gymnasium space."""

    signature: dict[str, Any] = {
        "type": f"{type(space).__module__}.{type(space).__qualname__}",
        "shape": list(space.shape) if space.shape is not None else None,
        "dtype": str(space.dtype) if getattr(space, "dtype", None) is not None else None,
    }
    if isinstance(space, gym.spaces.Box):
        low = np.asarray(space.low)
        high = np.asarray(space.high)
        signature.update(
            {
                "low_sha256": _array_digest(low),
                "high_sha256": _array_digest(high),
                "low_min_finite": _finite_extreme(low, np.min),
                "low_max_finite": _finite_extreme(low, np.max),
                "high_min_finite": _finite_extreme(high, np.min),
                "high_max_finite": _finite_extreme(high, np.max),
            }
        )
    elif isinstance(space, gym.spaces.Dict):
        signature["spaces"] = {
            key: space_signature(child)
            for key, child in sorted(space.spaces.items())
        }
    elif isinstance(space, gym.spaces.Discrete):
        signature.update({"n": int(space.n), "start": int(space.start)})
    elif isinstance(space, gym.spaces.MultiDiscrete):
        signature["nvec"] = np.asarray(space.nvec).astype(int).tolist()
    elif isinstance(space, gym.spaces.MultiBinary):
        signature["n"] = np.asarray(space.n).astype(int).tolist()
    return signature


def validate_model_environment_spaces(model: Any, env: gym.Env) -> dict[str, Any]:
    """Fail before inference if the loaded policy and environment disagree."""

    # This is intentionally the same equality contract used by SB3's
    # check_for_correct_spaces(), kept local so the lightweight paper runtime
    # audit does not import TensorFlow through optional tensorboard plugins.
    if model.observation_space != env.observation_space:
        raise ValueError(
            "Observation spaces do not match: "
            f"{model.observation_space} != {env.observation_space}"
        )
    if model.action_space != env.action_space:
        raise ValueError(
            f"Action spaces do not match: {model.action_space} != {env.action_space}"
        )
    environment_observation = space_signature(env.observation_space)
    model_observation = space_signature(model.observation_space)
    environment_action = space_signature(env.action_space)
    model_action = space_signature(model.action_space)
    if environment_observation != model_observation:
        raise ValueError("Observation-space signatures differ after SB3 equality check")
    if environment_action != model_action:
        raise ValueError("Action-space signatures differ after SB3 equality check")
    return {
        "model_environment_spaces_match": True,
        "environment_observation_space": environment_observation,
        "model_observation_space": model_observation,
        "environment_action_space": environment_action,
        "model_action_space": model_action,
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator


def _assert_rate(label: str, actual: float, expected: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}={actual} cannot be recomputed as {expected}")


def build_evaluation_provenance(
    *,
    model: Any,
    env: gym.Env,
    report: "DetailedEvaluationReport",
    algorithm: str,
    scenario: str,
    traffic_protocol: str,
    episode_limit_profile: str,
    evaluation_seed_start: int,
    environment_factory: str,
    space_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a completed evaluation and return its persisted provenance."""

    expected_class = expected_environment_class(algorithm, scenario)
    if not isinstance(env, expected_class):
        raise TypeError(
            f"Paper evaluation requires {expected_class.__module__}."
            f"{expected_class.__qualname__}, got {type(env).__module__}."
            f"{type(env).__qualname__}"
        )
    partition = expected_traffic_partition(traffic_protocol)
    actual_partition = getattr(env, "_traffic_partition", None)
    if actual_partition != partition:
        raise ValueError(
            f"Traffic partition is {actual_partition!r}, expected {partition!r}"
        )
    actual_limit_profile = getattr(env, "_episode_limit_profile", None)
    if actual_limit_profile != episode_limit_profile:
        raise ValueError(
            f"Episode-limit profile is {actual_limit_profile!r}, expected "
            f"{episode_limit_profile!r}"
        )
    asset_provenance = validate_environment_asset_provenance(env, scenario)

    space_contract = space_contract or validate_model_environment_spaces(model, env)
    records = list(report.episode_records)
    episodes = int(report.summary.episodes)
    if len(records) != episodes:
        raise ValueError(f"Expected {episodes} episode records, found {len(records)}")
    if [record.episode for record in records] != list(range(episodes)):
        raise ValueError("Episode indices are not contiguous from zero")
    expected_seeds = list(range(evaluation_seed_start, evaluation_seed_start + episodes))
    if [record.seed for record in records] != expected_seeds:
        raise ValueError("Evaluation seeds do not match the declared deterministic sequence")
    variants = [record.traffic_variant for record in records]
    if any(not isinstance(variant, str) or not variant for variant in variants):
        raise ValueError("Every paper evaluation episode must record traffic_variant")

    _assert_rate(
        "success_rate",
        report.summary.success_rate,
        _rate(sum(record.success for record in records), episodes),
    )
    _assert_rate(
        "collision_rate",
        report.summary.collision_rate,
        _rate(sum(record.collision for record in records), episodes),
    )
    _assert_rate(
        "off_route_rate",
        report.summary.off_route_rate,
        _rate(sum(record.off_route for record in records), episodes),
    )
    _assert_rate(
        "timeout_rate",
        report.summary.timeout_rate,
        _rate(sum(record.timeout for record in records), episodes),
    )

    environment_class = f"{type(env).__module__}.{type(env).__qualname__}"
    return {
        "contract_version": CONTRACT_VERSION,
        "validated": True,
        "environment_factory": environment_factory,
        "environment_class": environment_class,
        "expected_environment_class": (
            f"{expected_class.__module__}.{expected_class.__qualname__}"
        ),
        "algorithm": algorithm,
        "scenario": scenario,
        "traffic_protocol": traffic_protocol,
        "traffic_partition": partition,
        "episode_limit_profile": episode_limit_profile,
        **asset_provenance,
        "evaluation_seed_start": evaluation_seed_start,
        "episodes": episodes,
        "traffic_variant_episode_count": len(variants),
        "traffic_variants_observed": sorted(set(variants)),
        **space_contract,
    }


def saved_evaluation_contract_errors(
    detailed: dict[str, Any],
    *,
    algorithm: str,
    scenario: str,
    traffic_protocol: str,
    episode_limit_profile: str,
    episodes: int,
    evaluation_seed_start: int,
) -> list[str]:
    """Return every hard acceptance failure in a saved detailed evaluation."""

    errors: list[str] = []
    provenance = detailed.get("evaluation_provenance")
    if not isinstance(provenance, dict):
        return ["missing evaluation_provenance"]
    expected_class = expected_environment_class(algorithm, scenario)
    expected_class_name = f"{expected_class.__module__}.{expected_class.__qualname__}"
    expected_partition = expected_traffic_partition(traffic_protocol)
    expected_asset_provenance = expected_environment_asset_provenance(scenario)
    expected_fields = {
        "contract_version": CONTRACT_VERSION,
        "validated": True,
        "environment_class": expected_class_name,
        "expected_environment_class": expected_class_name,
        "algorithm": algorithm,
        "scenario": scenario,
        "traffic_protocol": traffic_protocol,
        "traffic_partition": expected_partition,
        "episode_limit_profile": episode_limit_profile,
        "uses_released_assets": expected_asset_provenance[
            "uses_released_assets"
        ],
        "evaluation_seed_start": evaluation_seed_start,
        "episodes": episodes,
        "traffic_variant_episode_count": episodes,
        "model_environment_spaces_match": True,
    }
    # Older completed released-SUMO evaluations predate the descriptive asset
    # fields and remain valid.  CARLA never had an accepted v1 evaluation under
    # the old released-assets-only gate, so its reconstruction fields are hard
    # requirements and prevent it from being reported as a released SUMO task.
    if scenario == "carla":
        expected_fields.update(expected_asset_provenance)
    for field, expected in expected_fields.items():
        if provenance.get(field) != expected:
            errors.append(
                f"evaluation_provenance.{field}={provenance.get(field)!r}, "
                f"expected {expected!r}"
            )
    for model_field, environment_field in (
        ("model_observation_space", "environment_observation_space"),
        ("model_action_space", "environment_action_space"),
    ):
        if provenance.get(model_field) != provenance.get(environment_field):
            errors.append(f"{model_field} does not match {environment_field}")

    records = detailed.get("episode_records")
    if not isinstance(records, list) or len(records) != episodes:
        errors.append(f"episode_records must contain exactly {episodes} rows")
        return errors
    if [record.get("episode") for record in records] != list(range(episodes)):
        errors.append("episode indices are not contiguous from zero")
    expected_seeds = list(range(evaluation_seed_start, evaluation_seed_start + episodes))
    if [record.get("seed") for record in records] != expected_seeds:
        errors.append("episode seed sequence does not match evaluation_seed_start")
    variants = [record.get("traffic_variant") for record in records]
    if any(not isinstance(variant, str) or not variant for variant in variants):
        errors.append("one or more episode records are missing traffic_variant")
    elif sorted(set(variants)) != provenance.get("traffic_variants_observed"):
        errors.append("traffic_variants_observed does not match episode records")

    summary = detailed.get("summary")
    if not isinstance(summary, dict):
        errors.append("missing detailed summary")
        return errors
    rate_fields = {
        "success_rate": "success",
        "collision_rate": "collision",
        "off_route_rate": "off_route",
        "timeout_rate": "timeout",
    }
    for rate_field, flag_field in rate_fields.items():
        expected_rate = sum(bool(record.get(flag_field, False)) for record in records) / episodes
        try:
            actual_rate = float(summary[rate_field])
        except (KeyError, TypeError, ValueError):
            errors.append(f"summary.{rate_field} is missing or invalid")
            continue
        if not math.isclose(actual_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
            errors.append(
                f"summary.{rate_field}={actual_rate} cannot be recomputed as "
                f"{expected_rate}"
            )
    return errors


def saved_evaluation_contract_is_valid(
    detailed: dict[str, Any],
    **kwargs: Any,
) -> bool:
    return not saved_evaluation_contract_errors(detailed, **kwargs)
