"""Evaluate one existing independent-v2 deployment over paired seed blocks.

This module never trains and never mutates the parent v2 run directory.  A
completed 100-episode logical test-seed block is the atomic resume unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch import evaluate_model_detailed  # noqa: E402
from envs.sumo.independent_v2_existing_models_100seeds_100episodes_v1 import (  # noqa: E402
    INDEPENDENT_V2_EXISTING_MODELS_EVALUATION_ID,
    IndependentV2ExistingModelsEvaluationEnvV1,
    IndependentV2ExistingModelsEvaluationEnvV4V1,
)
from envs.sumo.independent_v2_existing_3methods_100seeds_100episodes_v1 import (  # noqa: E402
    INDEPENDENT_V2_EXISTING_THREE_METHOD_EVALUATION_ID,
    IndependentV2ExistingThreeMethodEvaluationEnvV1,
    IndependentV2ExistingThreeMethodEvaluationEnvV4V1,
)
from tools import train_high_density_single_seed_100ep_v2 as parent_trainer  # noqa: E402
from tools.high_density_same_scene_v1_common import (  # noqa: E402
    METHOD_ADAPTERS,
    TARGET_SCENARIOS,
    load_protocol as load_parent_protocol,
)
from tools.paper_evaluation_contract import (  # noqa: E402
    build_evaluation_provenance,
    validate_model_environment_spaces,
)
from tools.paper_evaluation_contract_v2 import (  # noqa: E402
    build_evaluation_provenance_v2,
)


DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "independent_v2_existing_models_100seeds_100episodes_v1"
    / "protocol.json"
)
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results_iv2_eval_100s100e_v1"
METHODS = tuple(METHOD_ADAPTERS)
SCENARIOS = tuple(TARGET_SCENARIOS)
EXPERIMENT_ENVIRONMENTS = {
    INDEPENDENT_V2_EXISTING_MODELS_EVALUATION_ID: (
        IndependentV2ExistingModelsEvaluationEnvV1,
        IndependentV2ExistingModelsEvaluationEnvV4V1,
    ),
    INDEPENDENT_V2_EXISTING_THREE_METHOD_EVALUATION_ID: (
        IndependentV2ExistingThreeMethodEvaluationEnvV1,
        IndependentV2ExistingThreeMethodEvaluationEnvV4V1,
    ),
}
BLOCK_SCHEMA_VERSION = "independent-v2-existing-models-seed-block/v1"
JOB_MANIFEST_SCHEMA_VERSION = "independent-v2-existing-models-job/v1"


@dataclass(frozen=True)
class EvaluationJob:
    profile: str
    method: str
    scenario: str
    run_id: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(payload: Any, *, indent: int | None = 2) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {source}")
    return payload


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".j{os.getpid()}.tmp")
    temporary.write_text(_json_text(payload), encoding="utf-8")
    os.replace(temporary, path)


def _project_path(value: str) -> Path:
    candidate = (PROJECT_ROOT / value).resolve()
    if candidate != PROJECT_ROOT and PROJECT_ROOT not in candidate.parents:
        raise ValueError(f"path escapes the project root: {value!r}")
    return candidate


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_close(actual: Any, expected: float, label: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {actual!r}") from exc
    if not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}={value} cannot be recomputed as {expected}")


def validate_protocol(protocol: dict[str, Any], *, protocol_path: Path) -> None:
    _require(
        protocol.get("schema_version")
        == "independent-v2-existing-models-evaluation/v1",
        "unexpected evaluation protocol schema",
    )
    _require(protocol.get("no_fabrication") is True, "no_fabrication must be true")
    _require(
        protocol.get("experiment_id") in EXPERIMENT_ENVIRONMENTS,
        "experiment id conflicts with the supported isolated environments",
    )
    matrix = protocol.get("matrix")
    _require(isinstance(matrix, dict), "matrix is missing")
    method_order = tuple(matrix.get("method_order", ()))
    _require(method_order, "method order is empty")
    _require(len(set(method_order)) == len(method_order), "method order has duplicates")
    _require(
        all(method in METHOD_ADAPTERS for method in method_order),
        "method order contains an unsupported method",
    )
    _require(
        tuple(matrix.get("scenario_order", ())) == SCENARIOS,
        "scenario order drifted",
    )
    expected_jobs = len(method_order) * len(SCENARIOS)
    _require(
        int(matrix.get("method_scenario_jobs", -1)) == expected_jobs,
        "job count drifted",
    )
    _require(
        int(matrix.get("test_seed_blocks_per_job", -1)) == 100,
        "formal test-seed count drifted",
    )
    _require(
        int(matrix.get("episodes_per_test_seed_block", -1)) == 100,
        "formal episode count drifted",
    )
    _require(
        int(matrix.get("evaluation_blocks", -1)) == expected_jobs * 100,
        "formal block count drifted",
    )
    _require(
        int(matrix.get("total_evaluation_episodes", -1))
        == expected_jobs * 100 * 100,
        "formal total episode count drifted",
    )

    methods = protocol.get("methods")
    _require(
        isinstance(methods, dict) and tuple(methods) == method_order,
        "methods drifted",
    )
    for method in method_order:
        adapter, algorithm = METHOD_ADAPTERS[method]
        _require(methods[method].get("adapter") == adapter, f"adapter drifted for {method}")
        _require(
            methods[method].get("algorithm") == algorithm,
            f"algorithm drifted for {method}",
        )

    scenarios = protocol.get("scenarios")
    _require(
        isinstance(scenarios, dict) and tuple(scenarios) == SCENARIOS,
        "scenario settings drifted",
    )
    for scenario in SCENARIOS:
        setting = scenarios[scenario]
        _require(
            str(setting.get("scenario_id", "")).endswith("100s100e_v1"),
            f"scenario id is not isolated for {scenario}",
        )

    environment = protocol.get("environment_contract")
    _require(isinstance(environment, dict), "environment contract is missing")
    for unchanged in (
        "map_changed",
        "ego_route_changed",
        "traffic_density_changed",
        "traffic_partition_changed",
        "observation_changed",
        "action_space_changed",
        "reward_changed",
        "episode_limit_changed",
    ):
        _require(environment.get(unchanged) is False, f"{unchanged} must remain false")
    _require(
        environment.get("artifact_namespace_only_change") is True,
        "the new environment may change only its artifact namespace",
    )

    profiles = protocol.get("profiles")
    _require(isinstance(profiles, dict), "profiles are missing")
    for name, profile in profiles.items():
        count = int(profile["test_seed_count"])
        episodes = int(profile["episodes_per_test_seed"])
        stride = int(profile["episode_seed_stride"])
        base = int(profile["episode_seed_base"])
        start = int(profile["test_seed_index_start"])
        _require(count > 0, f"{name} test_seed_count must be positive")
        _require(episodes > 0, f"{name} episodes_per_test_seed must be positive")
        _require(stride >= episodes, f"{name} seed blocks overlap")
        _require(base >= 0 and start >= 0, f"{name} seed values must be non-negative")
    formal = profiles.get("comparison", {})
    _require(
        int(formal.get("test_seed_index_start", -1)) == 0,
        "formal seed index start drifted",
    )
    _require(int(formal.get("test_seed_count", -1)) == 100, "formal seed count drifted")
    _require(
        int(formal.get("episodes_per_test_seed", -1)) == 100,
        "formal episodes per seed drifted",
    )
    _require(
        int(formal.get("episode_seed_base", -1)) == 200_000,
        "formal episode seed base drifted",
    )
    _require(
        int(formal.get("episode_seed_stride", -1)) == 100,
        "formal episode seed stride drifted",
    )

    artifact = protocol.get("artifact_contract")
    _require(isinstance(artifact, dict), "artifact contract is missing")
    for key in ("result_root", "overlay_directory"):
        value = str(artifact.get(key, ""))
        _require(
            bool(value)
            and not Path(value).is_absolute()
            and len(Path(value).parts) == 1
            and value not in {".", ".."},
            f"{key} must be one safe directory basename",
        )

    parent = protocol.get("parent_v2")
    _require(isinstance(parent, dict), "parent_v2 contract is missing")
    _require(parent.get("training_is_forbidden") is True, "training must be forbidden")
    parent_path = _project_path(str(parent["protocol_path"]))
    _require(parent_path.is_file(), f"parent protocol is missing: {parent_path}")
    expected_parent_hash = str(parent["protocol_sha256"]).lower()
    _require(sha256(parent_path) == expected_parent_hash, "parent protocol hash drifted")
    parent_protocol = load_parent_protocol(parent_path)
    _require(
        all(
            method in tuple(parent_protocol["matrix"]["method_order"])
            for method in method_order
        ),
        "a selected method is absent from the parent protocol",
    )
    _require(
        tuple(parent_protocol["matrix"]["scenario_order"]) == SCENARIOS,
        "parent scenario order drifted",
    )
    for method in method_order:
        _require(
            methods[method]["algorithm"]
            == parent_protocol["methods"][method]["cli_algorithm"],
            f"new protocol algorithm differs from parent for {method}",
        )
        _require(
            methods[method]["adapter"]
            == parent_protocol["methods"][method]["trainer_adapter"],
            f"new protocol adapter differs from parent for {method}",
        )
    for scenario in SCENARIOS:
        for key in (
            "vehicle_scale",
            "pedestrian_scale",
            "clone_depart_jitter_seconds",
        ):
            _require(
                scenarios[scenario][key] == parent_protocol["density"][scenario][key],
                f"{scenario} {key} differs from parent v2",
            )
    _require(protocol_path.is_file(), f"protocol path is missing: {protocol_path}")


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = _read_json(protocol_path)
    validate_protocol(protocol, protocol_path=protocol_path)
    protocol["_path"] = str(protocol_path)
    protocol["_sha256"] = sha256(protocol_path)
    return protocol


def profile_setting(protocol: dict[str, Any], profile: str) -> dict[str, Any]:
    try:
        return dict(protocol["profiles"][profile])
    except KeyError as exc:
        raise ValueError(
            f"unknown profile {profile!r}; choose one of {tuple(protocol['profiles'])}"
        ) from exc


def job_id(protocol: dict[str, Any], profile: str, method: str, scenario: str) -> str:
    return f"{protocol['run_id_prefix']}__{profile}__{method}__{scenario}"


def build_jobs(protocol: dict[str, Any], profile: str) -> list[EvaluationJob]:
    profile_setting(protocol, profile)
    return [
        EvaluationJob(
            profile=profile,
            method=method,
            scenario=scenario,
            run_id=job_id(protocol, profile, method, scenario),
        )
        for method in protocol["matrix"]["method_order"]
        for scenario in protocol["matrix"]["scenario_order"]
    ]


def logical_test_seed_indices(profile: dict[str, Any]) -> tuple[int, ...]:
    start = int(profile["test_seed_index_start"])
    count = int(profile["test_seed_count"])
    return tuple(range(start, start + count))


def episode_seed_start(profile: dict[str, Any], logical_index: int) -> int:
    return int(profile["episode_seed_base"]) + int(logical_index) * int(
        profile["episode_seed_stride"]
    )


def _parent_run_id(protocol: dict[str, Any], method: str, scenario: str) -> str:
    return str(protocol["parent_v2"]["run_id_template"]).format(
        method=method, scenario=scenario
    )


def resolve_source_model(
    protocol: dict[str, Any], method: str, scenario: str
) -> dict[str, Any]:
    _require(
        method in tuple(protocol["matrix"]["method_order"]),
        f"method {method!r} is outside this protocol",
    )
    _require(scenario in SCENARIOS, f"unsupported scenario {scenario!r}")
    parent = protocol["parent_v2"]
    parent_root = _project_path(str(parent["result_root"]))
    run_id = _parent_run_id(protocol, method, scenario)
    run_dir = parent_root / str(parent["profile"]) / "runs" / run_id
    _require(run_dir.is_dir(), f"parent run directory is missing: {run_dir}")
    arguments_path = run_dir / "arguments.json"
    detailed_path = run_dir / "paper_evaluation_detailed.json"
    receipt_path = run_dir / "high_density_job_receipt.json"
    for source in (arguments_path, detailed_path, receipt_path):
        _require(source.is_file(), f"required parent artifact is missing: {source}")

    arguments = _read_json(arguments_path)
    detailed = _read_json(detailed_path)
    receipt = _read_json(receipt_path)
    requested = arguments.get("requested_raw_steps", arguments)
    _require(isinstance(requested, dict), "parent requested arguments are invalid")
    expected_method = protocol["methods"][method]
    _require(requested.get("scenario") == scenario, "parent argument scenario drifted")
    _require(
        requested.get("algo") == expected_method["algorithm"],
        "parent argument algorithm drifted",
    )
    _require(int(requested.get("seed", -1)) == 0, "parent training seed drifted")
    _require(detailed.get("scenario") == scenario, "parent evaluation scenario drifted")
    _require(
        detailed.get("algorithm") == expected_method["algorithm"],
        "parent evaluation algorithm drifted",
    )
    _require(
        int(detailed.get("evaluation_seed_start", -1)) == 10_000,
        "parent evaluation seed block drifted",
    )
    summary = detailed.get("summary")
    _require(
        isinstance(summary, dict) and int(summary.get("episodes", -1)) == 100,
        "parent evaluation is not the completed 100-episode v2 result",
    )
    provenance = detailed.get("evaluation_provenance")
    _require(
        isinstance(provenance, dict) and provenance.get("validated") is True,
        "parent evaluation provenance is not validated",
    )
    _require(receipt.get("status") == "completed", "parent run is not completed")
    _require(receipt.get("fabricated_values") is False, "parent receipt is invalid")
    _require(receipt.get("method") == method, "parent receipt method drifted")
    _require(receipt.get("scenario") == scenario, "parent receipt scenario drifted")
    _require(int(receipt.get("seed", -1)) == 0, "parent receipt seed drifted")
    _require(
        str(receipt.get("protocol_sha256", "")).lower()
        == str(parent["protocol_sha256"]).lower(),
        "parent receipt protocol hash drifted",
    )

    recorded_model = detailed.get("model")
    _require(isinstance(recorded_model, str) and recorded_model, "model is not recorded")
    model_name = Path(recorded_model).name
    _require(model_name.endswith(".zip"), "recorded deployment model is not an SB3 ZIP")
    model_path = run_dir / model_name
    _require(model_path.is_file(), f"recorded deployment model is missing: {model_path}")
    model_sha = sha256(model_path)
    recorded_sha = detailed.get("model_sha256")
    if recorded_sha is not None:
        _require(
            str(recorded_sha).lower() == model_sha,
            "recorded deployment model hash drifted",
        )
    receipt_hashes = receipt.get("artifact_sha256", {})
    if model_name in receipt_hashes:
        _require(
            str(receipt_hashes[model_name]).lower() == model_sha,
            "parent receipt deployment model hash drifted",
        )
    if method == "v4_8":
        _require(model_name == "selected_model.zip", "v4.8 must reuse selected_model.zip")
        for field in (
            "selected_checkpoint_kind",
            "selected_deployment_decoder",
            "selected_model_policy_class",
            "selected_model_parameter_state_sha256",
        ):
            _require(
                isinstance(detailed.get(field), str) and detailed[field],
                f"v4.8 deployment binding {field} is missing",
            )

    return {
        "method": method,
        "scenario": scenario,
        "display_label": expected_method["display_label"],
        "adapter": expected_method["adapter"],
        "algorithm": expected_method["algorithm"],
        "model_class": expected_method["model_class"],
        "training_seed": 0,
        "parent_run_id": run_id,
        "parent_run_directory": str(run_dir.resolve()),
        "arguments_path": str(arguments_path.resolve()),
        "arguments_sha256": sha256(arguments_path),
        "parent_evaluation_path": str(detailed_path.resolve()),
        "parent_evaluation_sha256": sha256(detailed_path),
        "parent_receipt_path": str(receipt_path.resolve()),
        "parent_receipt_sha256": sha256(receipt_path),
        "recorded_model_path": recorded_model,
        "model_path": str(model_path.resolve()),
        "model_basename": model_name,
        "model_sha256": model_sha,
        "requested_arguments": dict(requested),
        "selected_checkpoint_kind": detailed.get("selected_checkpoint_kind"),
        "selected_deployment_decoder": detailed.get("selected_deployment_decoder"),
        "selected_model_policy_class": detailed.get("selected_model_policy_class"),
        "selected_model_parameter_state_sha256": detailed.get(
            "selected_model_parameter_state_sha256"
        ),
    }


def _model_class(method: str) -> type[Any]:
    if method in {"mst_slt", "temporal_graph", "initial_full", "full_balanced"}:
        from algos.sb3_torch.sac import SceneRepresentationSAC

        return SceneRepresentationSAC
    if method == "v4_1":
        from algos.sb3_torch.sac_v4 import DecisionAlignedHybridSACV4

        return DecisionAlignedHybridSACV4
    if method == "v4_8":
        from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45

        return ConfidentActorFusionSACV45
    raise ValueError(f"unsupported method {method!r}")


def make_evaluation_environment(
    *,
    protocol: dict[str, Any],
    source: dict[str, Any],
    result_root: Path,
    profile: str,
) -> Any:
    requested = dict(source["requested_arguments"])
    requested["gui"] = False
    requested["evaluation_split"] = "validation"
    arguments = SimpleNamespace(**requested)
    density = dict(protocol["scenarios"][source["scenario"]])
    overlay_root = (
        result_root.resolve()
        / profile
        / str(protocol["artifact_contract"]["overlay_directory"])
        / source["method"]
    )
    baseline_environment_class, v4_environment_class = EXPERIMENT_ENVIRONMENTS[
        protocol["experiment_id"]
    ]
    factory = parent_trainer._make_v2_environment_factory(
        adapter=str(source["adapter"]),
        density=density,
        overlay_root=overlay_root,
        baseline_environment_class=baseline_environment_class,
        v4_environment_class=v4_environment_class,
    )
    return factory(arguments, evaluation=True)


def _job_root(result_root: Path, job: EvaluationJob) -> Path:
    return result_root.resolve() / job.profile / "runs" / job.run_id


def block_path(result_root: Path, job: EvaluationJob, logical_index: int) -> Path:
    return _job_root(result_root, job) / "blocks" / f"test_seed_{logical_index:03d}.json"


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise ValueError("cannot compute a mean from an empty sequence")
    return float(statistics.fmean(rows))


def _population_std(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise ValueError("cannot compute a standard deviation from an empty sequence")
    return float(statistics.pstdev(rows))


def validate_block_payload(
    payload: dict[str, Any],
    *,
    protocol_sha256: str,
    job: EvaluationJob,
    source_model_sha256: str,
    logical_index: int,
    expected_seed_start: int,
    expected_episodes: int,
) -> None:
    _require(payload.get("schema_version") == BLOCK_SCHEMA_VERSION, "block schema drifted")
    _require(payload.get("status") == "completed", "block is not completed")
    _require(payload.get("computed_from_real_run") is True, "block is not a real run")
    _require(payload.get("fabricated_values") is False, "block fabrication flag drifted")
    _require(payload.get("training_performed") is False, "block retrained a model")
    _require(payload.get("protocol_sha256") == protocol_sha256, "block protocol hash drifted")
    _require(payload.get("profile") == job.profile, "block profile drifted")
    _require(payload.get("method") == job.method, "block method drifted")
    _require(payload.get("scenario") == job.scenario, "block scenario drifted")
    _require(payload.get("job_id") == job.run_id, "block job id drifted")
    _require(
        payload.get("model_sha256") == source_model_sha256,
        "block deployment-model hash drifted",
    )
    seed_block = payload.get("test_seed_block")
    _require(isinstance(seed_block, dict), "test_seed_block is missing")
    _require(int(seed_block.get("logical_index", -1)) == logical_index, "logical seed drifted")
    _require(
        int(seed_block.get("episode_seed_start", -1)) == expected_seed_start,
        "episode seed start drifted",
    )
    _require(
        int(seed_block.get("episode_count", -1)) == expected_episodes,
        "episode count drifted",
    )
    _require(
        int(seed_block.get("episode_seed_end", -1))
        == expected_seed_start + expected_episodes - 1,
        "episode seed end drifted",
    )
    if job.profile == "comparison":
        _require(
            int(seed_block.get("source_traffic_episode_start", -1))
            == logical_index * expected_episodes,
            "source traffic-cycle start drifted",
        )

    summary = payload.get("summary")
    records = payload.get("episode_records")
    _require(isinstance(summary, dict), "block summary is missing")
    _require(isinstance(records, list), "block episode records are missing")
    _require(int(summary.get("episodes", -1)) == expected_episodes, "summary episodes drifted")
    _require(len(records) == expected_episodes, "episode record count drifted")
    expected_seeds = list(range(expected_seed_start, expected_seed_start + expected_episodes))
    _require(
        [int(row.get("seed", -1)) for row in records] == expected_seeds,
        "episode seeds are not the declared contiguous block",
    )
    _require(
        [int(row.get("episode", -1)) for row in records]
        == list(range(expected_episodes)),
        "episode indices are not contiguous",
    )
    for row in records:
        _require(
            isinstance(row.get("traffic_variant"), str) and row["traffic_variant"],
            "episode traffic variant is missing",
        )

    returns = [float(row["episode_return"]) for row in records]
    decision_steps = [float(row["decision_steps"]) for row in records]
    raw_steps = [float(row["raw_steps"]) for row in records]
    completions = [
        float(row["completion_time_seconds"])
        for row in records
        if row.get("completion_time_seconds") is not None
    ]
    successes = sum(bool(row.get("success")) for row in records)
    _require(
        int(payload.get("successful_episodes", -1)) == successes,
        "successful episode count drifted",
    )
    _require_close(summary.get("mean_return"), _mean(returns), "mean_return")
    _require_close(summary.get("std_return"), _population_std(returns), "std_return")
    _require_close(
        summary.get("mean_decision_steps"),
        _mean(decision_steps),
        "mean_decision_steps",
    )
    _require_close(summary.get("mean_raw_steps"), _mean(raw_steps), "mean_raw_steps")
    for metric, key in (
        ("success_rate", "success"),
        ("collision_rate", "collision"),
        ("off_route_rate", "off_route"),
        ("timeout_rate", "timeout"),
    ):
        expected = sum(bool(row.get(key)) for row in records) / expected_episodes
        _require_close(summary.get(metric), expected, metric)
    if completions:
        _require_close(
            payload.get("mean_success_completion_time_seconds"),
            _mean(completions),
            "mean_success_completion_time_seconds",
        )
        _require_close(
            payload.get("std_success_completion_time_seconds"),
            _population_std(completions),
            "std_success_completion_time_seconds",
        )
    else:
        _require(
            payload.get("mean_success_completion_time_seconds") is None,
            "completion mean must be null without successes",
        )
        _require(
            payload.get("std_success_completion_time_seconds") is None,
            "completion std must be null without successes",
        )
    provenance = payload.get("evaluation_provenance")
    _require(
        isinstance(provenance, dict) and provenance.get("validated") is True,
        "evaluation provenance is not validated",
    )
    _require(
        int(provenance.get("evaluation_seed_start", -1)) == expected_seed_start,
        "provenance seed start drifted",
    )
    _require(
        int(provenance.get("episodes", -1)) == expected_episodes,
        "provenance episode count drifted",
    )
    _require(
        provenance.get("model_environment_spaces_match") is True,
        "model/environment spaces were not validated",
    )


def read_valid_block(
    path: Path,
    *,
    protocol_sha256: str,
    job: EvaluationJob,
    source_model_sha256: str,
    logical_index: int,
    expected_seed_start: int,
    expected_episodes: int,
) -> dict[str, Any]:
    payload = _read_json(path)
    validate_block_payload(
        payload,
        protocol_sha256=protocol_sha256,
        job=job,
        source_model_sha256=source_model_sha256,
        logical_index=logical_index,
        expected_seed_start=expected_seed_start,
        expected_episodes=expected_episodes,
    )
    return payload


def _job_manifest(
    *,
    protocol: dict[str, Any],
    job: EvaluationJob,
    source: dict[str, Any],
    result_root: Path,
    space_contract: dict[str, Any] | None,
    completed_indices: list[int],
    started_at_utc: str,
) -> dict[str, Any]:
    profile = profile_setting(protocol, job.profile)
    expected_indices = list(logical_test_seed_indices(profile))
    completed = sorted(completed_indices)
    return {
        "schema_version": JOB_MANIFEST_SCHEMA_VERSION,
        "experiment_id": protocol["experiment_id"],
        "status": "completed" if completed == expected_indices else "running",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "training_performed": False,
        "protocol_path": protocol["_path"],
        "protocol_sha256": protocol["_sha256"],
        "job": asdict(job),
        "source_model": {
            key: value
            for key, value in source.items()
            if key != "requested_arguments"
        },
        "expected_test_seed_indices": expected_indices,
        "completed_test_seed_indices": completed,
        "expected_block_count": len(expected_indices),
        "completed_block_count": len(completed),
        "episodes_per_test_seed": int(profile["episodes_per_test_seed"]),
        "expected_total_episodes": len(expected_indices)
        * int(profile["episodes_per_test_seed"]),
        "space_contract": space_contract,
        "result_directory": str(_job_root(result_root, job).resolve()),
        "started_at_utc": started_at_utc,
        "updated_at_utc": _utc_now(),
    }


def evaluate_job(
    *,
    protocol_path: Path,
    result_root: Path,
    profile_name: str,
    method: str,
    scenario: str,
    device: str,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    profile = profile_setting(protocol, profile_name)
    _require(
        method in tuple(protocol["matrix"]["method_order"]),
        f"method {method!r} is outside this protocol",
    )
    _require(scenario in SCENARIOS, f"unsupported scenario {scenario!r}")
    job = EvaluationJob(
        profile=profile_name,
        method=method,
        scenario=scenario,
        run_id=job_id(protocol, profile_name, method, scenario),
    )
    source = resolve_source_model(protocol, method, scenario)
    indices = list(logical_test_seed_indices(profile))
    episodes = int(profile["episodes_per_test_seed"])
    completed: list[int] = []
    pending: list[int] = []
    for logical_index in indices:
        path = block_path(result_root, job, logical_index)
        if not path.exists():
            pending.append(logical_index)
            continue
        try:
            read_valid_block(
                path,
                protocol_sha256=protocol["_sha256"],
                job=job,
                source_model_sha256=source["model_sha256"],
                logical_index=logical_index,
                expected_seed_start=episode_seed_start(profile, logical_index),
                expected_episodes=episodes,
            )
        except Exception as exc:
            raise ValueError(
                f"refusing to overwrite invalid existing block {path}: {exc}"
            ) from exc
        completed.append(logical_index)

    started_at = _utc_now()
    job_root = _job_root(result_root, job)
    manifest_path = job_root / "job_manifest.json"
    if not pending:
        manifest = _job_manifest(
            protocol=protocol,
            job=job,
            source=source,
            result_root=result_root,
            space_contract=None,
            completed_indices=completed,
            started_at_utc=started_at,
        )
        manifest["status"] = "completed"
        _write_json_atomic(manifest_path, manifest)
        return {
            "job": job.run_id,
            "status": "skipped_complete",
            "completed_blocks": len(completed),
            "expected_blocks": len(indices),
        }

    env = make_evaluation_environment(
        protocol=protocol,
        source=source,
        result_root=result_root,
        profile=profile_name,
    )
    try:
        model_class = _model_class(method)
        _require(
            f"{model_class.__module__}.{model_class.__qualname__}"
            == source["model_class"],
            "runtime model class differs from the frozen protocol",
        )
        model = model_class.load(
            Path(source["model_path"]), env=env, device=device
        )
        space_contract = validate_model_environment_spaces(model, env)
        manifest = _job_manifest(
            protocol=protocol,
            job=job,
            source=source,
            result_root=result_root,
            space_contract=space_contract,
            completed_indices=completed,
            started_at_utc=started_at,
        )
        _write_json_atomic(manifest_path, manifest)

        for logical_index in pending:
            seed_start = episode_seed_start(profile, logical_index)
            traffic_schedule = env.set_logical_test_seed_block(
                logical_index=logical_index,
                profile_index_start=int(profile["test_seed_index_start"]),
                episodes_per_test_seed=episodes,
            )
            block_started = time.perf_counter()
            report = evaluate_model_detailed(
                model,
                env,
                episodes=episodes,
                deterministic=True,
                seed=seed_start,
                sumo_step_seconds=0.1,
                policy_action_hold=1,
            )
            if source["adapter"] == "base":
                provenance = build_evaluation_provenance(
                    model=model,
                    env=env,
                    report=report,
                    algorithm=source["algorithm"],
                    scenario=scenario,
                    traffic_protocol="frozen_80_20",
                    episode_limit_profile="source",
                    evaluation_seed_start=seed_start,
                    environment_factory=(
                        "tools.evaluate_independent_v2_existing_models_100s100e_v1."
                        "make_evaluation_environment"
                    ),
                    space_contract=space_contract,
                )
            else:
                provenance = build_evaluation_provenance_v2(
                    model=model,
                    env=env,
                    report=report,
                    algorithm=source["algorithm"],
                    scenario=scenario,
                    traffic_protocol="frozen_60_20_20",
                    evaluation_split="validation",
                    episode_limit_profile="source",
                    evaluation_seed_start=seed_start,
                    environment_factory=(
                        "tools.evaluate_independent_v2_existing_models_100s100e_v1."
                        "make_evaluation_environment"
                    ),
                    space_contract=space_contract,
                )
            payload = {
                "schema_version": BLOCK_SCHEMA_VERSION,
                "experiment_id": protocol["experiment_id"],
                "status": "completed",
                "computed_from_real_run": True,
                "fabricated_values": False,
                "training_performed": False,
                "deterministic": True,
                "protocol_path": protocol["_path"],
                "protocol_sha256": protocol["_sha256"],
                "profile": profile_name,
                "job_id": job.run_id,
                "method": method,
                "display_label": source["display_label"],
                "scenario": scenario,
                "training_seed": 0,
                "algorithm": source["algorithm"],
                "adapter": source["adapter"],
                "parent_run_id": source["parent_run_id"],
                "parent_evaluation_sha256": source["parent_evaluation_sha256"],
                "model_path": source["model_path"],
                "model_basename": source["model_basename"],
                "model_sha256": source["model_sha256"],
                "selected_checkpoint_kind": source["selected_checkpoint_kind"],
                "selected_deployment_decoder": source[
                    "selected_deployment_decoder"
                ],
                "test_seed_block": {
                    "logical_index": logical_index,
                    "episode_seed_start": seed_start,
                    "episode_seed_end": seed_start + episodes - 1,
                    "episode_count": episodes,
                    "source_traffic_episode_start": traffic_schedule[
                        "source_traffic_episode_start"
                    ],
                    "source_traffic_schedule": (
                        "profile-relative logical block offset; deterministic "
                        "inherited route roll is recomputed at every boundary"
                    ),
                    "episode_seed_formula": (
                        "episode_seed_base + logical_test_seed_index * "
                        "episode_seed_stride + episode_index"
                    ),
                },
                "evaluation_provenance": provenance,
                "completion_time_population_std_ddof": 0,
                "wall_seconds": time.perf_counter() - block_started,
                "completed_at_utc": _utc_now(),
                **report.to_dict(),
            }
            validate_block_payload(
                payload,
                protocol_sha256=protocol["_sha256"],
                job=job,
                source_model_sha256=source["model_sha256"],
                logical_index=logical_index,
                expected_seed_start=seed_start,
                expected_episodes=episodes,
            )
            path = block_path(result_root, job, logical_index)
            _write_json_atomic(path, payload)
            completed.append(logical_index)
            manifest = _job_manifest(
                protocol=protocol,
                job=job,
                source=source,
                result_root=result_root,
                space_contract=space_contract,
                completed_indices=completed,
                started_at_utc=started_at,
            )
            _write_json_atomic(manifest_path, manifest)
            print(
                _json_text(
                    {
                        "job": job.run_id,
                        "test_seed_index": logical_index,
                        "status": "completed",
                        "completed_blocks": len(completed),
                        "expected_blocks": len(indices),
                        "wall_seconds": payload["wall_seconds"],
                        "summary": payload["summary"],
                    },
                    indent=None,
                ),
                flush=True,
            )
    finally:
        env.close()

    return {
        "job": job.run_id,
        "status": "completed",
        "completed_blocks": len(completed),
        "expected_blocks": len(indices),
        "episodes": len(completed) * episodes,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--profile", default="comparison")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    result = evaluate_job(
        protocol_path=args.protocol,
        result_root=args.result_root,
        profile_name=args.profile,
        method=args.method,
        scenario=args.scenario,
        device=args.device,
    )
    print(_json_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCK_SCHEMA_VERSION",
    "DEFAULT_PROTOCOL_PATH",
    "DEFAULT_RESULT_ROOT",
    "EvaluationJob",
    "SCENARIOS",
    "METHODS",
    "block_path",
    "build_jobs",
    "episode_seed_start",
    "evaluate_job",
    "job_id",
    "load_protocol",
    "logical_test_seed_indices",
    "profile_setting",
    "read_valid_block",
    "resolve_source_model",
    "sha256",
    "validate_block_payload",
    "validate_protocol",
]
