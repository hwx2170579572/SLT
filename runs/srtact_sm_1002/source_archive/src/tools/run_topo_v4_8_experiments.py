"""Hash-bound v4.8 development, promotion, and formal experiment runner.

The runner reuses the audited v4.7 aggregation/gating machinery, but replaces
all scientific identities, paths, job matrices, and run acceptance checks with
the preregistered v4.8 tie-only replicated-calibration protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sb3_configs_v4_8 import (
    TEMPORAL_GRAPH,
    V48_CANDIDATE,
    V48_FORMAL_ALGORITHMS,
    V48_IMPLEMENTATION_IDS,
)
from tools import run_topo_v4_7_experiments as scientific
from tools.action_diagnostics_v4_5 import fusion_decoder_metrics
from tools.action_diagnostics_v4_6 import _target_integrity
from tools.checkpoint_decoder_selector_v4_6 import (
    CANDIDATE_DECODERS,
    FUSION_DECODER,
    PARENT_DECODER,
    TARGET_DECODER,
    select_deployment as select_parent_deployment,
)
from tools.checkpoint_decoder_selector_v4_8 import (
    SELECTOR_MODE,
    select_deployment,
    tied_top_pairs,
)
from tools.checkpoint_selector_v4_5_1 import terminal_event_evidence


DEFAULT_CONTRACT = (
    PROJECT_ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_8.yaml"
)
DEFAULT_STAGE0_RESULTS = scientific.DEFAULT_STAGE0_RESULTS
DEFAULT_STAGE0_ATTRIBUTION = scientific.DEFAULT_STAGE0_ATTRIBUTION
DEFAULT_FAILURE_ATTRIBUTION = (
    PROJECT_ROOT
    / "results_topo_v4_7_2_dev"
    / "attribution"
    / "g3_failure"
    / "attribution_summary.json"
)
DEFAULT_DEEP_ATTRIBUTION = DEFAULT_FAILURE_ATTRIBUTION.with_name("deep_attribution.md")
DEFAULT_PREREGISTRATION = PROJECT_ROOT / "results_topo_v4_8_dev" / "preregistration_receipt.json"
DEFAULT_ENGINEERING_ROOT = PROJECT_ROOT / "results_topo_v4_8_dev" / "engineering"
DEFAULT_FREEZE = DEFAULT_ENGINEERING_ROOT / "implementation_freeze.json"
DEFAULT_ENGINEERING_RECEIPT = DEFAULT_ENGINEERING_ROOT / "engineering_receipt.json"
DEFAULT_PROMOTION_GATE = PROJECT_ROOT / "results_topo_v4_8_promotion" / "promotion_gate.json"

LINEAGE_HASHES = {
    "parent_contract_sha256": "e97ef91a85ddeb60bee50fb6fdccdff4afb4e49f5a9d6e52ffee24431890ea7a",
    "parent_implementation_freeze_sha256": "8b38eb76de4e0287867e1a588ab1e65895b01d5a7331f24c9ce9debd25f7e49a",
    "parent_development_decision_sha256": "21808a9a1ec3ec4ff30e892cb962b8dc4cf235a5b8212e30deeaea0cd8b7e0f1",
    "failure_attribution_sha256": "1db6fbcd71d8effc1b08dd345e7ec91d6ab42705ff5c4ffe19b218fc1049cd92",
    "deep_attribution_sha256": "e376bc1d1f7689471a94ca58cd8d4515cdd6ce381ee09045a10999c35e024dfc",
    "attribution_tool_sha256": "cc0c94b9b10385f734afbdb91b29f781cf29651600ff2e4c4dca383d9d038823",
    "posthoc_replay_tool_sha256": "0ae2ebf69177142e2272c2c8bfc0e5659a4d6574dbcef09134496eff69e8eedf",
}

SCENARIOS = scientific.SCENARIOS
STAGES = scientific.STAGES
METHOD_ALGORITHMS = {
    "temporal_graph": TEMPORAL_GRAPH,
    "selected_v4_candidate": V48_CANDIDATE,
}
METHOD_CODES = {"temporal_graph": "tg", "selected_v4_candidate": "cand"}
SCENARIO_CODES = scientific.SCENARIO_CODES
FULL_RUN_REQUIRED_FILES = scientific.FULL_RUN_REQUIRED_FILES


class V48ProtocolError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V48ProtocolError(message)


def _sha256(path: Path) -> str:
    return scientific._sha256(path)


def _canonical_sha256(value: Any) -> str:
    return scientific._canonical_sha256(value)


def _load_json(path: Path) -> dict[str, Any]:
    return scientific._load_json(path)


def _version_tree(value: Any) -> Any:
    output = copy.deepcopy(value)
    if isinstance(output, dict):
        schema = output.get("schema_version")
        if isinstance(schema, str) and schema.startswith("topo-scene-v4.7."):
            output["schema_version"] = schema.replace(
                "topo-scene-v4.7.", "topo-scene-v4.8.", 1
            )
        for key, item in tuple(output.items()):
            output[key] = _version_tree(item)
    elif isinstance(output, list):
        output = [_version_tree(item) for item in output]
    return output


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_version_tree(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "v4.8 contract must be a mapping")
    return value


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version")
        == "topo-scene-v4.8.tie-only-replicated-train-calibration-contract/v1",
        "unexpected v4.8 contract schema",
    )
    _require(contract.get("no_fabrication") is True, "no_fabrication must be true")
    _require(contract.get("formal_test_locked") is True, "formal test must start locked")
    lineage = contract.get("lineage", {})
    for key, expected in LINEAGE_HASHES.items():
        _require(lineage.get(key) == expected, f"lineage.{key} drifted")
    _require(lineage.get("formal_test_accessed") is False, "lineage accessed formal test")

    change = contract.get("single_change", {})
    _require(change.get("name") == "tie_only_replicated_train_calibration", "single change drifted")
    _require(change.get("initial_calibration", {}).get("partition") == "train", "primary partition drifted")
    _require(change.get("initial_calibration", {}).get("episodes_per_pair") == 12, "primary budget drifted")
    _require(change.get("initial_calibration", {}).get("pairs") == 4, "primary matrix drifted")
    secondary = change.get("secondary_calibration", {})
    _require(secondary.get("trigger") == "tied_top_candidate_count_greater_than_one", "tie trigger drifted")
    _require(secondary.get("partition") == "train", "secondary partition drifted")
    _require(secondary.get("episodes_per_tied_pair") == 12, "secondary budget drifted")
    _require(secondary.get("seed_start_offset_from_primary") == 100, "secondary offset drifted")
    _require(secondary.get("candidate_scope") == "tied_top_only", "secondary scope drifted")
    _require(secondary.get("non_top_candidate_reentry_forbidden") is True, "non-top reentry allowed")
    _require(change.get("validation_used_for_selection") is False, "selector used validation")
    _require(change.get("formal_test_used_for_selection") is False, "selector used formal test")
    unchanged = contract.get("unchanged", {})
    _require(unchanged.get("return_n_step") == 16, "return horizon drifted")
    _require(unchanged.get("return_bootstrap_discount") == "gamma_power_actual_horizon", "bootstrap drifted")
    for key in (
        "reward",
        "observation_and_action_spaces",
        "network_and_representation",
        "optimizer_and_entropy",
        "training_budget_and_warmup",
        "decoder_definitions",
        "outcome_gates",
        "promotion_and_formal_protocol",
    ):
        _require(unchanged.get(key) is True, f"unchanged.{key} drifted")

    cells = contract.get("development", {}).get("ordered_cells", [])
    expected_cells = [
        ("H1", "cross", 8, 77_000, 77_100, 68_100),
        ("H2", "cross", 9, 78_000, 78_100, 69_100),
        ("H3", "carla", 5, 79_000, 79_100, 70_100),
        ("H4", "roundabout_medium", 3, 80_000, 80_100, 71_100),
    ]
    observed = [
        (
            str(row["id"]),
            str(row["scenario"]),
            int(row["seed"]),
            int(row["calibration_seed_start"]),
            int(row["secondary_calibration_seed_start"]),
            int(row["evaluation_seed_start"]),
        )
        for row in cells
    ]
    _require(observed == expected_cells, "development cell matrix drifted")
    _require(all(row.get("method") == "selected_v4_candidate" for row in cells), "development method drifted")
    _require(all(int(row["raw_steps"]) == 20_000 for row in cells), "development raw steps drifted")
    development = contract["development"]
    _require(development.get("strict_sequential_execution") is True, "development is not sequential")
    _require(development.get("historical_or_posthoc_gate_inputs_forbidden") is True, "post-hoc gate inputs allowed")
    _require(int(development["evaluation_episodes"]) == 12, "development evaluation budget drifted")
    _require(int(development["primary_calibration_episodes_per_pair"]) == 12, "primary budget drifted")
    _require(int(development["secondary_calibration_episodes_per_tied_pair"]) == 12, "secondary budget drifted")

    promotion = contract["promotion"]
    _require(len(promotion["methods"]) * len(promotion["scenarios"]) * len(promotion["seeds"]) == 12, "promotion matrix drifted")
    formal = contract["formal_test"]
    _require(formal.get("locked") is True and formal.get("accessed") is False, "formal lock drifted")
    _require(len(formal["methods"]) * len(formal["scenarios"]) * len(formal["seeds"]) == 120, "formal matrix drifted")
    _require(formal["methods"] == ["temporal_graph", "selected_v4_candidate"], "formal pair drifted")
    return contract


def contract_sha256(path: Path = DEFAULT_CONTRACT) -> str:
    return _sha256(path.resolve())


def _validate_failure_attribution(summary: Path, deep: Path) -> dict[str, str]:
    payload = _load_json(summary)
    _require(payload.get("schema_version") == "topo-scene-v4.7.g3-failure-attribution/v1", "attribution schema drifted")
    _require(payload.get("computed_from_real_runs") is True, "attribution is not real")
    _require(payload.get("fabricated_values") is False, "attribution is fabricated")
    _require(payload.get("post_hoc_diagnostic_only") is True, "attribution is not post-hoc")
    _require(payload.get("counted_for_development_gate") is False, "post-hoc evidence counted toward a gate")
    _require(payload.get("formal_test_accessed") is False, "attribution accessed formal test")
    selected = payload.get("selected_next_hypothesis", {})
    _require(selected.get("version") == "v4.8_tie_only_replicated_train_calibration", "attribution recommendation drifted")
    _require(selected.get("training_changed") is False, "attribution changed training")
    _require(_sha256(summary) == LINEAGE_HASHES["failure_attribution_sha256"], "attribution hash drifted")
    _require(_sha256(deep) == LINEAGE_HASHES["deep_attribution_sha256"], "deep attribution hash drifted")
    return {
        "failure_attribution_sha256": _sha256(summary),
        "deep_attribution_sha256": _sha256(deep),
    }


def validate_implementation_freeze(path: Path = DEFAULT_FREEZE) -> dict[str, Any]:
    payload = _load_json(path.resolve())
    _require(payload.get("schema_version") == "topo-scene-v4.8.implementation-freeze/v1", "invalid v4.8 freeze")
    _require(payload.get("formal_test_accessed") is False, "freeze accessed formal test")
    _require(payload.get("experiment_contract_sha256") == contract_sha256(), "freeze contract hash drifted")
    _require(payload.get("failure_attribution_sha256") == LINEAGE_HASHES["failure_attribution_sha256"], "freeze attribution drifted")
    _require(payload.get("preregistration_receipt_sha256") == _sha256(DEFAULT_PREREGISTRATION), "freeze preregistration drifted")
    files = payload.get("scientific_files")
    _require(isinstance(files, dict), "freeze scientific file map missing")
    for relative, expected in files.items():
        source = PROJECT_ROOT / relative
        _require(source.is_file(), f"frozen source missing: {relative}")
        _require(_sha256(source) == expected, f"frozen source drifted: {relative}")
    _require(_canonical_sha256(files) == payload.get("scientific_content_sha256"), "freeze content digest invalid")
    return payload


def protocol_hashes(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    stage0_results: Path = DEFAULT_STAGE0_RESULTS,
    stage0_attribution: Path = DEFAULT_STAGE0_ATTRIBUTION,
    failure_attribution: Path = DEFAULT_FAILURE_ATTRIBUTION,
    deep_attribution: Path = DEFAULT_DEEP_ATTRIBUTION,
    freeze_path: Path = DEFAULT_FREEZE,
) -> dict[str, str]:
    validate_contract(load_contract(contract_path.resolve()))
    hashes = {
        "experiment_contract_sha256": _sha256(contract_path.resolve()),
        **scientific.parent._validate_stage0(stage0_results.resolve(), stage0_attribution.resolve()),
        **_validate_failure_attribution(failure_attribution.resolve(), deep_attribution.resolve()),
        "preregistration_receipt_sha256": _sha256(DEFAULT_PREREGISTRATION),
    }
    validate_implementation_freeze(freeze_path.resolve())
    hashes["implementation_freeze_sha256"] = _sha256(freeze_path.resolve())
    return hashes


@dataclass(frozen=True)
class Job:
    stage: str
    job_id: str
    kind: str
    method: str
    algorithm: str
    implementation_id: str
    scenario: str
    seed: int
    raw_steps: int
    calibration_episodes: int
    calibration_seed_start: int
    evaluation_episodes: int
    evaluation_split: str
    evaluation_seed_start: int
    role: str
    protocol_tag: str

    @property
    def name(self) -> str:
        prefix = self.job_id if self.stage == "development" else self.stage[:1].upper()
        return f"{prefix}__{METHOD_CODES[self.method]}__{SCENARIO_CODES[self.scenario]}__s{self.seed}__p{self.protocol_tag}"


def jobs_for_stage(contract: dict[str, Any], digest: str, stage: str) -> list[Job]:
    _require(stage in STAGES, f"unsupported v4.8 stage {stage!r}")
    tag = digest[:8]
    jobs: list[Job] = []
    if stage == "development":
        settings = contract["development"]
        for specification in settings["ordered_cells"]:
            method = str(specification["method"])
            algorithm = METHOD_ALGORITHMS[method]
            jobs.append(
                Job(
                    stage=stage,
                    job_id=str(specification["id"]),
                    kind=str(specification["kind"]),
                    method=method,
                    algorithm=algorithm,
                    implementation_id=V48_IMPLEMENTATION_IDS[algorithm],
                    scenario=str(specification["scenario"]),
                    seed=int(specification["seed"]),
                    raw_steps=int(specification["raw_steps"]),
                    calibration_episodes=int(settings["primary_calibration_episodes_per_pair"]),
                    calibration_seed_start=int(specification["calibration_seed_start"]),
                    evaluation_episodes=int(settings["evaluation_episodes"]),
                    evaluation_split=str(settings["evaluation_split"]),
                    evaluation_seed_start=int(specification["evaluation_seed_start"]),
                    role=str(specification["role"]),
                    protocol_tag=tag,
                )
            )
    else:
        settings = contract["promotion" if stage == "promotion" else "formal_test"]
        calibration_base = 82_000 if stage == "promotion" else 110_000
        evaluation_base = 92_000 if stage == "promotion" else 210_000
        for method in settings["methods"]:
            algorithm = METHOD_ALGORITHMS[method]
            for scenario in settings["scenarios"]:
                for seed in settings["seeds"]:
                    jobs.append(
                        Job(
                            stage=stage,
                            job_id="P" if stage == "promotion" else "F",
                            kind="fresh_training_with_train_only_deployment_selection",
                            method=method,
                            algorithm=algorithm,
                            implementation_id=V48_IMPLEMENTATION_IDS[algorithm],
                            scenario=str(scenario),
                            seed=int(seed),
                            raw_steps=int(settings["raw_training_steps"]),
                            calibration_episodes=12,
                            calibration_seed_start=calibration_base + int(seed) * 1_000,
                            evaluation_episodes=int(settings["evaluation_episodes"]),
                            evaluation_split=str(settings["evaluation_split"]),
                            evaluation_seed_start=evaluation_base + int(seed) * 1_000,
                            role="paired_promotion" if stage == "promotion" else "untouched_formal_test",
                            protocol_tag=tag,
                        )
                    )
    expected = {"development": 4, "promotion": 12, "formal": 120}[stage]
    _require(len(jobs) == expected, f"{stage} must contain {expected} jobs")
    _require(len({job.name for job in jobs}) == expected, f"duplicate {stage} run identity")
    return jobs


def stage_root(stage: str) -> Path:
    return {
        "development": PROJECT_ROOT / "results_topo_v4_8_dev" / "development",
        "promotion": PROJECT_ROOT / "results_topo_v4_8_promotion",
        "formal": PROJECT_ROOT / "results_topo_v4_8_formal",
    }[stage]


def run_directory(job: Job) -> Path:
    return stage_root(job.stage) / "runs" / job.name


def command_for(
    job: Job,
    hashes: dict[str, str],
    *,
    device: str,
    promotion_gate: Path = DEFAULT_PROMOTION_GATE,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_paper_sb3_sumo_v4_8.py"),
        "--scenario", job.scenario,
        "--algo", job.algorithm,
        "--max-steps", str(job.raw_steps),
        "--learning-starts", "5000",
        "--checkpoint-freq", str(job.raw_steps),
        "--eval-freq", "0",
        "--eval-episodes", str(job.evaluation_episodes),
        "--evaluation-split", job.evaluation_split,
        "--evaluation-seed-start", str(job.evaluation_seed_start),
        "--calibration-episodes", str(job.calibration_episodes),
        "--calibration-seed-start", str(job.calibration_seed_start),
        "--seed", str(job.seed),
        "--device", device,
        "--batch-size", "32",
        "--learning-rate", "0.0001",
        "--discount", "0.99",
        "--buffer-size", "20000",
        "--action-repeat", "3",
        "--traffic-protocol", "frozen_60_20_20",
        "--episode-limit-profile", "source",
        "--output-dir", str((stage_root(job.stage) / "runs").resolve()),
        "--model-name", job.name,
        "--experiment-contract-sha256", hashes["experiment_contract_sha256"],
        "--stage0-results-sha256", hashes["stage0_results_sha256"],
        "--attribution-sha256", hashes["failure_attribution_sha256"],
        "--implementation-freeze-sha256", hashes["implementation_freeze_sha256"],
    ]
    if job.stage == "formal":
        command.extend(["--formal-unlock-receipt", str(promotion_gate.resolve())])
    return command


def _load_trace(path: Path) -> list[dict[str, Any]]:
    return scientific._load_trace(path)


def _candidate_from_artifacts(
    root: Path,
    row: dict[str, Any],
    *,
    round_name: str,
    fallback: bool,
) -> tuple[dict[str, Any], list[tuple[int, int, Any]]]:
    kind = str(row["checkpoint_kind"])
    decoder = str(row["deployment_decoder"])
    source_kind = "exact_final" if fallback and kind == "highest_training_success" else kind
    short = {"highest_training_success": "best", "exact_final": "final"}[source_kind]
    pair_dir = root / "selector" / round_name / short / decoder
    evaluation = _load_json(pair_dir / "evaluation.json")
    detailed = _load_json(pair_dir / "detailed.json")
    actions = _load_json(pair_dir / "actions.json")
    expected_schema = (
        "topo-scene-v4.8.secondary-checkpoint-decoder-calibration/v1"
        if round_name == "cal_secondary"
        else "topo-scene-v4.8.checkpoint-decoder-calibration/v1"
    )
    _require(detailed.get("schema_version") == expected_schema, f"{round_name} schema drifted")
    _require(evaluation == row["summary"], f"summary drifted: {kind} x {decoder}")
    observed_signature = [
        {"episode": int(ep["episode"]), "seed": int(ep["seed"]), "traffic_variant": ep.get("traffic_variant")}
        for ep in detailed["episode_records"]
    ]
    _require(observed_signature == row["episode_pair_signature"], f"signature drifted: {kind} x {decoder}")
    _require(_sha256(pair_dir / "detailed.json") == row["calibration_result_sha256"], f"detail hash drifted: {kind} x {decoder}")
    _require(_sha256(pair_dir / "actions.json") == row["calibration_action_diagnostics_sha256"], f"action hash drifted: {kind} x {decoder}")
    _require(actions.get("selected_deployment_decoder") == decoder, f"decoder trace drifted: {kind} x {decoder}")
    _require(row.get("decoder_integrity_passed") is True, f"decoder integrity failed: {kind} x {decoder}")
    terminal_event_evidence(evaluation, detailed["episode_records"])
    candidate = {
        "checkpoint_kind": kind,
        "deployment_decoder": decoder,
        "checkpoint_path": row["checkpoint_path"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "traffic_partition": "train",
        "calibration_seed_start": int(row["calibration_seed_start"]),
        "summary": evaluation,
        "episode_records": detailed["episode_records"],
        "calibration_result_sha256": row["calibration_result_sha256"],
        "calibration_action_diagnostics_sha256": row["calibration_action_diagnostics_sha256"],
        "decoder_integrity": row["decoder_integrity"],
        "decoder_integrity_passed": True,
    }
    signature = [(int(ep["episode"]), int(ep["seed"]), ep.get("traffic_variant")) for ep in detailed["episode_records"]]
    return candidate, signature


def _validate_selector_evidence(root: Path, job: Job) -> dict[str, Any]:
    receipt = _load_json(root / "selector" / "receipt.json")
    _require(receipt.get("selection_partition") == "train", "selector partition drifted")
    _require(receipt.get("paired_calibration") is True, "primary calibration is not paired")
    _require(receipt.get("calibration_episodes") == job.calibration_episodes, "primary episode count drifted")
    _require(receipt.get("calibration_seed_start") == job.calibration_seed_start, "primary seed block drifted")
    _require(receipt.get("validation_used_for_selection") is False, "selector used validation")
    _require(receipt.get("formal_test_used_for_selection") is False, "selector used formal test")
    _require(receipt.get("selector_receipt_precedes_validation_environment") is True, "selector ordering flag is false")
    is_candidate = job.algorithm == V48_CANDIDATE
    expected_schema = (
        "topo-scene-v4.8.tie-only-replicated-selector/v1"
        if is_candidate
        else "topo-scene-v4.8.checkpoint-decoder-selector-receipt/v1"
    )
    _require(receipt.get("schema_version") == expected_schema, "selector receipt schema drifted")
    candidates = receipt.get("candidates", [])
    expected_decoders = set(CANDIDATE_DECODERS) if is_candidate else {PARENT_DECODER}
    _require(len(candidates) == (4 if is_candidate else 2), "selector candidate count drifted")
    _require({row["deployment_decoder"] for row in candidates} == expected_decoders, "selector decoder set drifted")
    _require({row["checkpoint_kind"] for row in candidates} == {"highest_training_success", "exact_final"}, "selector checkpoint set drifted")
    fallback = bool(receipt.get("training_best_missing_fallback_used"))
    initial: list[dict[str, Any]] = []
    primary_signatures = []
    for row in candidates:
        candidate, signature = _candidate_from_artifacts(root, row, round_name="cal", fallback=fallback)
        initial.append(candidate)
        primary_signatures.append(signature)
    _require(all(signature == primary_signatures[0] for signature in primary_signatures), "primary pairs are not traffic-paired")

    if is_candidate:
        tied = tied_top_pairs(initial)
        tied_names = [f"{kind}__{decoder}" for kind, decoder in tied]
        _require(receipt.get("selector_mode") == SELECTOR_MODE, "selector mode drifted")
        _require(receipt.get("initial_empirical_tied_top_count") == len(tied), "tie count drifted")
        _require(receipt.get("initial_empirical_tied_top_pairs") == tied_names, "tied pair set drifted")
        _require(receipt.get("secondary_calibration_triggered") is (len(tied) > 1), "secondary trigger drifted")
        _require(receipt.get("secondary_calibration_seed_offset") == 100, "secondary offset drifted")
        _require(receipt.get("secondary_candidate_scope") == "empirical_tied_top_only", "secondary scope drifted")
        _require(receipt.get("non_top_candidate_reentry_forbidden") is True, "non-top reentry flag drifted")
        secondary: list[dict[str, Any]] | None = None
        secondary_dir = root / "selector" / "cal_secondary"
        if len(tied) == 1:
            _require(not secondary_dir.exists(), "secondary environment exists without a tie")
            _require(receipt.get("secondary_candidates") == [], "secondary receipt rows exist without a tie")
            _require(receipt.get("secondary_calibration_seed_start") is None, "secondary seed exists without a tie")
        else:
            rows = receipt.get("secondary_candidates", [])
            _require(len(rows) == len(tied), "secondary candidate count drifted")
            _require({(row["checkpoint_kind"], row["deployment_decoder"]) for row in rows} == set(tied), "secondary set differs from tied top")
            _require(receipt.get("secondary_calibration_seed_start") == job.calibration_seed_start + 100, "secondary seed block drifted")
            _require(receipt.get("secondary_calibration_episodes") == job.calibration_episodes, "secondary episode count drifted")
            secondary = []
            secondary_signatures = []
            for row in rows:
                candidate, signature = _candidate_from_artifacts(root, row, round_name="cal_secondary", fallback=fallback)
                secondary.append(candidate)
                secondary_signatures.append(signature)
            _require(all(signature == secondary_signatures[0] for signature in secondary_signatures), "secondary pairs are not traffic-paired")
            _require({seed for _, seed, _ in primary_signatures[0]}.isdisjoint({seed for _, seed, _ in secondary_signatures[0]}), "primary and secondary seeds overlap")
            _require([variant for _, _, variant in primary_signatures[0]] == [variant for _, _, variant in secondary_signatures[0]], "secondary train variant support drifted")
        recomputed = select_deployment(initial, secondary, secondary_seed_offset=100)
    else:
        _require(not (root / "selector" / "cal_secondary").exists(), "control unexpectedly created secondary calibration")
        recomputed = select_parent_deployment(initial)

    _require(recomputed["selected_checkpoint_kind"] == receipt.get("selected_checkpoint_kind"), "selected checkpoint does not recompute")
    _require(recomputed["selected_deployment_decoder"] == receipt.get("selected_deployment_decoder"), "selected decoder does not recompute")
    source = Path(receipt["selected_source_checkpoint_path"])
    selected_model = root / "selected_model.zip"
    scientific._verify_zip(source)
    scientific._verify_zip(selected_model)
    _require(_sha256(source) == receipt.get("selected_source_checkpoint_sha256") == receipt.get("selected_checkpoint_sha256"), "selected source hash drifted")
    _require(_sha256(selected_model) == receipt.get("selected_model_sha256"), "selected model hash drifted")
    _require(receipt.get("policy_parameter_state_preserved") is True, "selected deployment changed tensors")
    _require(receipt.get("source_policy_parameter_state_sha256") == receipt.get("selected_model_parameter_state_sha256"), "selected tensor digest drifted")
    expected_policy = {TARGET_DECODER: "TargetCriticDecisionAlignedSACPolicyV43", FUSION_DECODER: "ConfidentActorFusionSACPolicyV45"}.get(receipt["selected_deployment_decoder"])
    if expected_policy:
        _require(receipt.get("selected_model_policy_class") == expected_policy, "selected policy class drifted")
    return receipt


def _validate_return_estimator(root: Path, job: Job, training: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _load_json(root / "return_estimator_diagnostics.json")
    _require(training.get("return_estimator") == diagnostics, "return diagnostics are not bound to training")
    _require(diagnostics.get("schema_version") == "topo-scene-v4.8.return-estimator-diagnostics/v1", "return schema drifted")
    _require(diagnostics.get("algorithm") == job.algorithm, "return algorithm drifted")
    _require(diagnostics.get("implementation_id") == job.implementation_id, "return implementation drifted")
    _require(abs(float(diagnostics.get("gamma")) - 0.99) <= 1e-12, "return gamma drifted")
    if job.algorithm == V48_CANDIDATE:
        _require(diagnostics.get("n_step") == 16, "candidate n-step drifted")
        _require(diagnostics.get("horizon_correct_bootstrap") is True, "candidate bootstrap is not horizon-correct")
        _require(diagnostics.get("bootstrap_discount") == "gamma_power_actual_horizon", "candidate bootstrap discount drifted")
        _require(int(diagnostics.get("sampled_horizon_count", 0)) > 0, "candidate sampled no replay horizons")
        _require(1.0 <= float(diagnostics["mean_sampled_actual_horizon"]) <= 16.0, "candidate mean horizon invalid")
        _require(0.0 <= float(diagnostics["short_horizon_sample_rate"]) <= 1.0, "candidate short-horizon rate invalid")
        _require(diagnostics.get("return_estimator_changed_from_v4_7") is False, "v4.8 changed return estimator")
    else:
        _require(diagnostics.get("n_step") == 4, "control n-step drifted")
        _require(diagnostics.get("horizon_correct_bootstrap") is False, "control bootstrap drifted")
        _require(diagnostics.get("bootstrap_discount") == "single_gamma_source_equivalent", "control bootstrap discount drifted")
    return diagnostics


def accepted_run_reason(job: Job, hashes: dict[str, str]) -> tuple[bool, str]:
    root = run_directory(job)
    try:
        _require(root.is_dir(), "run directory is absent")
        for filename in FULL_RUN_REQUIRED_FILES:
            _require((root / filename).is_file(), f"missing {filename}")
        arguments = _load_json(root / "arguments.json")
        metadata = _load_json(root / "method_metadata.json")
        detailed = _load_json(root / "paper_evaluation_detailed.json")
        final_eval = _load_json(root / "final_evaluation.json")
        actions = _load_json(root / "action_diagnostics.json")
        training = _load_json(root / "training_diagnostics.json")
        checkpoints = _load_json(root / "checkpoint_audit.json")
        performance = _load_json(root / "performance_profile.json")
        selector = _validate_selector_evidence(root, job)
        returns = _validate_return_estimator(root, job, training)
        for key in ("experiment_contract_sha256", "stage0_results_sha256", "attribution_sha256", "implementation_freeze_sha256"):
            source_key = "failure_attribution_sha256" if key == "attribution_sha256" else key
            _require(arguments.get(key) == hashes[source_key], f"arguments {key} mismatch")
            _require(detailed.get(key) == hashes[source_key], f"detailed {key} mismatch")
        _require(selector.get("experiment_contract_sha256") == hashes["experiment_contract_sha256"], "selector contract hash mismatch")
        _require(selector.get("attribution_sha256") == hashes["failure_attribution_sha256"], "selector attribution hash mismatch")
        _require(selector.get("implementation_freeze_sha256") == hashes["implementation_freeze_sha256"], "selector freeze hash mismatch")
        requested = arguments["requested_raw_steps"]
        exact = {
            "algo": job.algorithm,
            "scenario": job.scenario,
            "seed": job.seed,
            "max_steps": job.raw_steps,
            "learning_starts": 5000,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "discount": 0.99,
            "buffer_size": 20_000,
            "action_repeat": 3,
            "traffic_protocol": "frozen_60_20_20",
            "episode_limit_profile": "source",
            "evaluation_split": job.evaluation_split,
            "evaluation_seed_start": job.evaluation_seed_start,
            "eval_episodes": job.evaluation_episodes,
            "calibration_seed_start": job.calibration_seed_start,
            "secondary_calibration_seed_start": job.calibration_seed_start + 100,
            "calibration_episodes": job.calibration_episodes,
            "eval_freq": 0,
            "checkpoint_freq": job.raw_steps,
        }
        for key, expected in exact.items():
            observed = requested.get(key)
            _require(abs(float(observed) - expected) <= 1e-12 if isinstance(expected, float) else observed == expected, f"argument {key} mismatch")
        fidelity = arguments.get("implementation_fidelity", {})
        _require(fidelity.get("implementation_id") == job.implementation_id, "implementation id mismatch")
        expected_change = "tie_only_replicated_train_calibration" if job.algorithm == V48_CANDIDATE else "none_frozen_control"
        _require(fidelity.get("single_change") == expected_change, "single change mismatch")
        _require(fidelity.get("return_estimator_changed_from_v4_7") is False, "return estimator changed from v4.7")
        _require(fidelity.get("training_changed_from_v4_7") is False, "training changed from v4.7")
        _require(metadata.get("implementation_id") == job.implementation_id, "metadata id mismatch")
        _require(metadata.get("checkpoint_calibration_partition") == "train", "metadata selector partition drifted")
        _require(metadata.get("validation_used_for_checkpoint_selection") is False, "metadata admits validation selection")
        _require(detailed.get("schema_version") == "topo-scene-v4.8.detailed-evaluation/v1", "detailed schema drifted")
        _require(detailed.get("implementation_id") == job.implementation_id, "detailed implementation mismatch")
        _require(detailed.get("algorithm") == job.algorithm and detailed.get("scenario") == job.scenario, "detailed identity mismatch")
        _require(detailed.get("evaluation_split") == job.evaluation_split and detailed.get("evaluation_seed_start") == job.evaluation_seed_start, "evaluation partition/seed drifted")
        _require(detailed.get("trained_raw_steps") == job.raw_steps == detailed.get("collected_training_raw_steps"), "raw training clock drifted")
        _require(detailed.get("model_sha256") == selector.get("selected_model_sha256"), "detailed selected model hash mismatch")
        bindings = {
            "selected_checkpoint_kind": selector.get("selected_checkpoint_kind"),
            "selected_deployment_decoder": selector.get("selected_deployment_decoder"),
            "selected_source_checkpoint_sha256": selector.get("selected_source_checkpoint_sha256"),
            "selected_model_policy_class": selector.get("selected_model_policy_class"),
            "selected_model_parameter_state_sha256": selector.get("selected_model_parameter_state_sha256"),
        }
        for key, expected in bindings.items():
            _require(detailed.get(key) == expected, f"detailed {key} binding mismatch")
        _require(detailed.get("selector_receipt_sha256") == _sha256(root / "selector" / "receipt.json"), "selector receipt binding mismatch")
        sealed = datetime.fromisoformat(str(detailed["selector_receipt_sealed_at_utc"]))
        constructed = datetime.fromisoformat(str(detailed["validation_environment_constructed_at_utc"]))
        _require(sealed <= constructed, "validation environment predates selector receipt")
        _require(final_eval == detailed["summary"], "final and detailed summaries differ")
        _require(final_eval.get("episodes") == job.evaluation_episodes, "final episode count mismatch")
        provenance = detailed.get("evaluation_provenance", {})
        _require(provenance.get("validated") is True and provenance.get("traffic_partition") == job.evaluation_split, "evaluation provenance drifted")
        _require(provenance.get("model_environment_spaces_match") is True, "space contract mismatch")
        _require(actions.get("schema_version") == "topo-scene-v4.8.action-diagnostics/v1", "action diagnostic schema drifted")
        _require(actions.get("aligned_with_paper_evaluation") is True, "action trace is not aligned")
        _require(actions.get("evaluation_seed_start") == job.evaluation_seed_start and actions.get("episodes") == job.evaluation_episodes, "action trace seed/episodes drifted")
        trace_path = root / "action_diagnostics_decisions.jsonl"
        _require(actions.get("trace", {}).get("sha256") == _sha256(trace_path), "action trace hash mismatch")
        decoder = selector["selected_deployment_decoder"]
        _require(actions.get("selected_deployment_decoder") == decoder, "action decoder differs from selector")
        trace_rows = _load_trace(trace_path)
        if job.algorithm == V48_CANDIDATE:
            _require(actions.get("hybrid_exact_lateral_code_rate") == 1.0, "candidate emitted non-exact lane code")
            _require(actions.get("selected_decoder_records", 0) > 0, "selected decoder trace is empty")
            _require(actions.get("selected_decoder_exact_rule_match_rate") == 1.0, "selected decoder rule mismatch")
            _require(actions.get("selected_decoder_exact_action_match_rate") == 1.0, "selected decoder action mismatch")
            _require(actions.get("selected_action_mask_feasible_rate") == 1.0, "selected decoder chose infeasible action")
            if decoder == TARGET_DECODER:
                _require(_target_integrity(trace_rows)["selected_decoder_exact_rule_match_rate"] == 1.0, "target rule does not recompute")
                _require(actions.get("exact_target_critic_argmax_rate") == 1.0, "target argmax drifted")
                _require(actions.get("target_keep_tie_rule_match_rate") == 1.0, "target keep tie drifted")
            elif decoder == FUSION_DECODER:
                _require(fusion_decoder_metrics(trace_rows)["exact_fusion_rule_match_rate"] == 1.0, "fusion rule does not recompute")
                _require(actions.get("actor_non_keep_confidence_threshold") == 0.90, "fusion threshold drifted")
                _require(actions.get("actor_override_predicate_valid_rate") == 1.0, "fusion override predicate drifted")
                _require(actions.get("target_fallback_rule_match_rate") == 1.0, "fusion fallback drifted")
            else:
                raise V48ProtocolError("candidate selected an unsupported decoder")
            entropy = training.get("statistics", {}).get("hybrid/effective_lane_entropy_coefficient", {})
            _require(entropy.get("last") == 0.0, "effective lane entropy coefficient is nonzero")
        else:
            _require(decoder == PARENT_DECODER, "TemporalGraph decoder drifted")
        _require(training.get("raw_steps") == job.raw_steps and int(training.get("learner_updates", 0)) > 0, "training diagnostics drifted")
        _require(returns.get("diagnostics_computed_from_runtime_buffer") is True, "return diagnostics are not runtime-derived")
        _require(checkpoints.get("expected_checkpoint_count") == 1, "checkpoint count mismatch")
        rows = checkpoints.get("checkpoints", [])
        _require(len(rows) == 1 and rows[0].get("raw_step") == job.raw_steps and rows[0].get("zip_crc_ok") is True, "checkpoint audit drifted")
        _require(performance.get("raw_steps") == job.raw_steps, "performance clock drifted")
        scientific._verify_zip(root / "final_model.zip")
        scientific._verify_zip(root / "selected_model.zip")
        _require(all(scientific._finite_tree(value) for value in (detailed, actions, training, performance, returns)), "non-finite run artifact")
    except (OSError, KeyError, TypeError, ValueError, V48ProtocolError, json.JSONDecodeError) as exc:
        return False, str(exc)
    return True, "accepted"


def accepted_run(job: Job, hashes: dict[str, str]) -> bool:
    return accepted_run_reason(job, hashes)[0]


_PARENT_RUN_ROW = scientific._run_row


def _run_row(job: Job) -> dict[str, Any]:
    row = _PARENT_RUN_ROW(job)
    receipt = _load_json(run_directory(job) / "selector" / "receipt.json")
    row.update(
        {
            "secondary_calibration_triggered": receipt.get("secondary_calibration_triggered", False),
            "initial_empirical_tied_top_count": receipt.get("initial_empirical_tied_top_count"),
            "initial_empirical_tied_top_pairs": receipt.get("initial_empirical_tied_top_pairs", []),
            "secondary_calibration_seed_start": receipt.get("secondary_calibration_seed_start"),
            "secondary_candidate_count": len(receipt.get("secondary_candidates", [])),
            "combined_tied_candidates": receipt.get("combined_tied_candidates", {}),
            "parent_v4_7_counterfactual_selected_pair": receipt.get("parent_v4_7_counterfactual_selected_pair"),
        }
    )
    return row


def _assert_stage_can_run(hashes: dict[str, str], stage: str) -> None:
    validate_implementation_freeze()
    _require(DEFAULT_ENGINEERING_RECEIPT.is_file(), "missing engineering receipt")
    engineering = _load_json(DEFAULT_ENGINEERING_RECEIPT)
    _require(engineering.get("status") == "passed", "engineering gate did not pass")
    _require(engineering.get("implementation_freeze_sha256") == hashes["implementation_freeze_sha256"], "engineering freeze mismatch")
    if stage == "promotion":
        decision = _load_json(stage_root("development") / "development_decision.json")
        _require(decision.get("decision") == "pass", "development gate did not pass")
        _require(decision.get("implementation_freeze_sha256") == hashes["implementation_freeze_sha256"], "development freeze mismatch")
    if stage == "formal":
        receipt = _load_json(DEFAULT_PROMOTION_GATE)
        _require(receipt.get("decision") == "pass", "formal test locked: promotion failed")
        _require(receipt.get("formal_algorithms") == list(V48_FORMAL_ALGORITHMS), "formal method pair drifted")
        for key, value in hashes.items():
            if key != "stage0_attribution_sha256":
                _require(receipt.get(key) == value, f"formal test locked: {key} mismatch")


@contextmanager
def _patched_scientific_runner() -> Iterator[None]:
    replacements = {
        "DEFAULT_CONTRACT": DEFAULT_CONTRACT,
        "DEFAULT_PREREGISTRATION": DEFAULT_PREREGISTRATION,
        "DEFAULT_ENGINEERING_ROOT": DEFAULT_ENGINEERING_ROOT,
        "DEFAULT_FREEZE": DEFAULT_FREEZE,
        "DEFAULT_ENGINEERING_RECEIPT": DEFAULT_ENGINEERING_RECEIPT,
        "DEFAULT_PROMOTION_GATE": DEFAULT_PROMOTION_GATE,
        "METHOD_ALGORITHMS": METHOD_ALGORITHMS,
        "METHOD_CODES": METHOD_CODES,
        "V47_CANDIDATE": V48_CANDIDATE,
        "V47_FORMAL_ALGORITHMS": V48_FORMAL_ALGORITHMS,
        "V47_IMPLEMENTATION_IDS": V48_IMPLEMENTATION_IDS,
        "FULL_RUN_REQUIRED_FILES": FULL_RUN_REQUIRED_FILES,
        "JOINT_METHODS": {"selected_v4_candidate"},
        "jobs_for_stage": jobs_for_stage,
        "stage_root": stage_root,
        "run_directory": run_directory,
        "command_for": command_for,
        "validate_implementation_freeze": validate_implementation_freeze,
        "accepted_run_reason": accepted_run_reason,
        "accepted_run": accepted_run,
        "_validate_selector_evidence": _validate_selector_evidence,
        "_validate_return_estimator": _validate_return_estimator,
        "_run_row": _run_row,
        "_assert_stage_can_run": _assert_stage_can_run,
        "_write_json": _write_json,
    }
    originals = {name: getattr(scientific, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(scientific, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(scientific, name, value)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    root.add_argument("--stage0-results", type=Path, default=DEFAULT_STAGE0_RESULTS)
    root.add_argument("--stage0-attribution", type=Path, default=DEFAULT_STAGE0_ATTRIBUTION)
    root.add_argument("--failure-attribution", type=Path, default=DEFAULT_FAILURE_ATTRIBUTION)
    root.add_argument("--deep-attribution", type=Path, default=DEFAULT_DEEP_ATTRIBUTION)
    root.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    plan = sub.add_parser("plan")
    plan.add_argument("--stage", choices=STAGES, required=True)
    plan.add_argument("--device", default="cuda")
    run = sub.add_parser("run")
    run.add_argument("--stage", choices=STAGES, required=True)
    run.add_argument("--job", default=None)
    run.add_argument("--device", default="cuda")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--dry-run", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("--stage", choices=STAGES, required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--stage", choices=STAGES, required=True)
    sub.add_parser("status")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    contract_path = args.contract.resolve()
    contract = validate_contract(load_contract(contract_path))
    digest = _sha256(contract_path)
    if args.command == "validate":
        values = {
            **scientific.parent._validate_stage0(args.stage0_results.resolve(), args.stage0_attribution.resolve()),
            **_validate_failure_attribution(args.failure_attribution.resolve(), args.deep_attribution.resolve()),
        }
        print(json.dumps({"status": "valid", "experiment_contract_sha256": digest, **values}, ensure_ascii=False, indent=2))
        return 0
    hashes = protocol_hashes(
        contract_path=contract_path,
        stage0_results=args.stage0_results.resolve(),
        stage0_attribution=args.stage0_attribution.resolve(),
        failure_attribution=args.failure_attribution.resolve(),
        deep_attribution=args.deep_attribution.resolve(),
        freeze_path=args.freeze.resolve(),
    )
    with _patched_scientific_runner():
        if args.command == "plan":
            value = scientific.write_plan(contract, digest, args.stage, hashes, device=args.device)
        elif args.command == "run":
            return scientific.execute_jobs(contract, digest, args.stage, hashes, selector=args.job, device=args.device, workers=args.workers, dry_run=args.dry_run)
        elif args.command == "summarize":
            value = scientific.summarize_stage(contract, digest, args.stage, hashes)
        elif args.command == "gate":
            if args.stage == "development":
                value = scientific.write_development_decision(contract, digest, hashes)
            elif args.stage == "promotion":
                value = scientific.compute_promotion_gate(contract, digest, hashes)
            else:
                value = scientific.compute_formal_decision(contract, digest, hashes)
        else:
            value = scientific.status_payload(contract, digest, hashes)
    print(json.dumps(_version_tree(value), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
