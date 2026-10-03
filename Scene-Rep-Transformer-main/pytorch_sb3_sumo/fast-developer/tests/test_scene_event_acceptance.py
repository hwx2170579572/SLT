"""Fail-closed acceptance receipt tests; no environment or training runs."""
import json
import hashlib
import math
from pathlib import Path
import sys
import zipfile

import pytest


FAST_DEVELOPER = Path(__file__).resolve().parents[1]
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

from scene_event import PROTOCOL_ID  # noqa: E402
from scene_event import acceptance  # noqa: E402
from scene_event.protocol import ExperimentConfig  # noqa: E402


METHOD = "sac_scene_dualgraph_v1"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _source_archive(root: Path, *, include_deadline_validator: bool) -> dict:
    rows = []
    repository = FAST_DEVELOPER.parents[1]
    paths = acceptance._runtime_source_paths(include_deadline_validator=include_deadline_validator)
    archive = root / "source_snapshot.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in paths:
            relative = path.relative_to(repository).as_posix()
            rows.append({"path": str(path.resolve()), "archive_path": relative,
                         "sha256": acceptance.sha256(path)})
            output.write(path, relative)
    return {"archive_sha256": acceptance.sha256(archive), "sources": rows}


def _episode(seed: int, outcome: str, checkpoint_sha: str) -> dict:
    outcomes = {key: key == outcome for key in ("success", "collision", "off_route", "timeout")}
    return {
        "phase": "eval", "episode": seed - 10_000, "seed": seed, "completed": True,
        "terminated": True, "truncated": False, **outcomes,
        "raw_steps": 123 if seed == 10_000 else 600,
        "decision_steps": 41 if seed == 10_000 else 200,
        "environment_step_reward_v2_return": 0.0,
        "reward_success": 0.0, "reward_collision": 0.0, "reward_off_route": 0.0,
        "reward_timeout": 0.0, "reward_step_cost": 0.0, "reward_progress": 0.0,
        "reward_component_error": 0.0, "max_component_step_error": 0.0,
        "checkpoint_sha256": checkpoint_sha,
    }


def _smoke_root(root: Path, *, method: str = METHOD, updates: int = 1,
                checkpoint_matches: bool = True, max_reward_error: float = 0.0) -> Path:
    root.mkdir(parents=True)
    config = ExperimentConfig(method=method, raw_steps=1200, eval_episodes=2,
                              eval_seed_start=10_000, learning_starts_raw=128)
    _write_json(root / "config.json", config.to_dict())
    checkpoint = root / "final_model.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    checkpoint_sha = acceptance.sha256(checkpoint)
    if not checkpoint_matches:
        checkpoint_sha = "0" * 64

    rows = [_episode(10_000, "success", checkpoint_sha),
            _episode(10_001, "timeout", checkpoint_sha)]
    result = {
        "method": method, "protocol": PROTOCOL_ID, "kind": "smoke",
        "scenario": "intersection_sorted_depart4p0", "seed": 0, "fresh": True,
        "raw_steps": 1200, "decision_steps": 400, "updates": updates,
        "replay_used_raw": 1200, "omitted_tail_raw": 0,
        "completed_eval_episodes": 2, "eval_seeds": [10_000, 10_001],
        "checkpoint_sha256": checkpoint_sha,
        "counts": {"success": 1, "collision": 0, "off_route": 0, "timeout": 1},
        "max_reward_component_error": max_reward_error,
    }
    _write_json(root / "result.json", result)
    _write_json(root / "status.json", {
        "state": "completed", "kind": "smoke", "method": method, "protocol": PROTOCOL_ID,
        "config_sha256": config.digest(), "raw_steps": 1200, "decision_steps": 400,
        "updates": updates,
    })
    runtime = _source_archive(root, include_deadline_validator=False)
    runtime.update({"fresh_training": True, "training_started": True,
                    "resume": False, "protocol": PROTOCOL_ID})
    _write_json(root / "runtime_provenance.json", runtime)
    (root / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_json(root / "diagnostic_summary.json", {
        "shadow": {"errors": 0, "train_unique_states": 0, "eval_unique_states": 2,
                   "additional_simulator_steps": 0,
                   "eval_states_per_episode": {"0": 1, "1": 1}},
    })
    for phase in ("train", "eval"):
        _write_json(root / f"factual_{phase}" / "manifest.json", {
            "accepted_anchors": 1, "written_anchors": 1,
            "input_contains_future": False, "valid_actor_future_samples": 1,
            "explicit_tick_axis_required": True, "missing_tick_axis_calls": 0,
        })
    return root


def _deadline_root(root: Path) -> Path:
    root.mkdir(parents=True)
    config = ExperimentConfig(device="cpu", raw_steps=600, eval_episodes=0)
    _write_json(root / "config.json", config.to_dict())
    trace = [{"raw": 3 * index, "collector_tick": 1000 + 3 * index,
              "remaining_seconds": 60.0 - 0.3 * index}
             for index in range(1, 201)]
    result = {
        "passed": True, "kind": "deadline_contract", "protocol": PROTOCOL_ID,
        "training_started": False, "additional_policy_evaluation_episodes": 0,
        "contract_test_episodes": 1, "completed": True,
        "raw_steps": 600, "decision_steps": 200,
        "success": False, "collision": False, "off_route": False, "timeout": True,
        "terminated": True, "truncated": False,
        "reward_component_error": 0.0, "max_component_step_error": 0.0,
        "environment_step_reward_v2_return": 0.0,
        "reward_success": 0.0, "reward_collision": 0.0, "reward_off_route": 0.0,
        "reward_timeout": 0.0, "reward_step_cost": 0.0, "reward_progress": 0.0,
        "trace": trace,
    }
    _write_json(root / "result.json", result)
    _write_json(root / "status.json", {
        "state": "completed", "kind": "deadline_contract", "training_started": False,
        "config_sha256": config.digest(), "raw_steps": 600, "decision_steps": 200,
    })
    runtime = _source_archive(root, include_deadline_validator=True)
    runtime.update({"fresh_training": False, "training_started": False,
                    "resume": False, "protocol": PROTOCOL_ID})
    _write_json(root / "runtime_provenance.json", runtime)
    return root


def _geometry_audit(root: Path, *, adjacent_class: str = "evidence_backed_adjacent_exit_topology_only") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    npz = root / "map.npz"
    npz.write_bytes(b"fixture public map")
    network = FAST_DEVELOPER.parent / "envs" / "sumo" / "original_scenarios_v1" / "intersection_sorted" / "map.net.xml"
    source_paths = {
        "map_cache.py": FAST_DEVELOPER / "scene_event" / "map_cache.py",
        "geometry.py": FAST_DEVELOPER / "scene_event" / "geometry.py",
        "collector.py": FAST_DEVELOPER / "scene_event" / "collector.py",
        "topology_graph.py": FAST_DEVELOPER.parent / "envs" / "sumo" / "topology_graph.py",
        "topology_graph_v2.py": FAST_DEVELOPER.parent / "envs" / "sumo" / "topology_graph_v2.py",
        "paper_scenario_registry.py": FAST_DEVELOPER.parent / "envs" / "sumo" / "paper_scenario_registry.py",
        "export_static_map.py": FAST_DEVELOPER / "analysis" / "scene_representation_redesign_20261003"
                               / "implementation_validation" / "export_static_map.py",
    }
    records = []
    class_counts = {
        "mapped_public_corridor_overlap_proxy": 82,
        "evidence_backed_foe_relation_without_corridor_overlap": 15,
        adjacent_class: 9,
        "unsupported_public_corridor_geometry": 0,
    }
    pair_index = 0
    for classification, count in class_counts.items():
        for _ in range(count):
            pair_index += 1
            row = {
                "pair_id": f"pair-{pair_index}", "lane_pair": [f"lane-{pair_index}-a", f"lane-{pair_index}-b"],
                "topology_relation": "merge" if classification == adjacent_class else "conflict",
                "classification": classification, "event_zone_created": True,
                "source_topology_relation_only": True, "candidate_relevant": True,
                "candidate_relevance_basis": "traffic-template participation was not used to filter legal topology candidates",
                "path_enumeration": {"left_path_count": 1, "right_path_count": 1,
                    "left_paths": [[f"lane-{pair_index}-a"]], "right_paths": [[f"lane-{pair_index}-b"]],
                    "expansion_truncated": False},
                "topology_source_evidence": [], "path_zone_evidence": [],
                "branch_pair_diagnostics": [], "maximum_intersection_area_m2": 1.0,
            }
            if classification == "mapped_public_corridor_overlap_proxy":
                row["path_zone_evidence"] = [{"intersection_area_m2": 1.0, "zone_ids": [pair_index]}]
            elif classification == "evidence_backed_foe_relation_without_corridor_overlap":
                row["event_zone_created"] = False
                row["maximum_intersection_area_m2"] = 0.0
                row["topology_source_evidence"] = [{
                    "evidence_source": "sumolib_node_areFoes", "request_indices": [2, 5],
                    "are_foes_forward": True, "are_foes_reverse": True,
                    "connections": [{"request_index": 2, "via_lane": row["lane_pair"][0]},
                                    {"request_index": 5, "via_lane": row["lane_pair"][1]}],
                }]
            elif classification in {"evidence_backed_adjacent_exit_topology_only",
                                     "evidence_backed_disjoint_adjacent_exit"}:
                row["event_zone_created"] = False
                row["maximum_intersection_area_m2"] = 0.0
                row["vehicle_footprint_clearance_status"] = "unknown_static_templates_omit_length_width"
                row["vehicle_dimensions_source"] = "Runtime vehicle dimensions from TraCI vehicle.getLength and vehicle.getWidth"
                row["topology_source_evidence"] = [{"evidence_source": "topology_graph_v2_merge_group"}]
                row["branch_pair_diagnostics"] = [{"compatible_exit_pair": True,
                    "adjacent_exit_lane_indices": True, "centerline_distance_m": 3.2}]
            records.append(row)
    pair_hash = hashlib.sha256(json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    no_zone = [row["pair_id"] for row in records if not row["event_zone_created"]]
    foe_ids = [row["pair_id"] for row in records
               if row["classification"] == "evidence_backed_foe_relation_without_corridor_overlap"]
    adjacent_ids = [row["pair_id"] for row in records
                    if row["classification"] == "evidence_backed_adjacent_exit_topology_only"]
    map_fingerprint = "fixture-map-fingerprint"
    topology_contract = {
        "key": "candidate_pair_topology_status", "dtype": "uint8",
        "axes": ["actor_i", "candidate_i", "actor_j", "candidate_j"],
        "shape_rule": "[max_actors, max_candidates, max_actors, max_candidates]",
        "bit_codes": {"1": "foe_without_zone", "2": "adjacent_exit_topology_only"},
        "bit_code_semantics": {
            "1": "public topology relation only; event and vehicle footprint clearance are unknown",
            "2": "adjacent exit topology only; event and vehicle footprint clearance are unknown",
        },
        "map_fingerprint": map_fingerprint, "pair_audit_sha256": pair_hash,
        "static_pair_counts": {"foe_without_zone": 15, "adjacent_exit_topology_only": 9},
        "foe_without_zone_pair_ids": foe_ids, "adjacent_exit_pair_ids": adjacent_ids,
        "static_endpoint_candidate_path_pair_counts": {
            "foe_without_zone": 18, "adjacent_exit_topology_only": 27,
        },
        "static_endpoint_candidate_path_pair_missing": [],
        "unknown_relation_is_safety_label": False, "physical_clearance_known": False,
        "zero_semantics": "no audited topology-only pair matched these currently valid candidate routes; does not mean safe or interaction-free",
        "dynamic_route_pair_hits": 0, "static_only_no_actor_observations": True,
    }
    report = {
        "geometry_audit_version": acceptance.GEOMETRY_AUDIT_VERSION,
        "scenario": "intersection_sorted_depart4p0", "network_path": str(network.resolve()),
        "network_sha256": acceptance.sha256(network), "public_map_fingerprint": map_fingerprint,
        "npz_path": str(npz.resolve()), "npz_sha256": acceptance.sha256(npz),
        "L_lanes": 46, "E_typed_edges": 348, "Z_zones": 90,
        "candidate_pair_topology_status_contract": topology_contract,
        "geometry_source_sha256": {name: acceptance.sha256(path) for name, path in source_paths.items()},
        "typed_pair_coverage": {
            "pair_audit_schema": "corridor_v2_pair_audit_v2",
            "record_count": len(records), **class_counts,
            "unsupported_candidate_relevant_pair_count": 0,
            "no_event_zone_candidate_relevant_count": len(no_zone),
            "no_event_zone_candidate_relevant_pair_ids": no_zone,
            "candidate_relevance_definition": "traffic-template participation was not used to filter legal topology candidates",
            "sumo_foe_classification_scope": "foe relation evidence is not a physical vehicle-contact oracle",
            "pair_audit_sha256": pair_hash, "pair_records": records,
        },
    }
    audit_path = root / "geometry.json"
    _write_json(audit_path, report)
    return audit_path


def _bind_geometry_contract(smoke_root: Path, audit_path: Path, *, m1_readout_delta: float = 0.0) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    typed = audit["typed_pair_coverage"]
    geometry = acceptance.verify_geometry_audit(audit_path)
    contract = {
        "key": "candidate_pair_topology_status", "dtype": "uint8",
        "axes": ["actor_i", "candidate_i", "actor_j", "candidate_j"],
        "shape": [24, 4, 24, 4], "bit_codes": {"1": "foe_without_zone", "2": "adjacent_exit_topology_only"},
        "map_fingerprint": geometry["public_map_fingerprint"],
        "pair_audit_sha256": geometry["pair_audit_sha256"],
        "static_pair_counts": {"foe_without_zone": 15, "adjacent_exit_topology_only": 9},
        "foe_without_zone_pair_ids": geometry["foe_without_zone_pair_ids"],
        "adjacent_exit_pair_ids": geometry["adjacent_exit_unknown_pair_ids"],
        "physical_clearance_known": False, "unknown_relation_is_safety_label": False,
        "zero_semantics": "no audited topology-only pair matched these currently valid candidate routes; does not mean safe or interaction-free",
    }
    _write_json(smoke_root / "scene_metadata.json", {
        "public_map": {"fingerprint": geometry["public_map_fingerprint"],
                       "network_sha256": geometry["network_sha256"],
                       "zone_count": geometry["zone_count"], "lane_count": geometry["lane_count"],
                       "typed_edge_count": geometry["typed_edge_count"],
                       "zone_geometry": {"pair_relation_audit": typed["pair_records"]}},
        "candidate_pair_topology_status_contract": contract,
    })
    (smoke_root / "observation_audit.jsonl").write_text(json.dumps({
        "phase": "train", "collector": {"candidate_pair_topology_status_contract": {
            "map_fingerprint": geometry["public_map_fingerprint"],
            "pair_audit_sha256": geometry["pair_audit_sha256"],
            "dynamic_foe_route_pair_hits": 1,
            "dynamic_adjacent_exit_route_pair_hits": 1,
        }},
    }) + "\n", encoding="utf-8")
    if json.loads((smoke_root / "result.json").read_text(encoding="utf-8"))["method"] == "sac_scene_eventgraph_cv_v1":
        (smoke_root / "updates.jsonl").write_text(json.dumps({"encoder": {
            "topology_relation_pairs": 1.0, "topology_relation_foe_pairs": 1.0,
            "topology_relation_merge_pairs": 1.0, "topology_relation_branches": 1.0,
            "topology_relation_token_norm": 1.0,
            "topology_relation_readout_delta_norm": m1_readout_delta,
        }}) + "\n", encoding="utf-8")
    return geometry


def _receipt(smoke_root: Path, source_hashes: dict[str, str]) -> dict:
    return {
        "protocol": PROTOCOL_ID, "source_hashes": source_hashes,
        "unit_tests_exit_code": 0, "passed_methods": [METHOD],
        "smoke_roots": {METHOD: str(smoke_root)},
        "deadline_contract_root": "must-not-be-read-if-smoke-fails",
    }


def test_receipt_rejects_stale_source_hashes_before_reading_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(acceptance, "current_sources", lambda: {"scene_event/policy.py": "new"})
    receipt_path = tmp_path / "receipt.json"
    _write_json(receipt_path, _receipt(tmp_path / "missing-smoke", {"scene_event/policy.py": "old"}))

    with pytest.raises(ValueError, match="stale or belongs to another protocol"):
        acceptance.validate_receipt(receipt_path, METHOD)


def test_receipt_source_set_covers_parent_wrapper_forwarding_test() -> None:
    wrapper_test = FAST_DEVELOPER.parent / "tests_sb3_sumo" / "test_scene_event_wrapper_forwarding.py"
    assert wrapper_test.resolve() in acceptance._receipt_source_paths()


def test_smoke_fixture_meets_current_positive_gates(tmp_path: Path) -> None:
    assert acceptance.verify_smoke(_smoke_root(tmp_path / "smoke"), METHOD)["updates"] == 1


def test_smoke_rejects_missing_runtime_source_even_if_archive_is_self_consistent(tmp_path: Path) -> None:
    root = _smoke_root(tmp_path / "smoke")
    runtime_path = root / "runtime_provenance.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["sources"] = runtime["sources"][1:]
    _write_json(runtime_path, runtime)
    with pytest.raises(ValueError, match="omits or mismatches required source"):
        acceptance.verify_smoke(root, METHOD)


def test_smoke_rejects_nan_reward_error(tmp_path: Path) -> None:
    root = _smoke_root(tmp_path / "smoke", max_reward_error=math.nan)
    with pytest.raises(ValueError, match="finite nonnegative error"):
        acceptance.verify_smoke(root, METHOD)


def test_smoke_rejects_nan_episode_reward_error(tmp_path: Path) -> None:
    root = _smoke_root(tmp_path / "smoke")
    episode_path = root / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["max_component_step_error"] = math.nan
    episode_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="finite nonnegative error"):
        acceptance.verify_smoke(root, METHOD)


def test_geometry_audit_accepts_explicit_topology_unknown_but_not_safe_no_event(tmp_path: Path) -> None:
    summary = acceptance.verify_geometry_audit(_geometry_audit(tmp_path / "geometry"))
    assert len(summary["adjacent_exit_unknown_pair_ids"]) == 9
    with pytest.raises(ValueError, match="unknown pair classification"):
        acceptance.verify_geometry_audit(_geometry_audit(
            tmp_path / "misclassified", adjacent_class="evidence_backed_disjoint_adjacent_exit"))


def test_geometry_audit_rejects_old_pair_schema(tmp_path: Path) -> None:
    audit_path = _geometry_audit(tmp_path / "geometry")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["typed_pair_coverage"]["pair_audit_schema"] = "corridor_v2_pair_audit_v1"
    _write_json(audit_path, audit)
    with pytest.raises(ValueError, match="missing typed pair records"):
        acceptance.verify_geometry_audit(audit_path)


@pytest.mark.parametrize(("mode", "message"), (
    ("missing", "source set is incomplete or unexpected"),
    ("stale", "Geometry audit source is stale: topology_graph_v2.py"),
))
def test_geometry_audit_fails_closed_on_missing_or_stale_geometry_source(
    tmp_path: Path, mode: str, message: str,
) -> None:
    audit_path = _geometry_audit(tmp_path / "geometry")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if mode == "missing":
        audit["geometry_source_sha256"].pop("topology_graph_v2.py")
    else:
        audit["geometry_source_sha256"]["topology_graph_v2.py"] = "0" * 64
    _write_json(audit_path, audit)
    with pytest.raises(ValueError, match=message):
        acceptance.verify_geometry_audit(audit_path)


@pytest.mark.parametrize(("field", "bad_value"), (
    ("pair_audit_sha256", "0" * 64),
    ("adjacent_exit_pair_ids", []),
    ("physical_clearance_known", True),
    ("static_endpoint_candidate_path_pair_missing", ["unresolved-pair"]),
))
def test_geometry_audit_rejects_mismatched_static_topology_contract(
    tmp_path: Path, field: str, bad_value: object,
) -> None:
    audit_path = _geometry_audit(tmp_path / "geometry")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["candidate_pair_topology_status_contract"][field] = bad_value
    _write_json(audit_path, audit)
    with pytest.raises(ValueError, match="does not bind the static topology-only observation contract"):
        acceptance.verify_geometry_audit(audit_path)


def test_smoke_rejects_geometry_unknown_missing_from_observation_contract(tmp_path: Path) -> None:
    root = _smoke_root(tmp_path / "smoke")
    geometry = acceptance.verify_geometry_audit(_geometry_audit(tmp_path / "geometry"))
    audit = json.loads(Path(geometry["path"]).read_text(encoding="utf-8"))
    _write_json(root / "scene_metadata.json", {
        "public_map": {"fingerprint": geometry["public_map_fingerprint"],
                       "network_sha256": geometry["network_sha256"], "zone_count": geometry["zone_count"],
                       "lane_count": geometry["lane_count"], "typed_edge_count": geometry["typed_edge_count"],
                       "zone_geometry": {"pair_relation_audit": audit["typed_pair_coverage"]["pair_records"]}},
    })
    with pytest.raises(ValueError, match="does not expose the audited adjacent-exit unknown"):
        acceptance.verify_smoke(root, METHOD, geometry)


def test_m1_smoke_requires_unknown_relation_readout_evidence(tmp_path: Path) -> None:
    method = "sac_scene_eventgraph_cv_v1"
    root = _smoke_root(tmp_path / "m1", method=method)
    audit_path = _geometry_audit(tmp_path / "geometry")
    geometry = _bind_geometry_contract(root, audit_path, m1_readout_delta=0.25)
    assert acceptance.verify_smoke(root, method, geometry)["method"] == method

    root2 = _smoke_root(tmp_path / "m1-zero-readout", method=method)
    geometry2 = _bind_geometry_contract(root2, audit_path, m1_readout_delta=0.0)
    with pytest.raises(ValueError, match="did not show unknown topology-relation messages"):
        acceptance.verify_smoke(root2, method, geometry2)


def test_smoke_rejects_zero_update_run(tmp_path: Path) -> None:
    root = _smoke_root(tmp_path / "smoke", updates=0)
    with pytest.raises(ValueError, match="Smoke must finish updates"):
        acceptance.verify_smoke(root, METHOD)


def test_smoke_rejects_wrong_checkpoint_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Smoke checkpoint identity mismatch"):
        acceptance.verify_smoke(_smoke_root(tmp_path / "smoke", checkpoint_matches=False), METHOD)


def test_deadline_acceptance_accepts_complete_nontraining_contract(tmp_path: Path) -> None:
    result = acceptance.verify_deadline(_deadline_root(tmp_path / "deadline"))
    assert result["raw_steps"] == 600
    assert result["decision_steps"] == 200


@pytest.mark.parametrize(
    ("artifact_path", "field", "bad_value"),
    (
        ("result.json", "training_started", True),
        ("result.json", "additional_policy_evaluation_episodes", 1),
        ("status.json", "kind", "smoke"),
        ("runtime_provenance.json", "training_started", True),
    ),
)
def test_deadline_acceptance_rejects_wrong_scope_metadata(
    tmp_path: Path, artifact_path: str, field: str, bad_value: object,
) -> None:
    root = _deadline_root(tmp_path / "deadline")
    artifact = root / artifact_path
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload[field] = bad_value
    _write_json(artifact, payload)

    with pytest.raises(ValueError):
        acceptance.verify_deadline(root)


@pytest.mark.parametrize(("field", "bad_value"),
                         (("reward_component_error", math.nan), ("reward_progress", math.inf)))
def test_deadline_acceptance_rejects_nonfinite_reward_accounting(
    tmp_path: Path, field: str, bad_value: float,
) -> None:
    root = _deadline_root(tmp_path / "deadline")
    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result[field] = bad_value
    _write_json(result_path, result)
    with pytest.raises(ValueError):
        acceptance.verify_deadline(root)
