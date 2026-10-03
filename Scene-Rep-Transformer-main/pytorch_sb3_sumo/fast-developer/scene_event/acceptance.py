"""Prevent full runs from silently bypassing this route's implementation gates."""
import json
import hashlib
import math
import os
from collections import Counter
from pathlib import Path
import zipfile

from . import PROTOCOL_ID
from .provenance import sha256

REWARD_ERROR_TOLERANCE = 2e-5
GEOMETRY_AUDIT_VERSION = "corridor_v2_audit_v2"
EXPECTED_PAIR_CLASSES = {
    "mapped_public_corridor_overlap_proxy",
    "evidence_backed_foe_relation_without_corridor_overlap",
    "evidence_backed_adjacent_exit_topology_only",
}


def _fd_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_source_paths(*, include_deadline_validator: bool = False) -> list[Path]:
    """Sources that RunArtifacts.snapshot must include for these executions."""
    root = _fd_root()
    project = root.parent
    # acceptance.py is executed by the separate receipt checker, not imported
    # by a bounded smoke process. The rest of this package is loaded by the
    # runner/encoder/environment and must be present in its runtime snapshot.
    paths = {path for path in (root / "scene_event").glob("*.py")
             if path.name != "acceptance.py"}
    paths.add(root / "train_scene_event_v1.py")
    # These modules define the public lane graph, environment wrapper, scenario
    # factory, and reward actually used by the independent runner.
    paths.update({
        project / "envs" / "sumo" / "topology_graph.py",
        project / "envs" / "sumo" / "topology_graph_v2.py",
        project / "envs" / "sumo" / "sumo_env.py",
        project / "envs" / "sumo" / "paper_env.py",
        project / "envs" / "sumo" / "paper_scenario_registry.py",
        project / "envs" / "sumo" / "independent_v2_five_methods_six_scenarios_100ep_v1.py",
        project / "tools" / "train_independent_v2_5m6s100e_v1.py",
        root / "reward_shaping_v2.py",
    })
    if include_deadline_validator:
        paths.add(root / "validate_scene_event_deadline_v1.py")
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError("Required runtime source is missing: " + ", ".join(missing))
    return sorted(path.resolve() for path in paths)


def _receipt_source_paths() -> list[Path]:
    """Code and tests whose exact identity is covered by an acceptance receipt."""
    root = _fd_root()
    paths = set(_runtime_source_paths(include_deadline_validator=True))
    paths.update({root / "scene_event" / "acceptance.py",
                  root / "validate_scene_event_v1.py", root / "launch_scene_event_pair_v1.py"})
    paths.add(root / "analysis" / "scene_representation_redesign_20261003" /
              "implementation_validation" / "export_static_map.py")
    for directory in (root.parent / "tests_sb3_sumo", root / "tests", root):
        if directory.is_dir():
            paths.update(directory.glob("test_scene_event_*.py"))
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError("Required receipt source is missing: " + ", ".join(missing))
    return sorted(path.resolve() for path in paths)


def _source_label(path: Path) -> str:
    return os.path.relpath(path.resolve(), _fd_root()).replace(os.sep, "/")


def _config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _finite_nonnegative_error(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite numeric error")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative error")
    if numeric > REWARD_ERROR_TOLERANCE:
        raise ValueError(f"{label} exceeded the reward accounting tolerance")
    return numeric


def _verify_runtime_snapshot(root: Path, runtime: dict, *, include_deadline_validator: bool = False) -> None:
    archive = root / "source_snapshot.zip"
    if not archive.is_file() or sha256(archive) != runtime.get("archive_sha256"):
        raise ValueError("Source archive identity mismatch")
    rows = runtime.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Runtime source manifest is missing")
    by_path: dict[Path, dict] = {}
    archive_names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not all(key in row for key in ("path", "archive_path", "sha256")):
            raise ValueError("Runtime source manifest row is incomplete")
        path = Path(row["path"]).resolve()
        archive_name = str(row["archive_path"])
        if path in by_path or archive_name in archive_names:
            raise ValueError("Runtime source manifest contains duplicate entries")
        by_path[path] = row
        archive_names.add(archive_name)
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"Source changed since artifact creation: {path}")
    expected = _runtime_source_paths(include_deadline_validator=include_deadline_validator)
    for path in expected:
        row = by_path.get(path)
        if row is None or row["sha256"] != sha256(path):
            raise ValueError(f"Runtime source manifest omits or mismatches required source: {path}")
    try:
        with zipfile.ZipFile(archive, "r") as source_zip:
            namelist = source_zip.namelist()
            names = set(namelist)
            if len(namelist) != len(names) or names != archive_names:
                raise ValueError("Archive entries disagree with the runtime source manifest")
            for row in rows:
                payload = source_zip.read(row["archive_path"])
                if sha256_bytes(payload) != row["sha256"]:
                    raise ValueError(f"Archived source content disagrees with manifest: {row['archive_path']}")
    except (OSError, RuntimeError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("Runtime source archive cannot be verified") from exc


def sha256_bytes(payload: bytes) -> str:
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def current_sources() -> dict[str, str]:
    return {_source_label(path): sha256(path) for path in _receipt_source_paths()}


def verify_geometry_audit(path: Path) -> dict:
    """Verify the static map artifact and reject unresolved task-relevant geometry."""
    path = path.resolve()
    payload = path.read_bytes()
    audit = json.loads(payload.decode("utf-8"))
    root = _fd_root()
    project = root.parent
    if audit.get("geometry_audit_version") != GEOMETRY_AUDIT_VERSION:
        raise ValueError("Geometry audit schema/version mismatch")
    if audit.get("scenario") != "intersection_sorted_depart4p0":
        raise ValueError("Geometry audit scenario mismatch")
    network = Path(audit.get("network_path", "")).resolve()
    expected_network = (project / "envs" / "sumo" / "original_scenarios_v1" /
                        "intersection_sorted" / "map.net.xml").resolve()
    if network != expected_network or not network.is_file() or sha256(network) != audit.get("network_sha256"):
        raise ValueError("Geometry audit network identity mismatch")
    npz = Path(audit.get("npz_path", "")).resolve()
    if npz.parent != path.parent or not npz.is_file() or sha256(npz) != audit.get("npz_sha256"):
        raise ValueError("Geometry audit NPZ identity mismatch")
    source_hashes = audit.get("geometry_source_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError("Geometry audit is missing source hashes")
    geometry_sources = {
        "map_cache.py": root / "scene_event" / "map_cache.py",
        "geometry.py": root / "scene_event" / "geometry.py",
        "collector.py": root / "scene_event" / "collector.py",
        "topology_graph.py": project / "envs" / "sumo" / "topology_graph.py",
        "topology_graph_v2.py": project / "envs" / "sumo" / "topology_graph_v2.py",
        "paper_scenario_registry.py": project / "envs" / "sumo" / "paper_scenario_registry.py",
        "export_static_map.py": root / "analysis" / "scene_representation_redesign_20261003"
                               / "implementation_validation" / "export_static_map.py",
    }
    if set(source_hashes) != set(geometry_sources):
        raise ValueError("Geometry audit source set is incomplete or unexpected")
    for name, source in geometry_sources.items():
        if source_hashes[name] != sha256(source):
            raise ValueError(f"Geometry audit source is stale: {name}")
    typed = audit.get("typed_pair_coverage")
    if (not isinstance(typed, dict)
            or typed.get("pair_audit_schema") != "corridor_v2_pair_audit_v2"
            or not isinstance(typed.get("pair_records"), list)):
        raise ValueError("Geometry audit is missing typed pair records")
    records = typed["pair_records"]
    if len(records) != 106 or typed.get("record_count") != 106:
        raise ValueError("Geometry audit must cover all 106 typed lane-pair relations")
    pair_hash = hashlib.sha256(json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    if pair_hash != typed.get("pair_audit_sha256"):
        raise ValueError("Geometry typed-pair audit hash mismatch")
    pair_ids = [row.get("pair_id") for row in records if isinstance(row, dict)]
    if len(pair_ids) != len(records) or len(set(pair_ids)) != len(records) or any(not item for item in pair_ids):
        raise ValueError("Geometry audit contains malformed or duplicate typed pair IDs")
    classes = Counter()
    no_zone_ids = []
    adjacent_unknown_ids = []
    for row in records:
        classification = row.get("classification")
        if classification not in EXPECTED_PAIR_CLASSES | {"unsupported_public_corridor_geometry"}:
            raise ValueError(f"Geometry audit has an unknown pair classification: {classification}")
        classes[classification] += 1
        if row.get("candidate_relevant") is not True:
            raise ValueError("A legal public lane-pair candidate was omitted from geometry coverage")
        if row.get("source_topology_relation_only") is not True:
            raise ValueError("Geometry audit does not preserve the topology-only interpretation boundary")
        relevance_basis = row.get("candidate_relevance_basis", "")
        if "traffic-template participation was not used" not in relevance_basis:
            raise ValueError("Geometry audit may have filtered legal candidates using traffic templates")
        paths = row.get("path_enumeration")
        if (not isinstance(paths, dict) or paths.get("expansion_truncated") is not False
                or paths.get("left_path_count", 99) > 4 or paths.get("right_path_count", 99) > 4
                or len(paths.get("left_paths", [])) != paths.get("left_path_count")
                or len(paths.get("right_paths", [])) != paths.get("right_path_count")):
            raise ValueError("Geometry audit has incomplete legal path enumeration")
        if classification == "unsupported_public_corridor_geometry":
            raise ValueError(f"Task-relevant typed pair remains unsupported: {row['pair_id']}")
        if classification == "mapped_public_corridor_overlap_proxy":
            evidence = row.get("path_zone_evidence")
            if row.get("event_zone_created") is not True or not evidence:
                raise ValueError(f"Mapped geometry pair lacks an event zone: {row['pair_id']}")
            if any(float(item.get("intersection_area_m2", 0.0)) <= 0.0 or not item.get("zone_ids")
                   for item in evidence):
                raise ValueError(f"Mapped geometry pair lacks positive-area zone evidence: {row['pair_id']}")
        elif classification == "evidence_backed_foe_relation_without_corridor_overlap":
            source_rows = [item for item in row.get("topology_source_evidence", [])
                           if item.get("evidence_source") == "sumolib_node_areFoes"]
            if (row.get("topology_relation") != "conflict" or row.get("event_zone_created") is not False
                    or float(row.get("maximum_intersection_area_m2", 0.0)) != 0.0
                    or not source_rows):
                raise ValueError(f"Foe relation lacks explicit non-overlap/topology evidence: {row['pair_id']}")
            for evidence in source_rows:
                requests = evidence.get("request_indices", [])
                connections = evidence.get("connections", [])
                if (evidence.get("are_foes_forward") is not True or evidence.get("are_foes_reverse") is not True
                        or len(requests) < 2 or len(connections) < 2
                        or {item.get("request_index") for item in connections} != set(requests)
                        or {item.get("via_lane") for item in connections} != set(row.get("lane_pair", []))):
                    raise ValueError(f"Foe relation lacks exact request-index evidence: {row['pair_id']}")
            no_zone_ids.append(row["pair_id"])
        else:
            # A topological convergence with no positive-area corridor zone is
            # not a physical no-collision proof. It can pass only with a
            # branch-specific positive vehicle-envelope clearance and a policy
            # observation contract that carries the relationship. The current
            # map only has 3.2 m centerline spacing and zero corridor clearance.
            if (row.get("topology_relation") != "merge" or row.get("event_zone_created") is not False
                    or row.get("maximum_intersection_area_m2") != 0.0
                    or classification != "evidence_backed_adjacent_exit_topology_only"
                    or row.get("vehicle_footprint_clearance_status") != "unknown_static_templates_omit_length_width"
                    or not all(token in str(row.get("vehicle_dimensions_source", "")).lower()
                               for token in ("traci", "getlength", "getwidth"))):
                raise ValueError(f"Adjacent-exit geometry is not explicitly classified as unknown: {row['pair_id']}")
            merge_evidence = [item for item in row.get("topology_source_evidence", [])
                              if item.get("evidence_source") == "topology_graph_v2_merge_group"]
            compatible_adjacent = [item for item in row.get("branch_pair_diagnostics", [])
                                   if item.get("compatible_exit_pair") is True
                                   and item.get("adjacent_exit_lane_indices") is True
                                   and float(item.get("centerline_distance_m", 0.0)) > 0.0]
            if not merge_evidence or not compatible_adjacent:
                raise ValueError(f"Adjacent-exit unknown lacks public branch/topology evidence: {row['pair_id']}")
            # This topology-only label preserves what the map proves. It is not
            # enough to pass until both smoke artifacts show this unknown as an
            # actor-pair input; verify_smoke applies that implementation gate.
            no_zone_ids.append(row["pair_id"])
            adjacent_unknown_ids.append(row["pair_id"])
    declared_counts = {key: typed.get(key, 0) for key in EXPECTED_PAIR_CLASSES |
                       {"unsupported_public_corridor_geometry"}}
    if any(declared_counts[key] != classes[key] for key in declared_counts):
        raise ValueError("Geometry pair classification totals disagree with records")
    expected_classes = {
        "mapped_public_corridor_overlap_proxy": 82,
        "evidence_backed_foe_relation_without_corridor_overlap": 15,
        "evidence_backed_adjacent_exit_topology_only": 9,
        "unsupported_public_corridor_geometry": 0,
    }
    if {key: classes[key] for key in expected_classes} != expected_classes:
        raise ValueError("Geometry pair classification coverage differs from the reviewed scenario contract")
    if typed.get("unsupported_candidate_relevant_pair_count") != 0:
        raise ValueError("Geometry audit reports unsupported task-relevant relations")
    if "traffic-template participation was not used" not in typed.get("candidate_relevance_definition", ""):
        raise ValueError("Geometry audit does not establish template-independent legal-candidate coverage")
    if "not a physical vehicle-contact oracle" not in typed.get("sumo_foe_classification_scope", ""):
        raise ValueError("SUMO foe relation is not bounded as topology evidence")
    if (typed.get("no_event_zone_candidate_relevant_count") != len(no_zone_ids)
            or set(typed.get("no_event_zone_candidate_relevant_pair_ids", [])) != set(no_zone_ids)):
        raise ValueError("Geometry no-zone coverage disagrees with typed-pair records")
    # The static audit must bind the topology-only relations to the exact
    # actor/candidate observation field that the runtime smoke will consume.
    # These bits carry public topology evidence only; they are neither a
    # collision label nor a clearance guarantee.
    topology = audit.get("candidate_pair_topology_status_contract")
    foe_ids = [row["pair_id"] for row in records
               if row["classification"] == "evidence_backed_foe_relation_without_corridor_overlap"]
    if (not isinstance(topology, dict)
            or topology.get("key") != "candidate_pair_topology_status"
            or topology.get("dtype") != "uint8"
            or topology.get("axes") != ["actor_i", "candidate_i", "actor_j", "candidate_j"]
            or topology.get("shape_rule") != "[max_actors, max_candidates, max_actors, max_candidates]"
            or topology.get("bit_codes") != {"1": "foe_without_zone", "2": "adjacent_exit_topology_only"}
            or topology.get("bit_code_semantics") != {
                "1": "public topology relation only; event and vehicle footprint clearance are unknown",
                "2": "adjacent exit topology only; event and vehicle footprint clearance are unknown",
            }
            or topology.get("map_fingerprint") != audit.get("public_map_fingerprint")
            or topology.get("pair_audit_sha256") != pair_hash
            or topology.get("static_pair_counts") != {
                "foe_without_zone": len(foe_ids),
                "adjacent_exit_topology_only": len(adjacent_unknown_ids),
            }
            or set(topology.get("foe_without_zone_pair_ids", [])) != set(foe_ids)
            or set(topology.get("adjacent_exit_pair_ids", [])) != set(adjacent_unknown_ids)
            or topology.get("static_endpoint_candidate_path_pair_counts") != {
                "foe_without_zone": 18, "adjacent_exit_topology_only": 27,
            }
            or topology.get("static_endpoint_candidate_path_pair_missing") != []
            or topology.get("unknown_relation_is_safety_label") is not False
            or topology.get("physical_clearance_known") is not False
            or topology.get("zero_semantics") != (
                "no audited topology-only pair matched these currently valid candidate routes; "
                "does not mean safe or interaction-free")
            or topology.get("dynamic_route_pair_hits") != 0
            or topology.get("static_only_no_actor_observations") is not True):
        raise ValueError("Geometry audit does not bind the static topology-only observation contract")
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
            "npz_sha256": audit["npz_sha256"],
            "network_sha256": audit["network_sha256"],
            "public_map_fingerprint": audit.get("public_map_fingerprint"),
            "pair_audit_sha256": pair_hash, "typed_pair_class_counts": dict(classes),
            "zone_count": audit.get("Z_zones"),
            "lane_count": audit.get("L_lanes"), "typed_edge_count": audit.get("E_typed_edges"),
            "adjacent_exit_unknown_pair_ids": adjacent_unknown_ids,
            "foe_without_zone_pair_ids": [row["pair_id"] for row in records
                                           if row["classification"] == "evidence_backed_foe_relation_without_corridor_overlap"]}


def verify_smoke(root: Path, method: str, geometry_audit: dict | None = None) -> dict:
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    runtime = json.loads((root / "runtime_provenance.json").read_text(encoding="utf-8"))
    if result["method"] != method or result["protocol"] != PROTOCOL_ID or result["kind"] != "smoke":
        raise ValueError("Smoke identity/protocol mismatch")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config_sha = _config_digest(config)
    if (status.get("state") != "completed" or status.get("kind") != "smoke"
            or status.get("method") != method or status.get("protocol") != PROTOCOL_ID
            or status.get("config_sha256") != config_sha):
        raise ValueError("Smoke status identity mismatch")
    if (config.get("method") != method or config.get("protocol") != PROTOCOL_ID
            or config.get("scenario") != "intersection_sorted_depart4p0"
            or config.get("seed") != 0 or config.get("raw_steps") != 1200 or config.get("eval_episodes") != 2
            or config.get("eval_seed_start") != 10_000 or config.get("deadline_seconds") != 60.0
            or config.get("deadline_is_terminal") is not True
            or config.get("action_repeat") != 3 or config.get("raw_dt") != 0.1
            or config.get("learning_starts_raw") != 128):
        raise ValueError("Smoke config does not match the bounded implementation contract")
    if (result.get("method") != method or result.get("protocol") != PROTOCOL_ID
            or result.get("kind") != "smoke" or result.get("fresh") is not True
            or result.get("scenario") != "intersection_sorted_depart4p0" or result.get("seed") != 0
            or result.get("raw_steps") != 1200 or result.get("eval_seeds") != [10_000, 10_001]
            or result.get("completed_eval_episodes") != 2
            or status.get("raw_steps") != result.get("raw_steps")
            or status.get("decision_steps") != result.get("decision_steps")
            or status.get("updates") != result.get("updates")):
        raise ValueError("Smoke result/status identity or bounded budget mismatch")
    integer_fields = ("raw_steps", "decision_steps", "updates", "replay_used_raw", "omitted_tail_raw")
    if any(type(result.get(key)) is not int for key in integer_fields):
        raise ValueError("Smoke raw/decision/update counters must be integer counts")
    if result["updates"] < 1:
        raise ValueError("Smoke must finish updates and at least two evaluation episodes")
    if (result["decision_steps"] < 1
            or result["replay_used_raw"] < 0 or result["omitted_tail_raw"] < 0
            or result["replay_used_raw"] + result["omitted_tail_raw"] != result["raw_steps"]):
        raise ValueError("Smoke raw/decision/replay accounting failed")
    if (runtime.get("fresh_training") is not True or runtime.get("training_started") is not True
            or runtime.get("resume") is not False or runtime.get("protocol") != PROTOCOL_ID):
        raise ValueError("Smoke runtime provenance is not a fresh training run")
    if result["updates"] < 1:
        raise ValueError("Smoke must finish updates and at least two evaluation episodes")
    _finite_nonnegative_error(result.get("max_reward_component_error"), "Smoke reward accounting error")
    if sha256(root / "final_model.pt") != result["checkpoint_sha256"]:
        raise ValueError("Smoke checkpoint identity mismatch")
    _verify_runtime_snapshot(root, runtime)
    if geometry_audit is not None:
        scene = json.loads((root / "scene_metadata.json").read_text(encoding="utf-8"))
        public_map = scene.get("public_map", {})
        if (public_map.get("fingerprint") != geometry_audit.get("public_map_fingerprint")
                or public_map.get("network_sha256") != geometry_audit.get("network_sha256")
                or public_map.get("zone_count") != geometry_audit.get("zone_count")
                or public_map.get("lane_count") != geometry_audit.get("lane_count")
                or public_map.get("typed_edge_count") != geometry_audit.get("typed_edge_count")):
            raise ValueError("Smoke public map does not match the audited geometry artifact")
        zone_geometry = public_map.get("zone_geometry", {})
        smoke_pairs = zone_geometry.get("pair_relation_audit")
        if not isinstance(smoke_pairs, list):
            raise ValueError("Smoke scene metadata omits the runtime typed-pair geometry audit")
        smoke_pair_hash = hashlib.sha256(json.dumps(
            smoke_pairs, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")).hexdigest()
        if smoke_pair_hash != geometry_audit.get("pair_audit_sha256"):
            raise ValueError("Smoke runtime pair geometry differs from the audited artifact")
        unknown_ids = geometry_audit.get("adjacent_exit_unknown_pair_ids", [])
        if unknown_ids:
            unknown_contract = scene.get("candidate_pair_topology_status_contract", {})
            if (unknown_contract.get("key") != "candidate_pair_topology_status"
                    or unknown_contract.get("adjacent_exit_pair_ids") != unknown_ids
                    or unknown_contract.get("foe_without_zone_pair_ids") != geometry_audit.get("foe_without_zone_pair_ids")
                    or unknown_contract.get("dtype") != "uint8"
                    or unknown_contract.get("axes") != ["actor_i", "candidate_i", "actor_j", "candidate_j"]
                    or unknown_contract.get("map_fingerprint") != geometry_audit.get("public_map_fingerprint")
                    or unknown_contract.get("pair_audit_sha256") != geometry_audit.get("pair_audit_sha256")
                    or unknown_contract.get("physical_clearance_known") is not False
                    or unknown_contract.get("unknown_relation_is_safety_label") is not False
                    or unknown_contract.get("zero_semantics") != (
                        "no audited topology-only pair matched these currently valid candidate routes; "
                        "does not mean safe or interaction-free")):
                raise ValueError("Smoke does not expose the audited adjacent-exit unknown relation contract")
            if unknown_contract.get("shape") != [24, 4, 24, 4]:
                raise ValueError("Smoke candidate-pair topology status tensor has the wrong shape")
            if unknown_contract.get("bit_codes") != {
                    "1": "foe_without_zone", "2": "adjacent_exit_topology_only"}:
                raise ValueError("Smoke candidate-pair topology status bit codes do not match the audit")
            if unknown_contract.get("static_pair_counts") != {
                    "foe_without_zone": len(geometry_audit.get("foe_without_zone_pair_ids", [])),
                    "adjacent_exit_topology_only": len(unknown_ids)}:
                raise ValueError("Smoke static topology-relation counts disagree with geometry audit")
            _verify_topology_status_runtime(root, method, geometry_audit)
    eval_rows = []
    episodes_path = root / "episodes.jsonl"
    if not episodes_path.is_file():
        raise ValueError("Smoke episode identity log is missing")
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("phase") == "eval":
                eval_rows.append(row)
    if len(eval_rows) != 2:
        raise ValueError("Smoke episode log does not contain exactly two evaluation episodes")
    counts = {key: 0 for key in ("success", "collision", "off_route", "timeout")}
    max_episode_reward_error = 0.0
    for expected_seed, row in zip((10_000, 10_001), eval_rows):
        if (row.get("seed") != expected_seed or row.get("completed") is not True
                or row.get("phase") != "eval" or row.get("checkpoint_sha256") != result["checkpoint_sha256"]
                or row.get("terminated") is not True or row.get("truncated") is not False
                or not isinstance(row.get("raw_steps"), int) or not 1 <= row["raw_steps"] <= 600
                or not isinstance(row.get("decision_steps"), int) or not 1 <= row["decision_steps"] <= 200):
            raise ValueError("Smoke evaluation episode identity/checkpoint mismatch")
        raw_outcomes = {key: row.get(key) for key in counts}
        if any(type(value) is not bool for value in raw_outcomes.values()):
            raise ValueError("Smoke evaluation outcome fields must be explicit booleans")
        outcomes = raw_outcomes
        if sum(outcomes.values()) != 1:
            raise ValueError("Smoke evaluation outcome is not mutually exclusive")
        for key, value in outcomes.items():
            counts[key] += int(value)
        reward_components = ("reward_success", "reward_collision", "reward_off_route",
                            "reward_timeout", "reward_step_cost", "reward_progress")
        component_total = 0.0
        for component in reward_components:
            value = row.get(component)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Smoke episode reward component {component} must be finite")
            component_total += float(value)
        shaped_return = row.get("environment_step_reward_v2_return")
        if (isinstance(shaped_return, bool) or not isinstance(shaped_return, (int, float))
                or not math.isfinite(float(shaped_return))
                or abs(component_total - float(shaped_return)) > REWARD_ERROR_TOLERANCE):
            raise ValueError("Smoke episode six-component reward does not reconcile")
        episode_error = _finite_nonnegative_error(row.get("reward_component_error"), "Smoke episode reward error")
        _finite_nonnegative_error(row.get("max_component_step_error"), "Smoke per-step reward error")
        max_episode_reward_error = max(max_episode_reward_error, episode_error)
    if result.get("counts") != counts or result.get("max_reward_component_error") != max_episode_reward_error:
        raise ValueError("Smoke summary disagrees with evaluation episode records")
    diagnostic = json.loads((root / "diagnostic_summary.json").read_text(encoding="utf-8"))
    shadow = diagnostic.get("shadow", {})
    if (type(shadow.get("errors")) is not int or shadow["errors"] != 0
            or type(shadow.get("train_unique_states")) is not int
            or not 0 <= shadow["train_unique_states"] <= 20
            or type(shadow.get("additional_simulator_steps")) is not int
            or shadow["additional_simulator_steps"] != 0):
        raise ValueError("Shadow error/budget gate failed")
    eval_budget = shadow.get("eval_states_per_episode", {})
    if (not isinstance(eval_budget, dict) or set(eval_budget) != {"0", "1"}
            or any(type(value) is not int or not 1 <= value <= 4 for value in eval_budget.values())
            or shadow.get("eval_unique_states") != sum(eval_budget.values())):
        raise ValueError("Evaluation shadow budget exceeded")
    for phase in ("train", "eval"):
        factual = json.loads((root / f"factual_{phase}" / "manifest.json").read_text(encoding="utf-8"))
        if factual["accepted_anchors"] != factual["written_anchors"] or factual["input_contains_future"]:
            raise ValueError("Factual-target accounting/input separation failed")
        if factual["accepted_anchors"] < 1 or factual["valid_actor_future_samples"] < 1:
            raise ValueError("Smoke did not exercise factual future-label collection")
        if not factual.get("explicit_tick_axis_required") or factual.get("missing_tick_axis_calls") != 0:
            raise ValueError("Factual labels did not use the actual per-frame physical clock")
    return result


def _verify_topology_status_runtime(root: Path, method: str, geometry_audit: dict) -> None:
    observed: dict[str, dict[str, int]] = {"train": {}, "eval": {}}
    audit_path = root / "observation_audit.jsonl"
    if not audit_path.is_file():
        raise ValueError("Smoke topology-relation observation audit is missing")
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        phase = row.get("phase")
        contract = row.get("collector", {}).get("candidate_pair_topology_status_contract", {})
        if phase not in observed or not contract:
            continue
        if (contract.get("map_fingerprint") != geometry_audit.get("public_map_fingerprint")
                or contract.get("pair_audit_sha256") != geometry_audit.get("pair_audit_sha256")):
            raise ValueError("Smoke dynamic topology audit uses a different map/pair audit")
        for key, field in (("foe_without_zone", "dynamic_foe_route_pair_hits"),
                           ("adjacent_exit_topology_only", "dynamic_adjacent_exit_route_pair_hits")):
            value = contract.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Smoke dynamic topology hit count {field} is malformed")
            observed[phase][key] = max(observed[phase].get(key, 0), value)
    if any(observed["train"].get(key, 0) < 1 for key in
           ("foe_without_zone", "adjacent_exit_topology_only")):
        raise ValueError("Training smoke did not exercise both audited topology-unknown channels")
    if method == "sac_scene_eventgraph_cv_v1":
        updates_path = root / "updates.jsonl"
        if not updates_path.is_file():
            raise ValueError("M1 smoke update diagnostics are missing")
        reached_readout = False
        for line in updates_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            encoder = row.get("encoder", {})
            fields = ("topology_relation_pairs", "topology_relation_foe_pairs",
                      "topology_relation_merge_pairs", "topology_relation_branches",
                      "topology_relation_token_norm", "topology_relation_readout_delta_norm")
            values = [encoder.get(field) for field in fields]
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) for value in values):
                continue
            if (values[0] > 0 and values[1] > 0 and values[2] > 0 and values[3] > 0
                    and values[4] > 0 and values[5] > 0):
                reached_readout = True
                break
        if not reached_readout:
            raise ValueError("M1 smoke did not show unknown topology-relation messages reaching its readout")


def validate_receipt(path: Path, method: str) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("protocol") != PROTOCOL_ID or receipt.get("source_hashes") != current_sources():
        raise ValueError("Acceptance receipt is stale or belongs to another protocol")
    if type(receipt.get("unit_tests_exit_code")) is not int or receipt["unit_tests_exit_code"] != 0 or method not in receipt.get("passed_methods", []):
        raise ValueError("The method has not passed its implementation gates")
    roots = receipt.get("smoke_roots", {})
    if set(roots) != {"sac_scene_dualgraph_v1", "sac_scene_eventgraph_cv_v1"}:
        raise ValueError("Acceptance receipt must bind both independent smoke roots")
    resolved_roots = [Path(value).resolve() for value in roots.values()]
    if len(set(resolved_roots)) != 2:
        raise ValueError("M0 and M1 smoke roots must be independent")
    geometry = verify_geometry_audit(Path(receipt["geometry_audit_path"]))
    for field in ("sha256", "npz_sha256", "network_sha256", "public_map_fingerprint", "pair_audit_sha256"):
        if receipt.get(f"geometry_{field}") != geometry.get(field):
            raise ValueError(f"Acceptance receipt geometry identity mismatch: {field}")
    for smoke_method, smoke_root in roots.items():
        verify_smoke(Path(smoke_root), smoke_method, geometry)
    verify_deadline(Path(receipt["deadline_contract_root"]))
    return receipt


def verify_deadline(root: Path) -> dict:
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if not (result.get("passed") and result.get("kind") == "deadline_contract"
            and result.get("protocol") == PROTOCOL_ID and result.get("raw_steps") == 600
            and result.get("terminated") and not result.get("truncated") and result.get("timeout")
            and not any(result.get(k) for k in ("success", "collision", "off_route"))
            and result.get("completed") is True and result.get("decision_steps") == 200
            and result.get("training_started") is False
            and result.get("additional_policy_evaluation_episodes") == 0
            and result.get("contract_test_episodes") == 1
            and status.get("state") == "completed" and status.get("kind") == "deadline_contract"
            and status.get("training_started") is False
            and status.get("raw_steps") == 600 and status.get("decision_steps") == 200):
        raise ValueError("Live finite-deadline contract did not pass")
    for key in ("passed", "terminated", "truncated", "timeout", "success", "collision", "off_route",
                "completed", "training_started"):
        if type(result.get(key)) is not bool:
            raise ValueError(f"Deadline outcome field {key} must be boolean")
    if type(status.get("training_started")) is not bool:
        raise ValueError("Deadline status training flag must be boolean")
    _finite_nonnegative_error(result.get("reward_component_error"), "Deadline reward accounting error")
    _finite_nonnegative_error(result.get("max_component_step_error"), "Deadline per-step reward error")
    components = ("reward_success", "reward_collision", "reward_off_route", "reward_timeout",
                  "reward_step_cost", "reward_progress")
    component_sum = 0.0
    for key in components:
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Deadline reward component {key} must be finite")
        component_sum += float(value)
    total = result.get("environment_step_reward_v2_return")
    if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(float(total)):
        raise ValueError("Deadline shaped return must be finite")
    if abs(component_sum - float(total)) > REWARD_ERROR_TOLERANCE:
        raise ValueError("Deadline cumulative six-component reward does not reconcile")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config_sha = _config_digest(config)
    if (config.get("protocol") != PROTOCOL_ID or config.get("scenario") != "intersection_sorted_depart4p0"
            or config.get("raw_steps") != 600 or config.get("eval_episodes") != 0
            or config.get("seed") != 0 or config.get("deadline_seconds") != 60.0
            or config.get("deadline_is_terminal") is not True or config.get("action_repeat") != 3
            or config.get("raw_dt") != 0.1):
        raise ValueError("Deadline contract config does not match the fixed finite-horizon scope")
    if status.get("config_sha256") != config_sha:
        raise ValueError("Deadline status/config identity mismatch")
    trace = result.get("trace")
    if not isinstance(trace, list) or len(trace) != 200:
        raise ValueError("Deadline trace must contain exactly 200 raw-clock decisions")
    prior_tick = None
    for index, row in enumerate(trace, 1):
        remaining = row.get("remaining_seconds")
        if (row.get("raw") != 3 * index
                or isinstance(remaining, bool) or not isinstance(remaining, (int, float))
                or not math.isfinite(float(remaining))
                or abs(float(remaining) - (60.0 - 0.3 * index)) > 2e-5
                or isinstance(row.get("collector_tick"), bool)
                or not isinstance(row.get("collector_tick"), int)):
            raise ValueError("Deadline trace disagrees with raw time or remaining-time observation")
        if prior_tick is not None and row["collector_tick"] - prior_tick != 3:
            raise ValueError("Deadline collector trace does not advance by three physical ticks per decision")
        prior_tick = row["collector_tick"]
    runtime = json.loads((root / "runtime_provenance.json").read_text(encoding="utf-8"))
    if (runtime.get("training_started") is not False or runtime.get("fresh_training") is not False
            or runtime.get("resume") is not False or runtime.get("protocol") != PROTOCOL_ID):
        raise ValueError("Deadline contract artifact incorrectly claims model training")
    _verify_runtime_snapshot(root, runtime, include_deadline_validator=True)
    return result
