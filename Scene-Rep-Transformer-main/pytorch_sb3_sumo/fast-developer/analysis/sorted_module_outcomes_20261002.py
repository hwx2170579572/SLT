"""Offline outcome audit for the completed sorted/depart4 module runs.

Reads existing JSON evaluation artifacts and sorted route XML copies only; it
does not load policies, start SUMO, or alter source results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, median, pstdev
from datetime import datetime


WS = Path(r"D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1")
FD = WS / "Scene-Rep-Transformer-main" / "pytorch_sb3_sumo" / "fast-developer"
OUT = FD / "analysis" / "sorted_module_outcomes_20261002.json"
COMPLETION = FD / "analysis" / "sorted_pair_completion_20261002.json"

SPECS = {
    "sac_mlp": WS / "runs/d0929_100k_diag/sac_mlp__intersection_sorted_depart4p0",
    "st": FD / "sac_mlp_d1_st__intersection_sorted_depart4p0",
    "strt": WS / "runs/d0929_100k_diag/sac_mlp_d1_st_rt__intersection_sorted_depart4p0",
    "topo": WS / "runs/t0930_topo3_retry01/sac_mlp_d1_st_rt_topo__intersection_sorted_depart4p0",
    "topo_3slot": WS / "runs/t0930_topo3_retry01/sac_mlp_d1_st_rt_topo_3slot__intersection_sorted_depart4p0",
    "strt_3slot": WS / "runs/sort2_1001/sac_mlp_d1_st_rt_3slot__intersection_sorted_depart4p0",
    "routeaware": WS / "runs/sort2_1001/sac_mlp_d1_st_rt_topo_routeaware_v1__intersection_sorted_depart4p0",
}

ASSET_DIRS = {
    "d0929": WS / "runs/d0929_100k_diag/_p4_lowdensity_s4p0__intersection_sorted",
    "t0930": WS / "runs/t0930_topo3_retry01/_p4_lowdensity_s4p0__intersection_sorted",
    "sort2": WS / "runs/sort2_1001/_p4_lowdensity_s4p0__intersection_sorted",
    "fd_st": FD / "_p4_lowdensity_s4p0__intersection_sorted",
}

PAIR_SPECS = [
    ("sac_mlp", "st"),
    ("sac_mlp", "strt"),
    ("st", "strt"),
    ("strt", "topo"),
    ("strt", "routeaware"),
    ("topo", "routeaware"),
    ("strt", "strt_3slot"),
    ("topo", "topo_3slot"),
    ("strt_3slot", "topo_3slot"),
    ("routeaware", "strt_3slot"),
]

OUTCOMES = ("success", "collision", "timeout", "off_route")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def outcome(record):
    true_flags = [name for name in OUTCOMES if bool(record.get(name))]
    if len(true_flags) == 1:
        return true_flags[0]
    if not true_flags:
        return "unknown"
    return "multi_label:" + "+".join(true_flags)


def numeric_stats(records, key):
    values = [float(r[key]) for r in records if isinstance(r.get(key), (int, float))]
    if not values:
        return {"n": 0, "mean": None, "median": None, "population_sd": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "population_sd": pstdev(values),
    }


def summarize_method(name, run_dir):
    eval_path = run_dir / "evaluation_results.json"
    args_path = run_dir / "arguments.json"
    complete_path = run_dir / "training_complete.json"
    manifest_path = run_dir / "experiment_manifest.json"
    final_path = run_dir / "final_model.zip"
    ev = read_json(eval_path)
    args = read_json(args_path)
    complete = read_json(complete_path)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    identity = ev.get("identity", {})
    summary = ev.get("summary", {})
    records = ev.get("episode_records", [])

    raw_flag_counts = {flag: sum(bool(r.get(flag)) for r in records) for flag in OUTCOMES}
    terminal = {label: sum(outcome(r) == label for r in records) for label in OUTCOMES}
    terminal["unknown"] = sum(outcome(r) == "unknown" for r in records)
    terminal["multi_label"] = sum(outcome(r).startswith("multi_label:") for r in records)
    seeds = [int(r["seed"]) for r in records if "seed" in r]
    templates = sorted({str(r.get("traffic_variant", "")) for r in records})
    seed_to_template = {int(r["seed"]): str(r.get("traffic_variant", "")) for r in records if "seed" in r}
    checkpoint_sha_actual = sha256_file(final_path)
    checkpoint_sha_complete = str(complete.get("checkpoint_sha256", "")).lower()
    checkpoint_sha_eval = str(identity.get("checkpoint_sha256", "")).lower()

    complete_eval_metrics = {
        k: summary.get(k)
        for k in (
            "episodes", "success_rate", "collision_rate", "timeout_rate", "off_route_rate",
            "mean_return", "std_return", "raw_mean_return", "raw_std_return",
            "raw_return_coverage", "reward_component_means", "reward_component_coverage",
            "reward_component_reconciliation_max_abs_error", "evaluation_return_protocol_version",
            "raw_return_protocol_version", "reward_component_protocol_version",
        )
        if k in summary
    }

    config_keys = (
        "raw_budget", "seed", "device", "learning_starts_raw_steps", "batch_size",
        "learning_rate", "buffer_size", "discount", "action_repeat", "env_contract",
        "reward_shaping", "use_route", "use_topology", "use_slots", "use_incremental_slots",
        "use_route_reachability", "use_graph_slt", "use_sbs", "representation_coef", "slot_balance_coef",
    )
    selected_config = {k: args.get(k) for k in config_keys if k in args}

    diagnostics = {}
    if name in {"strt_3slot", "routeaware"}:
        for phase in ("train", "eval"):
            diag_path = run_dir / "diagnostics" / phase / "summary.json"
            if not diag_path.exists():
                diagnostics[phase] = {"summary_path": str(diag_path), "present": False}
                continue
            ds = read_json(diag_path)
            diagnostics[phase] = {
                "summary_path": str(diag_path),
                "present": True,
                "diagnostic_error_count": ds.get("diagnostic_error_count"),
                "episodes_finished": ds.get("episodes_finished"),
                "raw_records": ds.get("raw_records"),
                "decision_records": ds.get("decision_records"),
                "optimization_samples": ds.get("optimization_samples"),
                "representation_samples": ds.get("representation_samples"),
                "representation_samples_by_source": ds.get("representation_samples_by_source"),
                "policy_observation_snapshots": ds.get("policy_observation_snapshots"),
            }

    returns = numeric_stats(records, "episode_return")
    by_outcome = {}
    for label in (*OUTCOMES, "unknown", "multi_label"):
        subset = [r for r in records if (outcome(r) == label if label != "multi_label" else outcome(r).startswith("multi_label:"))]
        by_outcome[label] = {
            "n": len(subset),
            "decision_steps": numeric_stats(subset, "decision_steps"),
            "raw_steps": numeric_stats(subset, "raw_steps"),
            "completion_time_seconds": numeric_stats(subset, "completion_time_seconds"),
            "episode_return": numeric_stats(subset, "episode_return"),
        }

    return {
        "method": name,
        "run_dir": str(run_dir),
        "source_paths": {
            "arguments": str(args_path),
            "training_complete": str(complete_path),
            "experiment_manifest": str(manifest_path) if manifest_path.exists() else None,
            "evaluation_results": str(eval_path),
            "final_model": str(final_path),
        },
        "protocol": {
            "scenario": identity.get("scenario", args.get("scenario")),
            "depart_scale": identity.get("depart_scale", manifest.get("depart_scale")),
            "eval_traffic_split": identity.get("eval_traffic_split", manifest.get("eval_traffic_split")),
            "evaluation_episodes_identity": identity.get("episodes"),
            "evaluation_episodes_records": len(records),
            "evaluation_workers": identity.get("workers"),
            "smoke": identity.get("smoke", args.get("smoke")),
            "behavior_diagnostics": identity.get("behavior_diagnostics", args.get("behavior_diagnostics")),
            "training_config": selected_config,
            "evaluation_return_protocol_version": summary.get("evaluation_return_protocol_version", "legacy_schema_no_explicit_version"),
        },
        "training": {
            "raw_steps": complete.get("raw_steps"),
            "updates": complete.get("updates"),
            "replay_size": complete.get("replay_size"),
            "wall_seconds": complete.get("wall_seconds"),
            "seed": args.get("seed"),
            "device": args.get("device"),
        },
        "checkpoint": {
            "sha256_actual_final_model": checkpoint_sha_actual,
            "sha256_training_complete": checkpoint_sha_complete,
            "sha256_eval_identity": checkpoint_sha_eval,
            "all_match": checkpoint_sha_actual == checkpoint_sha_complete == checkpoint_sha_eval,
        },
        "evaluation": {
            "summary": complete_eval_metrics,
            "raw_flag_counts": raw_flag_counts,
            "exclusive_terminal_counts": terminal,
            "seed_range": [min(seeds), max(seeds)] if seeds else None,
            "unique_seed_count": len(set(seeds)),
            "traffic_templates": templates,
            "unique_traffic_template_count": len(templates),
            "outcome_conditioned_metrics": by_outcome,
            "record_return_stats": returns,
            "seed_to_traffic_template": {str(k): v for k, v in sorted(seed_to_template.items())},
        },
        "diagnostics": diagnostics,
    }, records


def asset_bundle(path: Path):
    files = sorted(path.glob("traffic_*.rou.xml"))
    mapped = {p.name: sha256_file(p) for p in files}
    content = "\n".join(f"{name}:{digest}" for name, digest in mapped.items()).encode()
    return {"path": str(path), "file_count": len(files), "hashes": mapped, "bundle_sha256": hashlib.sha256(content).hexdigest()}


def make_pair(reference_name, candidate_name, method_records):
    left = {int(r["seed"]): r for r in method_records[reference_name] if "seed" in r}
    right = {int(r["seed"]): r for r in method_records[candidate_name] if "seed" in r}
    common_seeds = sorted(set(left) & set(right))
    matrix = {a: {b: 0 for b in OUTCOMES} for a in OUTCOMES}
    same_template = 0
    template_mismatch_seeds = []
    category_mismatch = []
    for seed in common_seeds:
        l, r = left[seed], right[seed]
        if l.get("traffic_variant") == r.get("traffic_variant"):
            same_template += 1
        else:
            template_mismatch_seeds.append(seed)
        lo, ro = outcome(l), outcome(r)
        if lo in OUTCOMES and ro in OUTCOMES:
            matrix[lo][ro] += 1
        else:
            category_mismatch.append({"seed": seed, "reference": lo, "candidate": ro})
    return {
        "direction": f"{reference_name}_rows_to_{candidate_name}_columns",
        "reference_method": reference_name,
        "candidate_method": candidate_name,
        "matched_seed_count": len(common_seeds),
        "missing_reference_seeds": sorted(set(right) - set(left)),
        "missing_candidate_seeds": sorted(set(left) - set(right)),
        "traffic_template_match_count_on_common_seeds": same_template,
        "traffic_template_mismatch_seeds": template_mismatch_seeds,
        "outcome_category_exceptions": category_mismatch,
        "transition_matrix": matrix,
        "matrix_total": sum(sum(row.values()) for row in matrix.values()),
        "reference_wins_success_vs_non_success": sum(
            1 for s in common_seeds if outcome(left[s]) == "success" and outcome(right[s]) != "success"
        ),
        "candidate_wins_success_vs_non_success": sum(
            1 for s in common_seeds if outcome(right[s]) == "success" and outcome(left[s]) != "success"
        ),
        "success_tie": sum(
            1 for s in common_seeds if (outcome(left[s]) == "success") == (outcome(right[s]) == "success")
        ),
    }


def main():
    methods, record_map = {}, {}
    for name, run_dir in SPECS.items():
        methods[name], record_map[name] = summarize_method(name, run_dir)

    assets = {name: asset_bundle(path) for name, path in ASSET_DIRS.items()}
    baseline_hashes = assets["d0929"]["hashes"]
    asset_checks = {}
    for name, item in assets.items():
        mismatches = sorted(
            filename for filename in set(baseline_hashes) | set(item["hashes"])
            if baseline_hashes.get(filename) != item["hashes"].get(filename)
        )
        asset_checks[name] = {
            "same_template_names_and_content_as_d0929": not mismatches and len(baseline_hashes) == 30,
            "mismatched_template_names": mismatches,
            "bundle_sha256": item["bundle_sha256"],
        }

    pairs = [make_pair(a, b, record_map) for a, b in PAIR_SPECS]
    eval_protocols = {v["protocol"]["evaluation_return_protocol_version"] for v in methods.values()}
    shared_keys = (
        "raw_budget", "seed", "device", "learning_starts_raw_steps", "batch_size",
        "learning_rate", "buffer_size", "discount", "action_repeat", "env_contract", "reward_shaping",
    )
    shared_training_audit = {}
    for key in shared_keys:
        values = {name: methods[name]["protocol"]["training_config"].get(key) for name in methods}
        fingerprints = {json.dumps(value, sort_keys=True) for value in values.values()}
        shared_training_audit[key] = {"same_across_all_methods": len(fingerprints) == 1, "values_by_method": values}

    legacy_terminal_mean_matches = {}
    for name, item in methods.items():
        summary = item["evaluation"]["summary"]
        counts = item["evaluation"]["exclusive_terminal_counts"]
        terminal_score = (counts["success"] - counts["collision"]) / item["protocol"]["evaluation_episodes_records"]
        compared_value = summary.get("raw_mean_return")
        if compared_value is None:
            compared_value = summary.get("mean_return")
        legacy_terminal_mean_matches[name] = {
            "terminal_success_minus_collision_per_episode": terminal_score,
            "reported_raw_mean_or_legacy_mean_return": compared_value,
            "equal_within_1e-9": isinstance(compared_value, (int, float)) and abs(float(compared_value) - terminal_score) <= 1e-9,
            "explicit_protocol_version": summary.get("evaluation_return_protocol_version"),
        }
    output = {
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "offline audit of existing final evaluations only; no model load, simulation, training, or re-evaluation",
        "primary_scene": {
            "scenario": "intersection_sorted",
            "depart_scale": 4.0,
            "traffic_template_assets": assets,
            "asset_hash_checks_vs_d0929": asset_checks,
            "episode_pairing_key": ["eval seed", "traffic_variant filename"],
            "evaluation_seed_source": {
                "seed_start": "train_intersection_yield_v2.py:78-86 defines sac_mlp evaluation start=10000; D1 evaluator sets episode_seed=seed_start+index (train_intersection_yield_v2_d1.py:1099-1104) and passes it to evaluate_model_detailed (same file:1119-1125); evaluation_model_detailed resets env with seed+episode (algos/sb3_torch/evaluation.py:337-340).",
                "stored_records": "All seven evaluations have unique stored seeds 10000..10099; each also stores traffic_variant. Each pair has 100/100 same-seed same-filename matches.",
                "exact_sumo_seed_field": "No separate actual_sumo_seed field exists in evaluation_records; env reset receives the stored logical seed."
            },
            "evaluation_schema_versions_present": sorted(eval_protocols),
            "return_comparison_note": "Outcome and paired transitions are compared across all listed runs. Historical files have no explicit evaluation_return_protocol_version; their reported mean_return equals (success-count minus collision-count)/100 for these seven summaries, consistent with terminal +1/-1/0 scoring but not sufficient to label the historical field as shaped reward. Current sort2 files explicitly identify environment_step_reward_v2, report shaped mean_return and separate raw_mean_return; do not compare the old mean_return directly with the new shaped mean_return.",
            "terminal_mean_return_crosscheck": legacy_terminal_mean_matches,
        },
        "methods": methods,
        "shared_training_config_audit": shared_training_audit,
        "paired_transitions": pairs,
        "limitations": [
            "Each policy is a single fresh training seed (seed 0); paired evaluation episodes do not estimate training-seed variance.",
            "Evaluation uses 100 logical seeds over 30 reused sorted route templates, so 100 episodes are not 100 independent traffic layouts.",
            "Outcome pairing is by seed plus traffic template filename; exact route-template content hashes are separately checked across the archived 30-template pools.",
            "The old and new evaluation return schemas differ. Outcome transitions remain comparable as stored terminal flags, but reward means should be interpreted within their own protocol.",
            "Current route-aware and current 3slot have different topology/slot configurations; direct pair transitions describe fixed policies and do not isolate general causal effects across training seeds.",
        ],
    }
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if COMPLETION.exists():
        completion_doc = read_json(COMPLETION)
        route = methods["routeaware"]
        completion_doc["pair_status"] = "both_methods_complete"
        completion_doc["status_updated_at_local"] = datetime.now().astimezone().isoformat(timespec="seconds")
        completion_doc["pending_snapshot_superseded"] = True
        completion_doc["pending_snapshot_superseded_by"] = "routeaware_method_completion"
        completion_doc["current_pair_results_source"] = str(OUT)
        completion_doc["routeaware_method_completion"] = {
            "method": route["method"],
            "run_dir": route["run_dir"],
            "training": route["training"],
            "checkpoint": route["checkpoint"],
            "evaluation": {
                "source": route["source_paths"]["evaluation_results"],
                "summary": route["evaluation"]["summary"],
                "exclusive_terminal_counts": route["evaluation"]["exclusive_terminal_counts"],
                "seed_range": route["evaluation"]["seed_range"],
                "unique_seed_count": route["evaluation"]["unique_seed_count"],
                "unique_traffic_template_count": route["evaluation"]["unique_traffic_template_count"],
            },
            "diagnostics": route["diagnostics"],
        }
        completion_doc["shared_eval_pool_verification"] = {
            "all_seven_methods_have_seeds_10000_to_10099": all(
                methods[name]["evaluation"]["seed_range"] == [10000, 10099]
                and methods[name]["evaluation"]["unique_seed_count"] == 100
                for name in methods
            ),
            "all_pair_seed_template_matches_are_100_of_100": all(
                pair["matched_seed_count"] == 100
                and pair["traffic_template_match_count_on_common_seeds"] == 100
                and not pair["traffic_template_mismatch_seeds"]
                for pair in pairs
            ),
            "all_sorted_route_xml_bundles_match_30_of_30": all(
                check["same_template_names_and_content_as_d0929"]
                for check in asset_checks.values()
            ),
            "route_xml_bundle_sha256": assets["d0929"]["bundle_sha256"],
        }
        COMPLETION.write_text(json.dumps(completion_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "method_counts": {k: v["evaluation"]["exclusive_terminal_counts"] for k, v in methods.items()},
        "eval_seed_alignment": {k: v["evaluation"]["unique_seed_count"] for k, v in methods.items()},
        "asset_checks": asset_checks,
        "pair_summaries": [
            {"pair": f"{p['reference_method']}->{p['candidate_method']}", "matched": p["matched_seed_count"], "template_matches": p["traffic_template_match_count_on_common_seeds"], "matrix": p["transition_matrix"]}
            for p in pairs
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
