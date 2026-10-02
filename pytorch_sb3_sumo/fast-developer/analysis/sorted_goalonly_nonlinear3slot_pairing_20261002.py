"""Read-only paired outcome audit for the sorted nonlinear 3-slot run.

This script only reads existing evaluation JSON and the 30 effective sorted
route templates. It never imports the simulator or model code.
Run from any directory with the project Python environment:
  python sorted_goalonly_nonlinear3slot_pairing_20261002.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


WS = Path(r"D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1")
FD = WS / "Scene-Rep-Transformer-main" / "pytorch_sb3_sumo" / "fast-developer"
RUNS = WS / "runs"
AUDIT = FD / "analysis" / "sorted_module_outcomes_20261002.json"
OUT = FD / "analysis" / "sorted_goalonly_nonlinear3slot_pairing_20261002.json"
TRAFFIC_DIR_NAME = "_p4_lowdensity_s4p0__intersection_sorted"
EXPECTED_SEEDS = set(range(10000, 10100))
LABELS = ("success", "collision", "timeout", "off_route")
SHORT = {"success": "S", "collision": "C", "timeout": "T", "off_route": "O"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def outcome(row: dict[str, Any]) -> str:
    active = [name for name in LABELS if row.get(name) is True]
    if len(active) != 1:
        return "invalid:" + ("+".join(active) if active else "unknown")
    return active[0]


def template_hashes(eval_path: Path) -> dict[str, str] | None:
    run_root = eval_path.parent.parent
    route_dir = run_root / TRAFFIC_DIR_NAME
    if not route_dir.is_dir():
        return None
    hashes: dict[str, str] = {}
    for index in range(30):
        name = f"traffic_{index:02d}.rou.xml"
        path = route_dir / name
        if not path.is_file():
            return None
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[name] = digest
    return hashes


def load_method(label: str, eval_path: Path) -> dict[str, Any]:
    data = read_json(eval_path)
    identity = data.get("identity", {})
    records = data.get("episode_records", [])
    keyed: dict[tuple[int, str], dict[str, Any]] = {}
    duplicates: list[tuple[int, str]] = []
    invalid_rows = 0
    outcomes: dict[str, int] = {name: 0 for name in LABELS}
    for row in records:
        key = (int(row["seed"]), str(row["traffic_variant"]))
        if key in keyed:
            duplicates.append(key)
        keyed[key] = row
        label_name = outcome(row)
        if label_name.startswith("invalid:"):
            invalid_rows += 1
        else:
            outcomes[label_name] += 1
    seed_set = {key[0] for key in keyed}
    variant_set = {key[1] for key in keyed}
    return {
        "label": label,
        "path": str(eval_path),
        "identity": {
            "method": identity.get("method"),
            "scenario": identity.get("scenario"),
            "depart_scale": identity.get("depart_scale"),
            "eval_traffic_split": identity.get("eval_traffic_split"),
            "episodes": identity.get("episodes"),
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
        },
        "rows": len(records),
        "unique_seed_count": len(seed_set),
        "seed_min": min(seed_set) if seed_set else None,
        "seed_max": max(seed_set) if seed_set else None,
        "seed_set_matches_expected": seed_set == EXPECTED_SEEDS,
        "traffic_variant_count": len(variant_set),
        "traffic_variants": sorted(variant_set),
        "duplicate_keys": [[seed, traffic] for seed, traffic in duplicates],
        "invalid_terminal_rows": invalid_rows,
        "outcomes": outcomes,
        "keyed": keyed,
        "template_hashes": template_hashes(eval_path),
    }


def pairing(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref_keys = set(reference["keyed"])
    cand_keys = set(candidate["keyed"])
    shared = sorted(ref_keys & cand_keys)
    matrix = {SHORT[row]: {SHORT[col]: 0 for col in LABELS} for row in LABELS}
    seed_mismatches: list[int] = []
    invalid = 0
    for seed, traffic in shared:
        ref = reference["keyed"][(seed, traffic)]
        cand = candidate["keyed"][(seed, traffic)]
        ro, co = outcome(ref), outcome(cand)
        if ro.startswith("invalid:") or co.startswith("invalid:"):
            invalid += 1
            continue
        matrix[SHORT[ro]][SHORT[co]] += 1
    # Verify the entire logical-seed -> template mapping, including any keys
    # missing from the intersection, instead of pairing by episode index.
    ref_map = {seed: traffic for seed, traffic in ref_keys}
    cand_map = {seed: traffic for seed, traffic in cand_keys}
    common_seeds = sorted(set(ref_map) & set(cand_map))
    for seed in common_seeds:
        if ref_map[seed] != cand_map[seed]:
            seed_mismatches.append(seed)
    ref_hashes = reference["template_hashes"]
    cand_hashes = candidate["template_hashes"]
    template_hash_match = (
        ref_hashes is not None
        and cand_hashes is not None
        and ref_hashes == cand_hashes
    )
    row_totals = {row: sum(cols.values()) for row, cols in matrix.items()}
    col_totals = {
        col: sum(matrix[row][col] for row in matrix) for col in matrix
    }
    return {
        "reference": reference["label"],
        "candidate": candidate["label"],
        "pair_key": "(eval seed, traffic_variant)",
        "matched_pairs": len(shared),
        "reference_only_keys": len(ref_keys - cand_keys),
        "candidate_only_keys": len(cand_keys - ref_keys),
        "seed_template_mapping_mismatches": seed_mismatches,
        "template_hashes_match_for_all_30": template_hash_match,
        "template_hash_count_reference": len(ref_hashes or {}),
        "template_hash_count_candidate": len(cand_hashes or {}),
        "invalid_terminal_pairs": invalid,
        "matrix_reference_rows_candidate_columns": matrix,
        "reference_marginals": row_totals,
        "candidate_marginals": col_totals,
    }


def main() -> None:
    old = read_json(AUDIT)
    paths: dict[str, Path] = {}
    for key in ("strt", "strt_3slot", "topo", "routeaware"):
        method = old["methods"][key]
        paths[key] = Path(method["source_paths"]["evaluation_results"])
    paths["nonlinear_3slot"] = (
        RUNS
        / "sortg3_1002"
        / "sac_mlp_d1_st_rt_3slot_nonlinear_v1__intersection_sorted_depart4p0"
        / "evaluation_results.json"
    )
    paths["goalonly"] = (
        RUNS
        / "sortg3_1002"
        / "sac_mlp_d1_st_rt_topo_goalonly_v1__intersection_sorted_depart4p0"
        / "evaluation_results.json"
    )

    loaded: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        if path.is_file():
            loaded[label] = load_method(label, path)

    pair_specs = [
        ("strt_3slot", "nonlinear_3slot"),
        ("strt", "nonlinear_3slot"),
        ("strt", "goalonly"),
        ("topo", "goalonly"),
        ("routeaware", "goalonly"),
    ]
    pairs = []
    for reference, candidate in pair_specs:
        if reference in loaded and candidate in loaded:
            pairs.append(pairing(loaded[reference], loaded[candidate]))
        else:
            pairs.append({
                "reference": reference,
                "candidate": candidate,
                "status": "candidate_or_reference_evaluation_pending",
                "expected_candidate_path": str(paths[candidate]),
            })

    methods_out = {
        label: {key: value for key, value in item.items() if key != "keyed"}
        for label, item in loaded.items()
    }
    out = {
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "scope": "offline paired outcome audit; no simulator/model execution",
        "source_audit": str(AUDIT),
        "expected_eval_seeds": [10000, 10099],
        "pairing_key": ["seed", "traffic_variant"],
        "methods": methods_out,
        "pairs": pairs,
        "interpretation_limits": [
            "Paired counts describe fixed checkpoints on the evaluated traffic templates, not training-seed variability.",
            "A 100-episode evaluation reuses 30 sorted traffic templates.",
            "Old and new return protocols are not merged; this artifact compares terminal outcomes only.",
            "A missing evaluation is marked pending and is never inferred from a progress or partial diagnostic summary.",
        ],
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE {OUT}")
    for pair in pairs:
        if "matrix_reference_rows_candidate_columns" in pair:
            print(json.dumps({k: pair[k] for k in (
                "reference", "candidate", "matched_pairs",
                "seed_template_mapping_mismatches",
                "template_hashes_match_for_all_30",
                "matrix_reference_rows_candidate_columns",
            )}, ensure_ascii=False))
        else:
            print(json.dumps(pair, ensure_ascii=False))


if __name__ == "__main__":
    main()
