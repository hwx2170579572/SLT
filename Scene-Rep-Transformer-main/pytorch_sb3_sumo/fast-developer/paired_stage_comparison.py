"""Compare already-completed evaluations; never creates an environment or model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


OUTCOMES = ("success", "collision", "timeout")


def _result_file(path):
    path = Path(path)
    return path / "evaluation_results.json" if path.is_dir() else path


def _read(path):
    path = _result_file(path)
    with path.open(encoding="utf-8-sig") as stream:
        result = json.load(stream)
    rows = result.get("episode_records")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No nonempty episode_records in {path}")
    return path, result, rows


def _method(result):
    identity = result.get("identity")
    return result.get("method") or (identity.get("method") if isinstance(identity, dict) else None)


def _flag(value):
    if value is True or value == 1:
        return True
    if value is False or value == 0:
        return False
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"Invalid or missing outcome flag: {value!r}")


def _outcome(row):
    active = [name for name in OUTCOMES if _flag(row.get(name))]
    if len(active) != 1:
        raise ValueError(f"Expected exactly one terminal outcome, got {active}")
    if row.get("off_route") is not None and _flag(row["off_route"]):
        raise ValueError("off_route episode cannot be silently merged into the three terminal outcomes")
    return active[0]


def _index(rows):
    indexed, errors, duplicate_keys = {}, [], set()
    for index, row in enumerate(rows):
        try:
            if not isinstance(row, dict):
                raise ValueError("Episode record is not an object")
            seed, traffic = row.get("seed"), row.get("traffic_variant")
            if seed is None or traffic is None or str(traffic) == "":
                raise ValueError("Missing seed or traffic_variant")
            seed_int = int(seed)
            if isinstance(seed, bool) or float(seed) != seed_int:
                raise ValueError("Seed must be an integer")
            key = (seed_int, str(traffic))
            outcome = _outcome(row)
            if key in indexed or key in duplicate_keys:
                duplicate_keys.add(key)
                indexed.pop(key, None)
                raise ValueError(f"Duplicate seed/traffic pair {key}")
            indexed[key] = (outcome, row)
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append({"row_index": index, "error": str(exc)})
    return indexed, errors


def _case(key, before, after):
    return {"seed": key[0], "traffic_variant": key[1],
            "reference_outcome": before[0], "candidate_outcome": after[0],
            "reference_raw_steps": before[1].get("raw_steps"),
            "candidate_raw_steps": after[1].get("raw_steps"),
            "reference_episode_return": before[1].get("episode_return"),
            "candidate_episode_return": after[1].get("episode_return")}


def compare_stages(reference_path, candidate_path, output_path, label=None, expected_episodes=None):
    """Save a complete pairing audit and outcome matrix to an explicit JSON file.

    Outcome attribution remains conditional on the experiment manifest proving
    matched budgets/configuration. Matching rows alone does not prove that.
    """
    ref_path, ref_result, ref_rows = _read(reference_path)
    cand_path, cand_result, cand_rows = _read(candidate_path)
    reference, ref_errors = _index(ref_rows)
    candidate, cand_errors = _index(cand_rows)
    common = sorted(reference.keys() & candidate.keys())
    missing_reference = sorted(candidate.keys() - reference.keys())
    missing_candidate = sorted(reference.keys() - candidate.keys())
    expected_count_met = (expected_episodes is None or
                          len(ref_rows) == len(cand_rows) == int(expected_episodes))
    comparable = (expected_count_met and not ref_errors and not cand_errors and not missing_reference
                  and not missing_candidate and len(common) == len(ref_rows) == len(cand_rows))
    matrix = {before: {after: 0 for after in OUTCOMES} for before in OUTCOMES}
    cases, per_traffic = [], {}
    for key in common:
        before, after = reference[key], candidate[key]
        matrix[before[0]][after[0]] += 1
        cases.append(_case(key, before, after))
        traffic = per_traffic.setdefault(key[1], {
            "pairs": 0, "reference": {name: 0 for name in OUTCOMES},
            "candidate": {name: 0 for name in OUTCOMES},
            "collision_to_success": 0, "success_to_collision": 0,
            "new_timeouts": 0})
        traffic["pairs"] += 1
        traffic["reference"][before[0]] += 1
        traffic["candidate"][after[0]] += 1
        traffic["collision_to_success"] += int(before[0] == "collision" and after[0] == "success")
        traffic["success_to_collision"] += int(before[0] == "success" and after[0] == "collision")
        traffic["new_timeouts"] += int(before[0] != "timeout" and after[0] == "timeout")
    ref_counts = {name: sum(matrix[name].values()) for name in OUTCOMES}
    cand_counts = {name: sum(matrix[before][name] for before in OUTCOMES) for name in OUTCOMES}
    success_delta = cand_counts["success"] - ref_counts["success"]
    timeout_delta = cand_counts["timeout"] - ref_counts["timeout"]
    collision_rescues = matrix["collision"]["success"]
    success_regressions = matrix["success"]["collision"]
    promising = comparable and success_delta > 0 and timeout_delta <= 0 and collision_rescues > success_regressions
    report = {
        "schema_version": 1, "label": label,
        "reference_file": str(ref_path.resolve()), "candidate_file": str(cand_path.resolve()),
        "reference_method": _method(ref_result), "candidate_method": _method(cand_result),
        "reference_identity": ref_result.get("identity"), "candidate_identity": cand_result.get("identity"),
        "pairing_key": ["seed", "traffic_variant"],
        "reference_episode_records": len(ref_rows), "candidate_episode_records": len(cand_rows),
        "expected_episode_records": expected_episodes, "expected_episode_count_met": expected_count_met,
        "matched_pairs": len(common), "paired_outcomes_complete": comparable,
        "reference_record_errors": ref_errors, "candidate_record_errors": cand_errors,
        "missing_reference_pairs": [{"seed": key[0], "traffic_variant": key[1]} for key in missing_reference],
        "missing_candidate_pairs": [{"seed": key[0], "traffic_variant": key[1]} for key in missing_candidate],
        "matrix_rows_reference_columns_candidate": matrix,
        "matched_reference_counts": ref_counts, "matched_candidate_counts": cand_counts,
        "matched_success_delta_count": success_delta, "matched_timeout_delta_count": timeout_delta,
        "collision_to_success": collision_rescues, "success_to_collision": success_regressions,
        "new_timeouts_from_non_timeout": matrix["success"]["timeout"] + matrix["collision"]["timeout"],
        "resolved_timeouts": matrix["timeout"]["success"] + matrix["timeout"]["collision"],
        "paired_outcome_criterion_met": promising if comparable else None,
        "decision_hint": ("inspect_pairing_errors_before_interpretation" if not comparable else
                          "candidate_gain_observed_check_module_activation_gradients_and_protocol" if promising else
                          "analyze_current_module_before_adding_another"),
        "per_traffic_variant": per_traffic,
        "changed_cases": [case for case in cases if case["reference_outcome"] != case["candidate_outcome"]],
        "all_paired_cases": cases,
        "limitations": [
            "Only saved evaluation records are compared; no additional simulation is run.",
            "Check the experiment manifests for matched training budget, density, seeds and checkpoint selection.",
            "A single training seed does not establish robustness across training seeds.",
            "Partial-pair counts are descriptive only when paired_outcomes_complete is false.",
        ],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label")
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args()
    result = compare_stages(args.reference, args.candidate, args.output, args.label, args.expected_episodes)
    print(json.dumps({key: result[key] for key in (
        "matched_pairs", "paired_outcomes_complete", "collision_to_success",
        "success_to_collision", "matched_timeout_delta_count", "decision_hint")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
