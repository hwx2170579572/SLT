"""Summarize frozen/source-all paper evaluations by seed and mean ± std."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.paper_evaluation_contract import saved_evaluation_contract_errors


PROTOCOLS = ("frozen_80_20", "source_all")
METRICS = (
    "success_rate",
    "collision_rate",
    "off_route_rate",
    "timeout_rate",
    "mean_return",
    "mean_raw_steps",
)


def _write_csv_contents(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write a CSV without failing the whole report when a viewer locks it.

    Windows spreadsheet and preview applications can keep an existing CSV open
    with an exclusive sharing mode.  Preserve that file and publish the current
    data beside it; the returned path is recorded in ``summary.json``.
    """

    try:
        _write_csv_contents(path, rows)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}.latest{path.suffix}")
        _write_csv_contents(fallback, rows)
        print(
            f"WARNING: {path} is locked; wrote current data to {fallback}",
            file=sys.stderr,
        )
        return fallback


def _method_from_run_name(run_name: str) -> str:
    parts = run_name.split("__")
    if len(parts) != 4:
        raise ValueError(f"Unexpected formal run name {run_name!r}")
    return parts[1]


def _space_brief(signature: dict[str, Any]) -> str:
    child_spaces = signature.get("spaces")
    if isinstance(child_spaces, dict):
        members = ", ".join(
            f"{key}:{value.get('shape')}:{value.get('dtype')}"
            for key, value in sorted(child_spaces.items())
        )
        return f"{signature.get('type')}[{members}]"
    return (
        f"{signature.get('type')}"
        f"(shape={signature.get('shape')},dtype={signature.get('dtype')})"
    )


def collect(
    result_root: Path,
    protocol: dict[str, Any],
    *,
    profile: str,
    methods: list[str],
    scenarios: list[str],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    expected = [
        (method, scenario, seed)
        for method in methods
        for scenario in scenarios
        if scenario in protocol["supported_methods"][method]["scenarios"]
        for seed in seeds
    ]
    for method, scenario, training_seed in expected:
        run_dir = result_root / f"{profile}__{method}__{scenario}__seed{training_seed}"
        if not run_dir.is_dir():
            for traffic_protocol in PROTOCOLS:
                quality.append(
                    {
                        "run": run_dir.name,
                        "protocol": traffic_protocol,
                        "status": "missing_run",
                    }
                )
            continue
        arguments_path = run_dir / "arguments.json"
        if not arguments_path.is_file():
            for traffic_protocol in PROTOCOLS:
                quality.append(
                    {
                        "run": run_dir.name,
                        "protocol": traffic_protocol,
                        "status": "missing_arguments",
                    }
                )
            continue
        argument_payload = json.loads(arguments_path.read_text(encoding="utf-8"))
        arguments = argument_payload["requested_raw_steps"]
        algorithm = str(arguments["algo"])
        actual_scenario = str(arguments["scenario"])
        actual_seed = int(arguments["seed"])
        if algorithm != method or actual_scenario != scenario or actual_seed != training_seed:
            for traffic_protocol in PROTOCOLS:
                quality.append(
                    {
                        "run": run_dir.name,
                        "protocol": traffic_protocol,
                        "status": "identity_mismatch",
                        "errors": (
                            f"arguments identify {algorithm}/{actual_scenario}/seed{actual_seed}; "
                            f"expected {method}/{scenario}/seed{training_seed}"
                        ),
                    }
                )
            continue
        episodes = int(arguments["eval_episodes"])
        episode_limit_profile = str(
            arguments.get("episode_limit_profile", "source")
        )
        parsed_method = _method_from_run_name(run_dir.name)
        if parsed_method != method:
            raise ValueError(
                f"Run-name method {parsed_method!r} does not match expected {method!r}"
            )
        for traffic_protocol in PROTOCOLS:
            detailed_path = (
                run_dir
                / "protocol_evaluations"
                / traffic_protocol
                / "paper_evaluation_detailed.json"
            )
            if not detailed_path.is_file():
                quality.append(
                    {
                        "run": run_dir.name,
                        "protocol": traffic_protocol,
                        "status": "missing",
                    }
                )
                continue
            detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
            errors = saved_evaluation_contract_errors(
                detailed,
                algorithm=algorithm,
                scenario=scenario,
                traffic_protocol=traffic_protocol,
                episode_limit_profile=episode_limit_profile,
                episodes=episodes,
                evaluation_seed_start=training_seed + 10_000,
            )
            if errors:
                quality.append(
                    {
                        "run": run_dir.name,
                        "protocol": traffic_protocol,
                        "status": "invalid",
                        "errors": "; ".join(errors),
                    }
                )
                continue
            summary = detailed["summary"]
            provenance = detailed["evaluation_provenance"]
            row = {
                "profile": "paper",
                "run": run_dir.name,
                "method": method,
                "algorithm": algorithm,
                "scenario": scenario,
                "seed": training_seed,
                "traffic_protocol": traffic_protocol,
                "episodes": episodes,
                "successful_episodes": int(detailed["successful_episodes"]),
                "mean_success_completion_time_seconds": detailed.get(
                    "mean_success_completion_time_seconds"
                ),
                "std_success_completion_time_seconds": detailed.get(
                    "std_success_completion_time_seconds"
                ),
                "environment_class": provenance["environment_class"],
                "traffic_partition": provenance["traffic_partition"],
                "traffic_variants_observed": len(
                    provenance["traffic_variants_observed"]
                ),
                "observation_space": _space_brief(
                    provenance["environment_observation_space"]
                ),
                "action_space": _space_brief(
                    provenance["environment_action_space"]
                ),
                "space_contract_match": provenance[
                    "model_environment_spaces_match"
                ],
                **{metric: summary[metric] for metric in METRICS},
            }
            runs.append(row)
            quality.append(
                {
                    "run": run_dir.name,
                    "protocol": traffic_protocol,
                    "status": "valid",
                    "environment_class": provenance["environment_class"],
                    "space_contract_match": provenance[
                        "model_environment_spaces_match"
                    ],
                    "traffic_variants_observed": len(
                        provenance["traffic_variants_observed"]
                    ),
                }
            )
    return runs, quality


def aggregate(
    runs: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    groups = sorted(
        {
            (row["traffic_protocol"], row["method"], row["scenario"])
            for row in runs
        }
    )
    output: list[dict[str, Any]] = []
    for traffic_protocol, method, scenario in groups:
        members = [
            row
            for row in runs
            if row["traffic_protocol"] == traffic_protocol
            and row["method"] == method
            and row["scenario"] == scenario
        ]
        item: dict[str, Any] = {
            "traffic_protocol": traffic_protocol,
            "method": method,
            "scenario": scenario,
            "seeds_completed": len(members),
            "seeds": ",".join(str(row["seed"]) for row in sorted(members, key=lambda row: row["seed"])),
            "episodes_per_seed": int(members[0]["episodes"]),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in members]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        reference = protocol.get("paper_reference_percent", {}).get(
            scenario, {}
        ).get(protocol["supported_methods"][method]["paper_label"])
        if reference is not None:
            item["paper_success_rate"] = float(reference[0]) / 100.0
            item["success_rate_delta_from_paper"] = (
                item["success_rate_mean"] - item["paper_success_rate"]
            )
        output.append(item)
    return output


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    quality: list[dict[str, Any]],
) -> None:
    valid = sum(row["status"] == "valid" for row in quality)
    invalid = sum(row["status"] == "invalid" for row in quality)
    missing = sum(row["status"] in {"missing", "missing_run"} for row in quality)
    lines = [
        "# Frozen/source-all evaluation summary",
        "",
        f"Protocol evaluations: {valid} valid, {invalid} invalid, {missing} missing.",
        "",
        "## Aggregate by scenario and algorithm",
        "",
        "| Protocol | Method | Scenario | Seeds | Success mean ± std | Collision mean ± std | Timeout mean ± std | Paper success | Delta |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        paper = row.get("paper_success_rate")
        delta = row.get("success_rate_delta_from_paper")
        lines.append(
            "| {traffic_protocol} | {method} | {scenario} | {seeds_completed} "
            "({seeds}) | {success_rate_mean:.3f} ± {success_rate_std:.3f} | "
            "{collision_rate_mean:.3f} ± {collision_rate_std:.3f} | "
            "{timeout_rate_mean:.3f} ± {timeout_rate_std:.3f} | {paper} | "
            "{delta} |".format(
                **row,
                paper="-" if paper is None else f"{paper:.3f}",
                delta="-" if delta is None else f"{delta:+.3f}",
            )
        )
    lines.extend(
        [
            "",
            "## Per-seed evaluations",
            "",
            "| Protocol | Method | Scenario | Seed | Success | Collision | Off-route | Timeout |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        runs,
        key=lambda item: (
            item["traffic_protocol"],
            item["method"],
            item["scenario"],
            item["seed"],
        ),
    ):
        lines.append(
            "| {traffic_protocol} | {method} | {scenario} | {seed} | "
            "{success_rate:.3f} | {collision_rate:.3f} | {off_route_rate:.3f} | "
            "{timeout_rate:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(
    result_root: Path,
    protocol_path: Path,
    output_dir: Path,
    *,
    profile: str,
    methods: list[str],
    scenarios: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    unknown_methods = sorted(set(methods) - set(protocol["supported_methods"]))
    unknown_scenarios = sorted(set(scenarios) - set(protocol["scenario_order"]))
    if unknown_methods or unknown_scenarios:
        raise ValueError(
            f"Unknown methods={unknown_methods} or scenarios={unknown_scenarios}"
        )
    runs, quality = collect(
        result_root.resolve(),
        protocol,
        profile=profile,
        methods=methods,
        scenarios=scenarios,
        seeds=seeds,
    )
    invalid = [
        row
        for row in quality
        if row["status"] in {"invalid", "identity_mismatch", "missing_arguments"}
    ]
    if invalid:
        raise ValueError(
            "Invalid protocol evaluation artifacts: "
            + "; ".join(f"{row['run']}:{row['protocol']}" for row in invalid)
        )
    aggregate_rows = aggregate(runs, protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "result_root": str(result_root.resolve()),
        "expected_matrix": {
            "profile": profile,
            "methods": methods,
            "scenarios": scenarios,
            "seeds": seeds,
            "protocols": list(PROTOCOLS),
        },
        "protocols": list(PROTOCOLS),
        "runs_valid": len(runs),
        "quality": quality,
        "aggregate": aggregate_rows,
        "runs": runs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    csv_outputs = {
        "runs_by_seed": _write_csv(output_dir / "runs_by_seed.csv", runs),
        "aggregate_mean_std": _write_csv(
            output_dir / "aggregate_mean_std.csv", aggregate_rows
        ),
        "quality": _write_csv(output_dir / "quality.csv", quality),
    }
    _write_markdown(output_dir / "summary.md", aggregate_rows, runs, quality)
    payload["output_files"] = {
        key: str(path.resolve()) for key, path in csv_outputs.items()
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--protocol-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--profile", default="paper")
    parser.add_argument("--methods", default="sac,ppo")
    parser.add_argument(
        "--scenarios",
        default="left_turn,cross,roundabout_easy,roundabout_medium,roundabout",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    output_dir = args.output_dir or (args.result_root / "summary_dual_protocol")
    payload = summarize(
        args.result_root,
        args.protocol_path,
        output_dir.resolve(),
        profile=args.profile,
        methods=[item.strip() for item in args.methods.split(",") if item.strip()],
        scenarios=[item.strip() for item in args.scenarios.split(",") if item.strip()],
        seeds=[int(item.strip()) for item in args.seeds.split(",") if item.strip()],
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
