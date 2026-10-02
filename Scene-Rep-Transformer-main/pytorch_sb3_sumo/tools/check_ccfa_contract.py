"""Validate and optionally execute the machine-checkable ``ccfa.yaml`` gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STAGE_STATUSES = {"planned", "in_progress", "passed", "failed", "blocked"}
REQUIRED_STAGE_FIELDS = {
    "id",
    "title",
    "owner",
    "status",
    "depends_on",
    "inputs",
    "outputs",
    "checks",
    "pass_condition",
    "blocker",
    "handoff",
}
SUPPORTED_CHECK_TYPES = {
    "path_exists",
    "sha256_manifest",
    "command",
    "json_assert",
}


class ContractError(ValueError):
    """A structural or executable contract violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "ccfa.yaml must contain a mapping")
    return payload


def validate_contract(contract: dict[str, Any]) -> list[str]:
    _require(
        contract.get("schema_version") == "ccfa.stage-contract/v1",
        "schema_version must be ccfa.stage-contract/v1",
    )
    for field in ("project", "runtime", "invariants", "research", "stages"):
        _require(field in contract, f"Missing top-level field {field!r}")
    _require(isinstance(contract["project"], dict), "project must be a mapping")
    _require(isinstance(contract["runtime"], dict), "runtime must be a mapping")
    _require(isinstance(contract["invariants"], dict), "invariants must be a mapping")
    _require(isinstance(contract["research"], dict), "research must be a mapping")
    stages = contract["stages"]
    _require(isinstance(stages, list) and stages, "stages must be a non-empty list")

    stage_ids: list[str] = []
    dependencies: dict[str, list[str]] = {}
    for stage_index, stage in enumerate(stages):
        prefix = f"stages[{stage_index}]"
        _require(isinstance(stage, dict), f"{prefix} must be a mapping")
        missing = REQUIRED_STAGE_FIELDS - set(stage)
        _require(not missing, f"{prefix} is missing fields {sorted(missing)}")
        stage_id = stage["id"]
        _require(isinstance(stage_id, str) and stage_id, f"{prefix}.id must be text")
        _require(stage_id not in stage_ids, f"Duplicate stage id {stage_id!r}")
        stage_ids.append(stage_id)
        _require(
            stage["status"] in ALLOWED_STAGE_STATUSES,
            f"{stage_id}: invalid status {stage['status']!r}",
        )
        for collection in ("depends_on", "inputs", "outputs", "checks"):
            _require(
                isinstance(stage[collection], list),
                f"{stage_id}.{collection} must be a list",
            )
        _require(stage["checks"], f"{stage_id}.checks cannot be empty")
        dependencies[stage_id] = list(stage["depends_on"])
        check_ids: list[str] = []
        for check_index, check in enumerate(stage["checks"]):
            _require(
                isinstance(check, dict),
                f"{stage_id}.checks[{check_index}] must be a mapping",
            )
            _require(
                isinstance(check.get("id"), str) and check["id"],
                f"{stage_id}.checks[{check_index}] needs an id",
            )
            _require(
                check["id"] not in check_ids,
                f"{stage_id}: duplicate check id {check['id']!r}",
            )
            check_ids.append(check["id"])
            _require(
                check.get("type") in SUPPORTED_CHECK_TYPES,
                f"{stage_id}.{check['id']}: unsupported type {check.get('type')!r}",
            )
        pass_condition = stage["pass_condition"]
        _require(
            isinstance(pass_condition, dict)
            and pass_condition.get("mode") == "all"
            and isinstance(pass_condition.get("check_ids"), list),
            f"{stage_id}.pass_condition must use mode=all and list check_ids",
        )
        _require(
            set(pass_condition["check_ids"]) == set(check_ids),
            f"{stage_id}.pass_condition.check_ids must name every check exactly once",
        )
        _require(isinstance(stage["blocker"], dict), f"{stage_id}.blocker must be a mapping")
        _require(isinstance(stage["handoff"], dict), f"{stage_id}.handoff must be a mapping")

    known = set(stage_ids)
    for stage_id, required in dependencies.items():
        unknown = set(required) - known
        _require(not unknown, f"{stage_id} depends on unknown stages {sorted(unknown)}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in visited:
            return
        _require(stage_id not in visiting, f"Stage dependency cycle reaches {stage_id}")
        visiting.add(stage_id)
        for dependency in dependencies[stage_id]:
            visit(dependency)
        visiting.remove(stage_id)
        visited.add(stage_id)

    for stage_id in stage_ids:
        visit(stage_id)
    return stage_ids


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _read_field(payload: Any, field: str) -> Any:
    value = payload
    for component in field.split("."):
        if isinstance(value, list):
            value = value[int(component)]
        elif isinstance(value, dict):
            value = value[component]
        else:
            raise KeyError(f"Cannot descend through {component!r} in {field!r}")
    return value


def _evaluate_path_exists(check: dict[str, Any], root: Path) -> dict[str, Any]:
    path = _resolve(root, str(check["path"]))
    kind = str(check.get("kind", "any"))
    passed = path.exists()
    if kind == "file":
        passed = path.is_file()
    elif kind == "directory":
        passed = path.is_dir()
    elif kind != "any":
        raise ContractError(f"Unknown path_exists kind {kind!r}")
    return {"passed": passed, "path": str(path), "kind": kind}


def _evaluate_sha256_manifest(check: dict[str, Any], root: Path) -> dict[str, Any]:
    manifest_path = _resolve(root, str(check["path"]))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("protected_files")
    _require(isinstance(rows, list) and rows, "SHA manifest needs protected_files")
    results: list[dict[str, Any]] = []
    for row in rows:
        path = _resolve(root, str(row["path"]))
        actual = _sha256(path) if path.is_file() else None
        expected = str(row["sha256"]).lower()
        results.append(
            {
                "path": str(path),
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
        )
    return {"passed": all(row["passed"] for row in results), "files": results}


def _evaluate_command(check: dict[str, Any], root: Path) -> dict[str, Any]:
    argv = check.get("argv")
    _require(
        isinstance(argv, list) and argv and all(isinstance(value, str) for value in argv),
        f"{check['id']}: command argv must be a non-empty string list",
    )
    cwd = _resolve(root, str(check.get("cwd", ".")))
    timeout_seconds = int(check.get("timeout_seconds", 600))
    completed = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    output = completed.stdout or ""
    required_text = check.get("stdout_contains")
    passed = completed.returncode == int(check.get("expected_exit_code", 0))
    if required_text is not None:
        passed = passed and str(required_text) in output
    return {
        "passed": passed,
        "argv": argv,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "output_tail": output[-4000:],
    }


def _evaluate_json_assert(check: dict[str, Any], root: Path) -> dict[str, Any]:
    path = _resolve(root, str(check["path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = _read_field(payload, str(check["field"]))
    operation_name = str(check.get("operator", "eq"))
    operations: dict[str, Callable[[Any, Any], bool]] = {
        "eq": operator.eq,
        "ne": operator.ne,
        "gt": operator.gt,
        "ge": operator.ge,
        "lt": operator.lt,
        "le": operator.le,
        "in": lambda left, right: left in right,
    }
    if operation_name == "truthy":
        passed = bool(actual)
        expected = True
    else:
        _require(operation_name in operations, f"Unknown JSON operator {operation_name!r}")
        expected = check.get("expected")
        passed = bool(operations[operation_name](actual, expected))
    return {
        "passed": passed,
        "path": str(path),
        "field": check["field"],
        "operator": operation_name,
        "expected": expected,
        "actual": actual,
    }


def evaluate_check(check: dict[str, Any], root: Path) -> dict[str, Any]:
    evaluators = {
        "path_exists": _evaluate_path_exists,
        "sha256_manifest": _evaluate_sha256_manifest,
        "command": _evaluate_command,
        "json_assert": _evaluate_json_assert,
    }
    try:
        details = evaluators[check["type"]](check, root)
    except Exception as exc:  # Keep every failed gate in one machine report.
        details = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"id": check["id"], "type": check["type"], **details}


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=PROJECT_ROOT / "ccfa.yaml")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    contract_path = args.contract.resolve()
    contract = load_contract(contract_path)
    stage_ids = validate_contract(contract)
    selected = set(args.stage or stage_ids)
    unknown = selected - set(stage_ids)
    if unknown:
        raise ContractError(f"Unknown selected stages: {sorted(unknown)}")
    stage_results: list[dict[str, Any]] = []
    if args.evaluate:
        for stage in contract["stages"]:
            if stage["id"] not in selected:
                continue
            checks = [
                evaluate_check(check, PROJECT_ROOT) for check in stage["checks"]
            ]
            stage_results.append(
                {
                    "id": stage["id"],
                    "declared_status": stage["status"],
                    "passed": all(check["passed"] for check in checks),
                    "checks": checks,
                }
            )
    result = {
        "contract": str(contract_path),
        "schema_valid": True,
        "stage_ids": stage_ids,
        "evaluated": bool(args.evaluate),
        "selected_stages": sorted(selected),
        "passed": all(stage["passed"] for stage in stage_results),
        "stage_results": stage_results,
    }
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(json.dumps({"schema_valid": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc
