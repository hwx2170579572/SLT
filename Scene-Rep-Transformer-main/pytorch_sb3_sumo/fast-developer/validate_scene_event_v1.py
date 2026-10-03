"""Verify finite implementation evidence; never launch training or simulation."""
import argparse
from pathlib import Path
import subprocess
import sys
import uuid

from scene_event import PROTOCOL_ID
from scene_event.acceptance import current_sources, verify_deadline, verify_geometry_audit, verify_smoke
from scene_event.provenance import save_json, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-smoke", type=Path, required=True)
    parser.add_argument("--m1-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-contract", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parent
    tests = sorted((root.parent / "tests_sb3_sumo").glob("test_scene_event_*.py"))
    tests += sorted((root / "tests").glob("test_scene_event_*.py"))
    tests += sorted(root.glob("test_scene_event_*.py"))
    if not tests:
        raise RuntimeError("No implementation tests found")
    # The host's default pytest temp directory may be owned by another session.
    # Use a fresh workspace-local location; never reuse/delete a user run root.
    temporary_root = root / "analysis"
    basetemp = temporary_root / ("sv_" + uuid.uuid4().hex[:8])
    while basetemp.exists():
        basetemp = temporary_root / ("sv_" + uuid.uuid4().hex[:8])
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--basetemp", str(basetemp), *map(str, tests)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    methods = {"sac_scene_dualgraph_v1": args.m0_smoke.resolve(),
               "sac_scene_eventgraph_cv_v1": args.m1_smoke.resolve()}
    if result.returncode != 0:
        raise RuntimeError("Unit tests failed; no acceptance receipt was written")
    geometry = verify_geometry_audit(args.geometry_audit)
    smokes = {method: verify_smoke(path, method, geometry) for method, path in methods.items()}
    deadline = verify_deadline(args.deadline_contract.resolve())
    save_json(args.output, {"protocol": PROTOCOL_ID, "created_utc": utc_now(),
        "scope": "Stages 0-4 implementation and bounded smoke; no success-rate claim or Stage 5-7 empirical acceptance",
        "source_hashes": current_sources(), "unit_tests_exit_code": result.returncode,
        "unit_test_command": command, "unit_test_output": result.stdout,
        "passed_methods": list(methods), "smoke_roots": {k: str(v) for k, v in methods.items()},
        "geometry_audit_path": geometry["path"],
        "geometry_sha256": geometry["sha256"],
        "geometry_npz_sha256": geometry["npz_sha256"],
        "geometry_network_sha256": geometry["network_sha256"],
        "geometry_public_map_fingerprint": geometry["public_map_fingerprint"],
        "geometry_pair_audit_sha256": geometry["pair_audit_sha256"],
        "geometry_summary": geometry,
        "deadline_contract_root": str(args.deadline_contract.resolve()), "deadline_summary": deadline,
        "smoke_summaries": smokes})
    print(f"Acceptance receipt written: {args.output}")


if __name__ == "__main__":
    main()
