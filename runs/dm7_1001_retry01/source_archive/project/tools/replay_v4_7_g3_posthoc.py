"""Replay a frozen v4.7 selected pair for G3 failure attribution only.

v4.7 changed the training-time return estimator but retained the v4.6 model,
environment, and deployment-decoder stack.  This wrapper deliberately reuses
the audited v4.6 closed-loop replay implementation, then strengthens the
result with v4.7 source-run and return-estimator checks.  Outputs are excluded
from development, promotion, and formal-test gates.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import replay_v4_6_pair_posthoc as parent


EXPECTED_ALGORITHM = "topo_v4_7_horizon_correct_credit"
EXPECTED_N_STEP = 16
EXPECTED_BOOTSTRAP = "gamma_power_actual_horizon"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parent.parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    arguments_path = run_dir / "arguments.json"
    diagnostics_path = run_dir / "return_estimator_diagnostics.json"
    source_arguments = _load(arguments_path)
    requested = source_arguments.get("requested_raw_steps", {})
    diagnostics = _load(diagnostics_path)

    if requested.get("algo") != EXPECTED_ALGORITHM:
        raise ValueError("post-hoc source is not the frozen v4.7 candidate")
    if int(diagnostics.get("n_step", -1)) != EXPECTED_N_STEP:
        raise ValueError("post-hoc source did not use the frozen 16-step return")
    if diagnostics.get("bootstrap_discount") != EXPECTED_BOOTSTRAP:
        raise ValueError("post-hoc source did not use horizon-correct bootstrap")
    if diagnostics.get("diagnostics_computed_from_runtime_buffer") is not True:
        raise ValueError("return diagnostics were not computed from runtime replay")

    result = parent.main(argv)
    if result != 0:
        return result

    output_dir = args.output_dir.resolve()
    result_path = output_dir / "attribution_result.json"
    payload = _load(result_path)
    payload.update(
        {
            "schema_version": "topo-scene-v4.7.g3-failure-pair-posthoc-replay/v1",
            "scientific_version": "v4.7_horizon_correct_16_step_terminal_credit",
            "failed_development_cell": "G3",
            "source_algorithm_verified": EXPECTED_ALGORITHM,
            "source_return_n_step_verified": EXPECTED_N_STEP,
            "source_bootstrap_discount_verified": EXPECTED_BOOTSTRAP,
            "source_return_estimator_diagnostics": str(diagnostics_path),
            "source_return_estimator_diagnostics_sha256": _sha256(diagnostics_path),
            "reused_compatible_replay_stack": "tools/replay_v4_6_pair_posthoc.py",
            "cross_block_result_is_gate_input": False,
        }
    )
    _write(result_path, payload)
    print(
        json.dumps(
            {
                "v4_7_posthoc_output": str(output_dir),
                "result_sha256": _sha256(result_path),
                "outcomes": payload["outcomes"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
