"""Replay one frozen v4.8 promotion checkpoint with another decoder.

This is a post-hoc closed-loop attribution intervention.  It reuses the
audited v4.6 replay engine because v4.8 retained the same model/deployment
stack.  It validates v4.8 provenance, forbids test traffic, never writes in the
source run, and explicitly excludes its outputs from every scientific gate.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import replay_v4_6_pair_posthoc as parent


EXPECTED_ALGORITHM = "topo_v4_8_tie_only_replicated_calibration"
EXPECTED_CONTRACT = "42f54135e4d105563b0ad2113147bfef7811c80664793419538a18e9d8c08e51"
EXPECTED_FREEZE = "5827ca63caf5cfd187ac5ad0bdc2fd0fbb0fb00c324459007cf24620f0b57aa6"
DEVELOPMENT_DECISION = ROOT / "results_topo_v4_8_dev" / "development" / "development_decision.json"
ATTRIBUTION_ROOT = ROOT / "results_topo_v4_8_promotion" / "attribution" / "closed_loop"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def validate_source(args: Any) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    arguments_path = run_dir / "arguments.json"
    diagnostics_path = run_dir / "return_estimator_diagnostics.json"
    selector_path = run_dir / "selector" / "receipt.json"
    arguments = _load(arguments_path)
    diagnostics = _load(diagnostics_path)
    selector = _load(selector_path)
    requested = arguments.get("requested_raw_steps", {})
    if requested.get("algo") != EXPECTED_ALGORITHM:
        raise ValueError("source is not the frozen v4.8 candidate")
    if arguments.get("experiment_contract_sha256") != EXPECTED_CONTRACT:
        raise ValueError("source contract hash drifted")
    if arguments.get("implementation_freeze_sha256") != EXPECTED_FREEZE:
        raise ValueError("source scientific freeze drifted")
    if args.experiment_contract_sha256.lower() != EXPECTED_CONTRACT:
        raise ValueError("CLI contract hash drifted")
    if args.implementation_freeze_sha256.lower() != EXPECTED_FREEZE:
        raise ValueError("CLI scientific freeze hash drifted")
    expected_decision = _sha256(DEVELOPMENT_DECISION)
    if args.development_decision_sha256.lower() != expected_decision:
        raise ValueError("CLI development decision hash drifted")
    if int(diagnostics.get("n_step", -1)) != 16:
        raise ValueError("source did not use frozen 16-step returns")
    if diagnostics.get("bootstrap_discount") != "gamma_power_actual_horizon":
        raise ValueError("source bootstrap discount drifted")
    if requested.get("evaluation_split") != "validation":
        raise ValueError("source did not use validation")
    if args.traffic_partition != "validation":
        raise ValueError("v4.8 promotion attribution permits validation only")
    if int(args.evaluation_seed_start) != int(requested["evaluation_seed_start"]):
        raise ValueError("post-hoc seed block must equal the source validation block")
    if int(args.episodes) != int(requested["eval_episodes"]):
        raise ValueError("post-hoc episode count must equal the source validation count")
    model_path = (run_dir / args.model_file).resolve()
    if _sha256(model_path) != args.checkpoint_sha256.lower():
        raise ValueError("checkpoint hash drifted")
    selected_source = Path(selector["selected_source_checkpoint_path"]).resolve()
    if model_path != selected_source:
        raise ValueError("post-hoc model is not the selected source checkpoint")
    if args.checkpoint_kind != selector["selected_checkpoint_kind"]:
        raise ValueError("checkpoint kind differs from the selected source")
    try:
        output_dir.relative_to(ATTRIBUTION_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("output must stay under the v4.8 attribution root") from exc
    try:
        output_dir.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise ValueError("post-hoc output cannot be inside the source run")
    return {
        "arguments": arguments,
        "diagnostics_path": diagnostics_path,
        "selector_path": selector_path,
        "development_decision_sha256": expected_decision,
    }


def main(argv: list[str] | None = None) -> int:
    args = parent.parser().parse_args(argv)
    provenance = validate_source(args)
    result = parent.main(argv)
    if result != 0:
        return result
    output_dir = args.output_dir.resolve()
    result_path = output_dir / "attribution_result.json"
    payload = _load(result_path)
    payload.update(
        {
            "schema_version": "topo-scene-v4.8.promotion-pair-posthoc-replay/v1",
            "scientific_version": "v4.8_tie_only_replicated_calibration",
            "post_hoc_diagnostic_only": True,
            "counted_for_development_gate": False,
            "counted_for_promotion_gate": False,
            "counted_for_formal_gate": False,
            "formal_test_accessed": False,
            "source_algorithm_verified": EXPECTED_ALGORITHM,
            "source_return_n_step_verified": 16,
            "source_bootstrap_discount_verified": "gamma_power_actual_horizon",
            "source_return_estimator_diagnostics": str(
                provenance["diagnostics_path"].resolve()
            ),
            "source_return_estimator_diagnostics_sha256": _sha256(
                provenance["diagnostics_path"]
            ),
            "source_selector_receipt": str(provenance["selector_path"].resolve()),
            "source_selector_receipt_sha256": _sha256(provenance["selector_path"]),
            "reused_compatible_replay_stack": "tools/replay_v4_6_pair_posthoc.py",
            "promotion_result_reinterpretation_forbidden": True,
        }
    )
    _write(result_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "result_sha256": _sha256(result_path),
                "decoder": args.decoder,
                "outcomes": payload["outcomes"],
                "post_hoc_diagnostic_only": True,
                "formal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
