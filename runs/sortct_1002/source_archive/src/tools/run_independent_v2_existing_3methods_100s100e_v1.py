"""Plan, audit, run, resume, and summarize the 3x3x100x100 evaluation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (  # noqa: E402
    run_independent_v2_existing_models_100s100e_v1 as shared,
)


DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "independent_v2_existing_3methods_100seeds_100episodes_v1"
    / "protocol.json"
)
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results_iv2_eval_3m_100s100e_v1"
EVALUATOR_SCRIPT = (
    PROJECT_ROOT
    / "tools"
    / "evaluate_independent_v2_existing_3methods_100s100e_v1.py"
)


def _with_three_method_defaults(argv: list[str]) -> list[str]:
    values = list(argv)
    if "--protocol" not in values:
        values.extend(("--protocol", str(DEFAULT_PROTOCOL_PATH)))
    if "--result-root" not in values:
        values.extend(("--result-root", str(DEFAULT_RESULT_ROOT)))
    return values


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else list(argv)
    shared.RUNNER_SCRIPT = Path(__file__).resolve()
    shared.EVALUATOR_SCRIPT = EVALUATOR_SCRIPT
    return shared.main(_with_three_method_defaults(values))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_PROTOCOL_PATH", "DEFAULT_RESULT_ROOT", "main"]
