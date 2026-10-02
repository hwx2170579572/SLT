"""Plan, preflight, run, resume, and summarize the single-seed 100-episode v2 study."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import run_high_density_same_scene_v1 as shared_runner  # noqa: E402


DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "high_density_single_seed_100ep_v2"
    / "protocol.json"
)
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results_hd_ss100_v2"
TRAIN_SCRIPT = PROJECT_ROOT / "tools" / "train_high_density_single_seed_100ep_v2.py"
_SUBCOMMANDS = {"plan", "preflight", "run", "status", "summarize"}


def _with_v2_defaults(argv: list[str]) -> list[str]:
    output = list(argv)
    if output and output[0] in _SUBCOMMANDS:
        if "--protocol" not in output:
            output.extend(("--protocol", str(DEFAULT_PROTOCOL_PATH)))
        if "--result-root" not in output:
            output.extend(("--result-root", str(DEFAULT_RESULT_ROOT)))
    return output


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else list(argv)
    shared_runner.TRAIN_SCRIPT = TRAIN_SCRIPT
    return shared_runner.main(_with_v2_defaults(values))


if __name__ == "__main__":
    raise SystemExit(main())
