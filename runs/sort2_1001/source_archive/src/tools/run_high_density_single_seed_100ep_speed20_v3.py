"""Plan, preflight, run, resume, and summarize speed20-v3 experiments."""

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
    / "high_density_single_seed_100ep_speed20_v3"
    / "protocol.json"
)
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results_hd_ss100_s20_v3"
TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "tools"
    / "train_high_density_single_seed_100ep_speed20_v3.py"
)
_SUBCOMMANDS = {"plan", "preflight", "run", "status", "summarize"}


def _with_speed20_v3_defaults(argv: list[str]) -> list[str]:
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
    return shared_runner.main(_with_speed20_v3_defaults(values))


if __name__ == "__main__":
    raise SystemExit(main())
