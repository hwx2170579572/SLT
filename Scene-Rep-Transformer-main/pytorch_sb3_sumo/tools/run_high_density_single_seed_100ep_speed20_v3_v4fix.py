"""Resume speed20-v3 with the isolated v4 short-TensorBoard trainer."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import run_high_density_same_scene_v1 as shared_runner  # noqa: E402
from tools import run_high_density_single_seed_100ep_speed20_v3 as parent  # noqa: E402


TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "tools"
    / "train_high_density_single_seed_100ep_speed20_v3_v4fix.py"
)


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else list(argv)
    shared_runner.TRAIN_SCRIPT = TRAIN_SCRIPT
    return shared_runner.main(parent._with_speed20_v3_defaults(values))


if __name__ == "__main__":
    raise SystemExit(main())
