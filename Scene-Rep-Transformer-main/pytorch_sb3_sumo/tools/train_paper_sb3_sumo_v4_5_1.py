"""Run frozen v4.5 training with the v4.5.1 selector compatibility patch."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import train_sb3_v4_5
from tools.checkpoint_selector_v4_5_1 import select_checkpoint
from tools.train_paper_sb3_sumo_v4_5 import _make_paper_env_v4_5


def main(argv: list[str] | None = None) -> int:
    # The parent main resolves this module global at the selection boundary.
    # No model/training/evaluation function is replaced.
    train_sb3_v4_5.select_checkpoint = select_checkpoint
    return train_sb3_v4_5.main(
        argv,
        env_factory=_make_paper_env_v4_5,
        default_output_dir=PROJECT_ROOT / "results_topo_v4_5",
        require_paper_evaluation_contract=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
