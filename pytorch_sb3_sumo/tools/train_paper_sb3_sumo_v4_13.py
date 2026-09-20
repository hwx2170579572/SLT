"""Run v4.13 gradient-isolated tempered support on frozen paper SUMO."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import train_sb3_v4_13
from tools.train_paper_sb3_sumo_v4_5 import _make_paper_env_v4_5


def main(argv: list[str] | None = None) -> int:
    return train_sb3_v4_13.main(
        argv,
        env_factory=_make_paper_env_v4_5,
        default_output_dir=PROJECT_ROOT / "results_topo_v4_13",
        require_paper_evaluation_contract=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
