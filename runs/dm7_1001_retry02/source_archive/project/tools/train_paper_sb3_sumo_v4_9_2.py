"""Run strict target-only v4.9.2 models on the paper SUMO protocol."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_sb3_v4_9_2
from tools.train_paper_sb3_sumo_v4_5 import _make_paper_env_v4_5


def main(argv: list[str] | None = None) -> int:
    return train_sb3_v4_9_2.main(
        argv,
        env_factory=_make_paper_env_v4_5,
        default_output_dir=ROOT / "results_topo_v4_9_2",
        require_paper_evaluation_contract=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
