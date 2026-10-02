"""Use the frozen v4.5 orchestrator with the v4.5.1 training entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import run_topo_v4_5_experiments as base


_BASE_COMMAND_FOR = base.command_for
_PATCHED_TRAINER = PROJECT_ROOT / "tools" / "train_paper_sb3_sumo_v4_5_1.py"


def command_for(job: Any, hashes: dict[str, str], *, device: str) -> list[str]:
    command = _BASE_COMMAND_FOR(job, hashes, device=device)
    if len(command) < 2:
        raise ValueError("v4.5 command is missing its training entry point")
    command[1] = str(_PATCHED_TRAINER.resolve())
    return command


def main(argv: list[str] | None = None) -> int:
    base.command_for = command_for
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
