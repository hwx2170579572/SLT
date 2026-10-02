"""Run all v4.9 promotion jobs under the frozen v4.9.1 acceptance view."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_all_v4_9_promotion as automation
from tools import run_topo_v4_9_1_experiments as patch


def main(argv: list[str] | None = None) -> int:
    patch.validate_patch_contract()
    patch.validate_preregistration()
    patch.validate_patch_freeze()
    patch.validate_engineering_receipt()
    with patch.patched_acceptance():
        return automation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
