"""Run speed20-v3 through the Windows extended-length path namespace.

This recovery entry point is intentionally separate from the frozen speed20-v3
runner.  It changes only the filesystem spelling of ``--result-root``; the
protocol, job IDs, trainer, environments, seeds, and evaluation budget remain
unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (  # noqa: E402
    run_high_density_single_seed_100ep_speed20_v3 as frozen_runner,
)


WINDOWS_EXTENDED_PREFIX = "\\\\?\\"
WINDOWS_UNC_PREFIX = "\\\\"
WINDOWS_EXTENDED_UNC_PREFIX = "\\\\?\\UNC\\"
_SUBCOMMANDS = {"plan", "preflight", "run", "status", "summarize"}


def windows_extended_path(path: Path) -> Path:
    """Return an absolute path that bypasses legacy Windows MAX_PATH limits."""

    text = str(path)
    if os.name != "nt" or text.startswith(WINDOWS_EXTENDED_PREFIX):
        return path
    resolved = str(path.resolve())
    if resolved.startswith(WINDOWS_UNC_PREFIX):
        return Path(WINDOWS_EXTENDED_UNC_PREFIX + resolved[2:])
    return Path(WINDOWS_EXTENDED_PREFIX + resolved)


def _with_extended_result_root(argv: list[str]) -> list[str]:
    output = list(argv)
    if not output or output[0] not in _SUBCOMMANDS:
        return output
    if "--result-root" in output:
        index = output.index("--result-root") + 1
        if index >= len(output):
            raise ValueError("--result-root requires a path")
        output[index] = str(windows_extended_path(Path(output[index])))
    else:
        output.extend(
            (
                "--result-root",
                str(windows_extended_path(frozen_runner.DEFAULT_RESULT_ROOT)),
            )
        )
    return output


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else list(argv)
    return frozen_runner.main(_with_extended_result_root(values))


if __name__ == "__main__":
    raise SystemExit(main())
