"""Run speed20-v3 jobs with an isolated Windows v4 TensorBoard redirect.

The original v4 trainers are left untouched.  Their in-run TensorBoard path is
261+ characters under the speed20-v3 namespace, so this wrapper redirects only
the logger destination to a short, hashed experiment-specific directory at
runtime.  Model, environment, protocol, seeds, budgets, and result run IDs are
unchanged.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3.common import utils as sb3_utils  # noqa: E402
from tools import train_high_density_single_seed_100ep_speed20_v3 as parent  # noqa: E402


SHORT_TENSORBOARD_ROOT = PROJECT_ROOT / "_tb_hd_ss100_s20_v3_v4fix"
REDIRECT_RECEIPT_NAME = "tb_v4fix.json"
_ORIGINAL_CONFIGURE_LOGGER = sb3_utils.configure_logger


def _short_tensorboard_directory(long_tensorboard_log: str) -> tuple[Path, Path]:
    source = Path(long_tensorboard_log).resolve()
    run_dir = source.parent if source.name == "tensorboard" else source
    token = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()[:16]
    return run_dir, (SHORT_TENSORBOARD_ROOT / token).resolve()


def _write_redirect_receipt(
    run_dir: Path, source: Path, target: Path
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "speed20-v3-v4-tensorboard-redirect/v1",
        "engineering_workaround_only": True,
        "experimental_contract_changed": False,
        "reason": "Windows legacy path limit exceeded by v4 TensorBoard event filename",
        "original_tensorboard_log": str(source),
        "redirected_tensorboard_log": str(target),
        "original_base_path_length": len(str(source)),
        "redirected_base_path_length": len(str(target)),
    }
    path = run_dir / REDIRECT_RECEIPT_NAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _configure_logger_with_short_v4_path(
    verbose: int = 0,
    tensorboard_log: str | None = None,
    tb_log_name: str = "",
    reset_num_timesteps: bool = True,
):
    if tensorboard_log is None:
        return _ORIGINAL_CONFIGURE_LOGGER(
            verbose, tensorboard_log, tb_log_name, reset_num_timesteps
        )
    run_dir, target = _short_tensorboard_directory(tensorboard_log)
    target.mkdir(parents=True, exist_ok=True)
    _write_redirect_receipt(
        run_dir,
        Path(tensorboard_log).resolve(),
        target,
    )
    return _ORIGINAL_CONFIGURE_LOGGER(
        verbose, str(target), tb_log_name, reset_num_timesteps
    )


def main(argv: list[str] | None = None) -> int:
    original = sb3_utils.configure_logger
    sb3_utils.configure_logger = _configure_logger_with_short_v4_path
    try:
        return parent.main(argv)
    finally:
        sb3_utils.configure_logger = original


if __name__ == "__main__":
    raise SystemExit(main())
