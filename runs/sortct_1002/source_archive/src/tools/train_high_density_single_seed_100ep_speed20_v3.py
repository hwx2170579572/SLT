"""Run one isolated speed-[0, 20] job derived from independent v2."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sumo.high_density_single_seed_100ep_speed20_v3 import (  # noqa: E402
    HighDensitySingleSeed100EpisodeSpeed20EnvV3,
    HighDensitySingleSeed100EpisodeSpeed20EnvV4V3,
    SPEED20_V3_EXPERIMENT_ID,
)
from tools import train_high_density_same_scene_v1 as shared_trainer  # noqa: E402
from tools import train_high_density_single_seed_100ep_v2 as v2_trainer  # noqa: E402


DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "high_density_single_seed_100ep_speed20_v3"
    / "protocol.json"
)
OVERLAY_DIRECTORY_NAME = "overlays_hd_ss100_s20_v3"
RECEIPT_SCHEMA_VERSION = (
    "high-density-single-seed-100ep-speed20-job-receipt/v3"
)
TENSORBOARD_DIRECTORY_NAME = "_tb_hd_ss100_s20_v3"


def _make_speed20_v3_environment_factory(
    *,
    adapter: str,
    density: dict[str, Any],
    overlay_root: Path,
    baseline_environment_class: type[Any] = (
        HighDensitySingleSeed100EpisodeSpeed20EnvV3
    ),
    v4_environment_class: type[Any] = (
        HighDensitySingleSeed100EpisodeSpeed20EnvV4V3
    ),
) -> Callable[..., Any]:
    """Reuse the validated v2 partition mapping with new environment classes."""

    return v2_trainer._make_v2_environment_factory(
        adapter=adapter,
        density=density,
        overlay_root=overlay_root,
        baseline_environment_class=baseline_environment_class,
        v4_environment_class=v4_environment_class,
    )


def _with_speed20_v3_protocol(argv: list[str]) -> list[str]:
    output = list(argv)
    if "--protocol" not in output:
        output.extend(("--protocol", str(DEFAULT_PROTOCOL_PATH)))
    return output


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else list(argv)
    return shared_trainer.main(
        _with_speed20_v3_protocol(values),
        baseline_environment_class=(
            HighDensitySingleSeed100EpisodeSpeed20EnvV3
        ),
        v4_environment_class=(
            HighDensitySingleSeed100EpisodeSpeed20EnvV4V3
        ),
        overlay_directory_name=OVERLAY_DIRECTORY_NAME,
        receipt_schema_version=RECEIPT_SCHEMA_VERSION,
        comparison_experiment_id=SPEED20_V3_EXPERIMENT_ID,
        environment_factory_builder=_make_speed20_v3_environment_factory,
        tensorboard_log_root=PROJECT_ROOT / TENSORBOARD_DIRECTORY_NAME,
    )


if __name__ == "__main__":
    raise SystemExit(main())
