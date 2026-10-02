"""Run one job from the distinct single-seed, 100-episode v2 study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sumo.high_density_single_seed_100ep_v2 import (  # noqa: E402
    HighDensitySingleSeed100EpisodeEnvV2,
    HighDensitySingleSeed100EpisodeEnvV4V2,
    SINGLE_SEED_100EP_EXPERIMENT_ID,
)
from tools import train_high_density_same_scene_v1 as shared_trainer  # noqa: E402


DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "high_density_single_seed_100ep_v2"
    / "protocol.json"
)
OVERLAY_DIRECTORY_NAME = "overlays_hd_ss100_v2"
RECEIPT_SCHEMA_VERSION = "high-density-single-seed-100ep-job-receipt/v2"
TENSORBOARD_DIRECTORY_NAME = "_tb_hd_ss100_v2"


def _make_v2_environment_factory(
    *,
    adapter: str,
    density: dict[str, Any],
    overlay_root: Path,
    baseline_environment_class: type[Any] = HighDensitySingleSeed100EpisodeEnvV2,
    v4_environment_class: type[Any] = HighDensitySingleSeed100EpisodeEnvV4V2,
) -> Callable[..., Any]:
    """Map v4 stage labels onto the matched physical 80/20 partition.

    v4 calibration explicitly requests ``train`` even though it constructs an
    evaluation-style environment.  Final v4 evaluation requests ``validation``;
    physically this is the same held-out 20% used by all six methods.
    """

    environment_class = (
        baseline_environment_class if adapter == "base" else v4_environment_class
    )

    def make_environment(args: argparse.Namespace, *, evaluation: bool = False):
        if not evaluation:
            physical_partition = "train"
            contract_partition = "train"
        elif adapter == "base":
            physical_partition = "evaluation"
            contract_partition = "evaluation"
        else:
            contract_partition = str(
                getattr(args, "evaluation_split", "validation")
            )
            if contract_partition == "train":
                physical_partition = "train"
            elif contract_partition in {"validation", "test"}:
                physical_partition = "evaluation"
            else:
                raise ValueError(
                    f"unsupported v4 evaluation split {contract_partition!r}"
                )
        return environment_class(
            scenario=args.scenario,
            history_steps=args.history_steps,
            neighbors=args.neighbors,
            path_length=args.path_length,
            action_repeat=args.action_repeat,
            reward_discount=args.discount,
            ego_control_profile=args.ego_control_profile,
            include_state_lstm=False,
            state_lstm_only=False,
            episode_limit_profile=args.episode_limit_profile,
            render_mode="human" if args.gui and not evaluation else None,
            high_density_vehicle_scale=float(density["vehicle_scale"]),
            high_density_pedestrian_scale=float(density["pedestrian_scale"]),
            high_density_clone_jitter_seconds=tuple(
                float(value) for value in density["clone_depart_jitter_seconds"]
            ),
            high_density_overlay_root=overlay_root,
            high_density_partition=physical_partition,
            high_density_contract_partition=contract_partition,
        )

    return make_environment


def _with_v2_protocol(argv: list[str]) -> list[str]:
    output = list(argv)
    if "--protocol" not in output:
        output.extend(("--protocol", str(DEFAULT_PROTOCOL_PATH)))
    return output


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else list(argv)
    return shared_trainer.main(
        _with_v2_protocol(values),
        baseline_environment_class=HighDensitySingleSeed100EpisodeEnvV2,
        v4_environment_class=HighDensitySingleSeed100EpisodeEnvV4V2,
        overlay_directory_name=OVERLAY_DIRECTORY_NAME,
        receipt_schema_version=RECEIPT_SCHEMA_VERSION,
        comparison_experiment_id=SINGLE_SEED_100EP_EXPERIMENT_ID,
        environment_factory_builder=_make_v2_environment_factory,
        tensorboard_log_root=PROJECT_ROOT / TENSORBOARD_DIRECTORY_NAME,
    )


if __name__ == "__main__":
    raise SystemExit(main())
