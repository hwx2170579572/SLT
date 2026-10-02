"""Run v4.5 confident-fusion methods on the frozen paper SUMO protocol."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sumo.paper_env_v4 import PaperSumoSceneEnvV4
from tools import train_sb3_v4_5


def _make_paper_env_v4_5(
    args, *, evaluation: bool = False
) -> PaperSumoSceneEnvV4:
    partition = (
        "all"
        if args.traffic_protocol == "source_all"
        else (args.evaluation_split if evaluation else "train")
    )
    return PaperSumoSceneEnvV4(
        scenario=args.scenario,
        history_steps=args.history_steps,
        neighbors=args.neighbors,
        path_length=args.path_length,
        action_repeat=args.action_repeat,
        reward_discount=args.discount,
        ego_control_profile=args.ego_control_profile,
        include_state_lstm=False,
        state_lstm_only=False,
        traffic_partition=partition,
        episode_limit_profile=args.episode_limit_profile,
        render_mode="human" if args.gui and not evaluation else None,
    )


def main(argv: list[str] | None = None) -> int:
    return train_sb3_v4_5.main(
        argv,
        env_factory=_make_paper_env_v4_5,
        default_output_dir=PROJECT_ROOT / "results_topo_v4_5",
        require_paper_evaluation_contract=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
