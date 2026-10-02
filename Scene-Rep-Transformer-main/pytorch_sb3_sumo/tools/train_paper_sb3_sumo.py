"""Train through the original released SUMO networks and traffic variants."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sumo.paper_env import PaperSumoSceneEnv
from envs.sumo.ppo_env import PaperPpoCarlaEnv, PaperPpoRgbEnv
from tools import train_sb3


def _make_paper_env(args, *, evaluation: bool = False) -> PaperSumoSceneEnv:
    traffic_partition = (
        "all"
        if args.traffic_protocol == "source_all"
        else ("evaluation" if evaluation else "train")
    )
    if args.algo == "ppo":
        ppo_env_class = PaperPpoCarlaEnv if args.scenario == "carla" else PaperPpoRgbEnv
        return ppo_env_class(
            scenario=args.scenario,
            action_repeat=args.action_repeat,
            reward_discount=args.discount,
            training=not evaluation,
            ego_control_profile=args.ego_control_profile,
            traffic_partition=traffic_partition,
            episode_limit_profile=args.episode_limit_profile,
            render_mode="human" if args.gui and not evaluation else None,
        )
    return PaperSumoSceneEnv(
        scenario=args.scenario,
        history_steps=args.history_steps,
        neighbors=args.neighbors,
        path_length=args.path_length,
        action_repeat=args.action_repeat,
        reward_discount=args.discount,
        ego_control_profile=args.ego_control_profile,
        include_state_lstm=args.algo == "sac",
        state_lstm_only=args.algo == "sac",
        traffic_partition=traffic_partition,
        episode_limit_profile=args.episode_limit_profile,
        render_mode="human" if args.gui and not evaluation else None,
    )


def main(argv: list[str] | None = None) -> int:
    return train_sb3.main(
        argv,
        env_factory=_make_paper_env,
        default_output_dir=PROJECT_ROOT / "results_sb3_sumo_paper",
        require_paper_evaluation_contract=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
