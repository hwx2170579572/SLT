"""Live SUMO-GUI preview of ``cross_left_unreg`` (random-policy, no checkpoint).

Before (or while) the new uncontrolled 4-way scenario is trained, this drives the
exact pipeline environment contract used by the trainers/evaluators but with a
*visible* SUMO-GUI window and a random policy, so the layout, junction behaviour
and 0.2 veh/s-per-approach traffic density can be inspected directly.

Nothing here trains, evaluates a checkpoint, or writes to the results tree; it is
a pure visual sanity check.  ``--method`` selects the environment contract (and
therefore the overlay root), but ``cross_left_unreg`` runs at ``vehicle_scale=1.0``
(empty additive overlay), so every method observes identical traffic.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.three_scene_hold35k_vs_mst_v1.common import (  # noqa: E402
    ALL_METHODS,
    ALL_SCENES,
)
from tools.three_scene_visual_verification import (  # noqa: E402
    SEED_START,
    _beautify_rendering,
    _install_visible_sumo_spawn,
    make_live_env,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="cross_left_unreg", choices=ALL_SCENES)
    parser.add_argument("--method", default="hold35k", choices=ALL_METHODS)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args(argv)

    _install_visible_sumo_spawn()
    env = make_live_env(args.method, args.scenario)
    try:
        for episode in range(args.episodes):
            seed = SEED_START[args.method] + episode
            observation, _ = env.reset(seed=seed)
            _beautify_rendering(env)
            decision_steps = 0
            episode_return = 0.0
            while True:
                action = env.action_space.sample()  # random policy: layout/density check only
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(info.get("undiscounted_reward", reward))
                decision_steps += 1
                time.sleep(0.3 / args.speed)
                if terminated or truncated:
                    break
            print(
                f"[ep {episode}] success={bool(info.get('is_success'))} "
                f"collision={bool(info.get('collision'))} "
                f"off_route={bool(info.get('off_route'))} "
                f"timeout={bool(info.get('max_time'))} "
                f"steps={decision_steps} return={episode_return:.1f}",
                flush=True,
            )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
