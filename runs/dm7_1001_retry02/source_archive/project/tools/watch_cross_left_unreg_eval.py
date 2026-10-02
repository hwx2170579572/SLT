"""Live SUMO-GUI playback of the trained hold35k / mst_slt policies on
``cross_left_unreg``.

Loads a checkpoint produced by ``tools/train_cross_left_unreg_preview.py`` (under
``results_cross_left_unreg_preview/``) and drives the ego vehicle through the
completely-uncontrolled 4-way junction with that policy in a *visible* SUMO-GUI
window, so the vehicle driving scenes of the evaluation can be watched directly.

Inference-only: it loads one checkpoint, replays a few deterministic evaluation
episodes with the same seed block and traffic-cycling as the headless evaluation
in ``train_cross_left_unreg_preview.py``, and prints each episode's outcome.
Nothing is trained, evaluated headlessly, or overwritten.

Usage (from the repo root ``pytorch_sb3_sumo``):

    python tools/watch_cross_left_unreg_eval.py --method hold35k --raw-step 10000
    python tools/watch_cross_left_unreg_eval.py --method mst_slt --final --episodes 5

``--method`` selects the environment contract (``v4_8`` hybrid head for
``hold35k``, ``base`` continuous for ``mst_slt``).  ``cross_left_unreg`` runs at
``vehicle_scale=1.0`` (empty additive overlay), so both methods observe the same
physical traffic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.three_scene_visual_verification import (  # noqa: E402
    SEED_START,
    _beautify_rendering,
    _install_visible_sumo_spawn,
    load_sb3_model,
    live_sb3_episode,
    make_live_env,
)

RESULT_ROOT = PROJECT_ROOT / "results_cross_left_unreg_preview"
SCENE = "cross_left_unreg"
METHODS = ("hold35k", "mst_slt")
METHOD_LABELS = {"hold35k": "TASAC (hold35k)", "mst_slt": "MST+SLT"}
CONTRACT = {"hold35k": "v4_8", "mst_slt": "base"}


def _checkpoint(method: str, raw_step: int) -> Path:
    prefix = "ckpt" if method == "hold35k" else "scene_rep"
    return (
        RESULT_ROOT
        / "train"
        / f"{method}__{SCENE}"
        / "checkpoints"
        / f"{prefix}_raw_{raw_step}_steps.zip"
    )


def _final_model(method: str) -> Path:
    return RESULT_ROOT / "train" / f"{method}__{SCENE}" / "final_model.zip"


def _latest_checkpoint(method: str) -> Path:
    """Return the highest raw-step checkpoint present for ``method``."""
    directory = RESULT_ROOT / "train" / f"{method}__{SCENE}" / "checkpoints"
    prefix = "ckpt" if method == "hold35k" else "scene_rep"
    steps: list[int] = []
    for path in directory.glob(f"{prefix}_raw_*_steps.zip"):
        try:
            steps.append(int(path.stem.split("_raw_")[1].split("_steps")[0]))
        except (IndexError, ValueError):
            continue
    if not steps:
        raise FileNotFoundError(f"no {prefix} checkpoints under {directory}")
    return _checkpoint(method, max(steps))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="hold35k")
    parser.add_argument("--raw-step", type=int, default=None,
                        help="checkpoint raw step to load (default: the latest checkpoint)")
    parser.add_argument("--final", action="store_true",
                        help="load final_model.zip instead of a step checkpoint")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-beautify", action="store_true",
                        help="disable car/pedestrian silhouette beautification")
    args = parser.parse_args(argv)

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.speed <= 0:
        parser.error("--speed must be positive")

    if args.final:
        checkpoint = _final_model(args.method)
    elif args.raw_step is not None:
        checkpoint = _checkpoint(args.method, args.raw_step)
    else:
        checkpoint = _latest_checkpoint(args.method)
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")

    import numpy as np
    import torch

    torch.set_num_threads(1)
    _install_visible_sumo_spawn()
    env = make_live_env(args.method, SCENE)
    contract = CONTRACT[args.method]
    try:
        model = load_sb3_model(args.method, checkpoint, env, args.device)
        raw_seen = int(getattr(model, "_raw_steps_seen", -1))
        print(
            f"[load] {METHOD_LABELS[args.method]} / {SCENE}\n"
            f"       checkpoint = {checkpoint}\n"
            f"       raw_steps_seen(model) = {raw_seen}",
            flush=True,
        )
        for index in range(args.episodes):
            seed = SEED_START[args.method] + index
            np.random.seed(seed + 600000)
            torch.manual_seed(seed + 600000)
            if contract == "v4_8":
                env._traffic_episode_index, env._traffic_roll = index, None
            record = live_sb3_episode(
                model, env, episode=index, seed=seed, speed=args.speed,
                beautify=not args.no_beautify,
            )
            print(
                f"[ep {index}] seed={seed} success={record['success']} "
                f"collision={record['collision']} off_route={record['off_route']} "
                f"timeout={record['timeout']} return={record['episode_return']:.1f} "
                f"raw_steps={record['raw_steps']}",
                flush=True,
            )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
