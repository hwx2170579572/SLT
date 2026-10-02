"""Capture SUMO-GUI screenshots of cross_left_unreg to inspect rendering.

Saves three PNGs: before beautify, after beautify (car shapes), and a later
frame after several steps.  Also logs ego lane/speed over steps to diagnose sway.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from envs.sumo.independent_v2_five_methods_six_scenarios_100ep_v1 import (  # noqa: E402
    IndependentV2FiveBySixEnvV1,
)
from tools.three_scene_hold35k_vs_mst_v1.common import density  # noqa: E402
from tools.three_scene_visual_verification import (  # noqa: E402
    _beautify_rendering,
    _install_visible_sumo_spawn,
)

OUT = PROJECT_ROOT / "tools" / "_diag_shots"
OUT.mkdir(parents=True, exist_ok=True)

SCENE = "cross_left_unreg"
den = density(SCENE)
_install_visible_sumo_spawn()

env = IndependentV2FiveBySixEnvV1(
    scenario=SCENE,
    history_steps=10,
    neighbors=5,
    path_length=10,
    action_repeat=3,
    reward_discount=0.99,
    ego_control_profile="direct",
    include_state_lstm=False,
    state_lstm_only=False,
    episode_limit_profile="source",
    render_mode="human",
    high_density_vehicle_scale=float(den["vehicle_scale"]),
    high_density_pedestrian_scale=float(den["pedestrian_scale"]),
    high_density_clone_jitter_seconds=tuple(
        float(v) for v in den["clone_depart_jitter_seconds"]
    ),
    high_density_overlay_root=PROJECT_ROOT / "results_cross_left_unreg_preview" / "overlays" / "base" / "seed_0",
    high_density_partition="evaluation",
    high_density_contract_partition="evaluation",
    sumo_args=["--window-pos", "40,40", "--window-size", "1280,800"],
)

try:
    obs, _ = env.reset(seed=10000)
    conn = env._connection
    gui = conn.gui
    views = gui.getIDList()
    print("views:", views, flush=True)
    view = views[0] if views else "View #0"

    time.sleep(1.0)
    gui.screenshot(view, str(OUT / "01_before_beautify.png"))
    print("saved 01_before_beautify.png", flush=True)

    _beautify_rendering(env)
    time.sleep(0.5)
    gui.screenshot(view, str(OUT / "02_after_beautify.png"))
    print("saved 02_after_beautify.png", flush=True)

    # Drive a few decision steps with a lateral-commanding policy-like action.
    for step in range(30):
        # oscillating lateral to see the lane-change behaviour
        lateral = 0.8 if (step % 4) < 2 else -0.8
        action = np.array([0.0, lateral], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        if step % 6 == 0:
            road = conn.vehicle.getRoadID("ego")
            lane = conn.vehicle.getLaneIndex("ego")
            speed = conn.vehicle.getSpeed("ego")
            print(f"  step={step} road={road} lane={lane} speed={speed:.2f}", flush=True)
        if term or trunc:
            print(f"  ended step={step}: {info}")
            break
    gui.screenshot(view, str(OUT / "03_after_steps.png"))
    print("saved 03_after_steps.png", flush=True)
finally:
    env.close()
