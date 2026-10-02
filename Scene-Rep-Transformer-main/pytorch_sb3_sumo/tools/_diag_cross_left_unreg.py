"""Headless diagnostic for cross_left_unreg visualization issues.

Answers three questions empirically:
  1. Do the traffic/ego vTypes carry a shape class?  Does the runtime
     ``setShapeClass`` beautification stick?
  2. How many background vehicles are active over time (density)?
  3. How does the ego's lateral (lane-change) control behave?
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from envs.sumo.independent_v2_five_methods_six_scenarios_100ep_v1 import (  # noqa: E402
    IndependentV2FiveBySixEnvV1,
)
from tools.three_scene_hold35k_vs_mst_v1.common import density  # noqa: E402
from tools.three_scene_visual_verification import _beautify_rendering  # noqa: E402

SCENE = "cross_left_unreg"
den = density(SCENE)

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
    render_mode=None,
    high_density_vehicle_scale=float(den["vehicle_scale"]),
    high_density_pedestrian_scale=float(den["pedestrian_scale"]),
    high_density_clone_jitter_seconds=tuple(
        float(v) for v in den["clone_depart_jitter_seconds"]
    ),
    high_density_overlay_root=PROJECT_ROOT / "results_cross_left_unreg_preview" / "overlays" / "base" / "seed_0",
    high_density_partition="evaluation",
    high_density_contract_partition="evaluation",
    sumo_args=[],
)

try:
    obs, _ = env.reset(seed=10000)
    conn = env._connection

    print("=== vTypes ===")
    type_ids = sorted(conn.vehicletype.getIDList())
    for tid in type_ids:
        try:
            vc = conn.vehicletype.getVehicleClass(tid)
            sc = conn.vehicletype.getShapeClass(tid)
        except BaseException as exc:
            print(f"  {tid!r}: err={exc!r}")
            continue
        print(f"  {tid!r}: vClass={vc} shapeClass={sc}")

    print("\n=== after _beautify_rendering ===")
    _beautify_rendering(env)
    for tid in type_ids:
        try:
            sc = conn.vehicletype.getShapeClass(tid)
        except BaseException as exc:
            print(f"  {tid!r}: err={exc!r}")
            continue
        print(f"  {tid!r}: shapeClass={sc}")

    print("\n=== active vehicles / ego over time ===")
    import numpy as np

    for step in range(80):
        # fixed-ish action: moderate speed, no lateral
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        if step % 10 == 0:
            vids = conn.vehicle.getIDList()
            ego = "ego" in vids
            road = conn.vehicle.getRoadID("ego") if ego else "-"
            lane = conn.vehicle.getLaneIndex("ego") if ego else -1
            speed = conn.vehicle.getSpeed("ego") if ego else -1.0
            print(
                f"  step={step:2d} active={len(vids):3d} ego={ego} "
                f"road={road} lane={lane} speed={speed:.2f}",
                flush=True,
            )
        if term or trunc:
            print(f"  -> episode ended at step {step}: {info}")
            break
finally:
    env.close()
