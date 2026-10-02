"""一次性验证：手工保守让行控制器能否在 intersection 上稳定让行成功（P2 示范来源）。"""
import sys
from math import hypot
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import train_intersection_hold35k_mst_fixed as base

JUNCTION = "J1"


def decide(conn, ego_id):
    ex, ey = conn.vehicle.getPosition(ego_id)
    d_ego = hypot(ex, ey)  # J1 在 (0,0)
    near = []
    for vid in conn.vehicle.getIDList():
        if vid == ego_id:
            continue
        vx, vy = conn.vehicle.getPosition(vid)
        d = hypot(vx, vy)
        if d < 45:
            near.append((vid, d))
    if d_ego < 25:
        if near:
            return 0.0
        return 10.0
    if near and d_ego < 55:
        return 3.0
    return 9.0


def main():
    overlay_root = _THIS_DIR / "_p2_verify_overlay"
    env = base.make_env_factory("base", overlay_root)(base._environment_namespace(), evaluation=False)
    n_ep = 30
    succ = 0
    details = []
    for ep in range(n_ep):
        env._traffic_episode_index = ep
        env._traffic_roll = None
        obs, info = env.reset(seed=base.SEED_START["mst_slt"] + ep)
        conn = env._connection
        ego_id = env.specification.ego_id
        done = False
        steps = 0
        while not done and steps < 250:
            speed = decide(conn, ego_id)
            action = np.array([speed / 5.0 - 1.0, 0.0], dtype=np.float32)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        ok = bool(info.get("is_success"))
        succ += int(ok)
        details.append((ep, ok, bool(info.get("collision")), bool(info.get("max_time")), steps))
    env.close()
    print(f"success={succ}/{n_ep}")
    print("ep ok collision timeout steps")
    for d in details:
        print(d)


if __name__ == "__main__":
    main()
