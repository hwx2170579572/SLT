"""P2 诊断：ego 接近路口时，冲突车流到路口中心的 ETA（gap）分布。

让 ego 以中速接近 J1(0,0)，在 d_ego<35 时停车，每 2 决策步打印周围冲突车
(id, 到路口距离 d, 速度 v, ETA=d/v)，观察「让行窗口」的实际时序。
"""
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


def diagnose(conn, ego_id):
    """返回 ego 状态 + 排序后的冲突车 (d, v, eta) 列表。"""
    ex, ey = conn.vehicle.getPosition(ego_id)
    d_ego = hypot(ex, ey)
    ego_v = conn.vehicle.getSpeed(ego_id)
    others = []
    for vid in conn.vehicle.getIDList():
        if vid == ego_id:
            continue
        vx, vy = conn.vehicle.getPosition(vid)
        d = hypot(vx, vy)
        if d > 90:
            continue
        v = conn.vehicle.getSpeed(vid)
        eta = d / v if v > 0.3 else 1e9
        others.append((round(d, 1), round(v, 2), round(eta, 2), vid))
    others.sort(key=lambda t: t[0])
    return d_ego, ego_v, others


def main():
    overlay_root = _THIS_DIR / "_p2_verify_overlay"
    env = base.make_env_factory("base", overlay_root)(base._environment_namespace(), evaluation=False)
    n_ep = 3
    for ep in range(n_ep):
        env._traffic_episode_index = ep
        env._traffic_roll = None
        obs, info = env.reset(seed=base.SEED_START["mst_slt"] + ep)
        conn = env._connection
        ego_id = env.specification.ego_id
        done = False
        steps = 0
        while not done and steps < 250:
            d_ego, ego_v, others = diagnose(conn, ego_id)
            # 控制：接近路口时减速停车，观察 gap
            if d_ego < 35:
                speed = 0.0
            elif d_ego < 60:
                speed = 4.0
            else:
                speed = 9.0
            if steps % 2 == 0 and d_ego < 40:
                print(f"[ep{ep} step{steps}] ego d={d_ego:.1f} v={ego_v:.2f} | near(<=90m): "
                      + " ".join(f"{vid}:d{d}v{v}e{eta}" for (d, v, eta, vid) in others[:8]))
            action = np.array([speed / 5.0 - 1.0, 0.0], dtype=np.float32)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        print(f"--- ep{ep} done ok={bool(info.get('is_success'))} "
              f"collision={bool(info.get('collision'))} timeout={bool(info.get('max_time'))} steps={steps}\n")
    env.close()


if __name__ == "__main__":
    main()
