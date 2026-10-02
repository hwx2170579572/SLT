"""P2 验证：精确 TTC gap 检测让行控制器能否稳定穿过 intersection。

核心：ego 以「比最近冲突车晚 MARGIN 秒到路口中心」的速度接近，冲突车流间隙出现时
（min_eta 大）满速穿过。比之前「等路口清空」更精确——不停死，保持中速跟随车流，
gap 一到就加速。验证「让行」在 unregulated junction 是否本质可达，作示范来源的可靠上限。
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


def decide(conn, ego_id, margin: float, min_eta_thresh: float, max_speed: float):
    ex, ey = conn.vehicle.getPosition(ego_id)
    d_ego = hypot(ex, ey)  # J1 在 (0,0)

    # 已进入路口（含中心附近）→ 满速穿过，绝不减速（否则在路口内停车被撞）
    if d_ego < 15.0:
        return max_speed

    min_eta = 1e9
    for vid in conn.vehicle.getIDList():
        if vid == ego_id:
            continue
        vx, vy = conn.vehicle.getPosition(vid)
        d = hypot(vx, vy)
        if d > 80:
            continue
        v = conn.vehicle.getSpeed(vid)
        if v < 0.5:
            continue
        min_eta = min(min_eta, d / v)

    if min_eta > min_eta_thresh:
        return max_speed  # 无近期冲突车，满速穿过
    # 目标速度：让 ego 比冲突车晚 margin 秒到路口中心
    target = d_ego / max(min_eta + margin, 0.5)
    return max(0.0, min(max_speed, target))


def run(n_ep: int, margin: float, min_eta_thresh: float, max_speed: float = 10.0):
    overlay_root = _THIS_DIR / "_p2_verify_overlay"
    env = base.make_env_factory("base", overlay_root)(base._environment_namespace(), evaluation=False)
    succ = coll = timeout = 0
    for ep in range(n_ep):
        env._traffic_episode_index = ep
        env._traffic_roll = None
        obs, info = env.reset(seed=base.SEED_START["mst_slt"] + ep)
        conn = env._connection
        ego_id = env.specification.ego_id
        done = False
        steps = 0
        while not done and steps < 250:
            speed = decide(conn, ego_id, margin, min_eta_thresh, max_speed)
            action = np.array([speed / 5.0 - 1.0, 0.0], dtype=np.float32)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        ok = bool(info.get("is_success"))
        is_coll = bool(info.get("collision"))
        is_to = bool(info.get("max_time"))
        succ += int(ok)
        coll += int(is_coll)
        timeout += int(is_to)
    env.close()
    print(f"margin={margin} min_eta_thresh={min_eta_thresh} max_speed={max_speed}: "
          f"success={succ}/{n_ep} collision={coll} timeout={timeout}", flush=True)
    return succ


if __name__ == "__main__":
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    margin = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    min_eta_thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    max_speed = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    run(n_ep, margin, min_eta_thresh, max_speed)
