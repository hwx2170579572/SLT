"""P2 验证：恢复 ego speed mode（含 right-of-way），看 SUMO 是否自动让行。

假设：reset 里 _configure_policy_controlled_ego 设 setSpeedMode(ego,0) 关闭让行；
若 reset 后手动恢复 setSpeedMode(ego, mode)，再每步给「满速期望」action，
Krauss 的 right-of-way(bit2) 会在路口自动减速，ego 的**实际速度**即「让行调节轨迹」。
本脚本验证该轨迹能否 30/30 success，并打印实际速度曲线。
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

SPEED_MODES = [0, 7, 15, 31]   # 0=关闭(基线), 7=bit0-2, 15=bit0-3, 31=默认全开
EXPECT_ACTIONS = [
    ("full", 1.0),     # 满速期望 10 m/s
    ("mid", 0.4),      # 中速期望 7 m/s
]


def run(mode, expect_name, expect, n_ep=30):
    overlay_root = _THIS_DIR / "_p2_verify_overlay"
    env = base.make_env_factory("base", overlay_root)(base._environment_namespace(), evaluation=False)
    succ = 0
    coll = 0
    speed_profiles = []
    for ep in range(n_ep):
        env._traffic_episode_index = ep
        env._traffic_roll = None
        obs, info = env.reset(seed=base.SEED_START["mst_slt"] + ep)
        conn = env._connection
        ego_id = env.specification.ego_id
        conn.vehicle.setSpeedMode(ego_id, mode)  # 覆盖 reset 的 setSpeedMode(0)
        done = False
        steps = 0
        speeds = []
        while not done and steps < 250:
            action = np.array([expect, 0.0], dtype=np.float32)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
            if ego_id in conn.vehicle.getIDList():
                speeds.append(conn.vehicle.getSpeed(ego_id))
        ok = bool(info.get("is_success"))
        is_coll = bool(info.get("collision"))
        succ += int(ok)
        coll += int(is_coll)
        if ep == 0:
            speed_profiles = speeds
    env.close()
    return succ, coll, speed_profiles


def main():
    print("mode  expect  success  collision")
    for mode in SPEED_MODES:
        for name, expect in EXPECT_ACTIONS:
            succ, coll, prof = run(mode, name, expect)
            print(f"{mode:4d}  {name:6s}  {succ}/30    {coll}/30")
            if mode == 31 and name == "full" and succ >= 28:
                # 打印速度曲线，确认「让行减速」确实发生
                n = len(prof)
                print("  speed profile (每10步采样):", [round(s, 1) for s in prof[::max(1, n // 20)]])
                print("  min speed:", round(min(prof), 2), " max:", round(max(prof), 2), " steps:", n)


if __name__ == "__main__":
    main()
