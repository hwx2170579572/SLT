"""P2 验证：完全不干预 ego（保持 SUMO 默认 speed mode，不 setSpeed），看是否自动让行。

这是 memory 里「SUMO 默认让行（不干预 ego）30/30」的复现：reset 后不 setSpeedMode(0)，
也不 setSpeed，让 SUMO 的 Krauss + route 追踪自动驾驶 ego。若 30/30 success，则 ego
的实际速度序列即「让行速度调节轨迹」，可作为 P2 的 BC 示范。
"""
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import train_intersection_hold35k_mst_fixed as base


def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    restore_mode = int(sys.argv[2]) if len(sys.argv) > 2 else 31  # 31=默认全开

    overlay_root = _THIS_DIR / "_p2_verify_overlay"
    env = base.make_env_factory("base", overlay_root)(base._environment_namespace(), evaluation=False)
    succ = coll = timeout = off = 0
    first_profile = []
    for ep in range(n_ep):
        env._traffic_episode_index = ep
        env._traffic_roll = None
        obs, info = env.reset(seed=base.SEED_START["mst_slt"] + ep)
        conn = env._connection
        ego_id = env.specification.ego_id
        # 关键：恢复默认 speed mode（reset 里已设 0），且此后不 setSpeed
        conn.vehicle.setSpeedMode(ego_id, restore_mode)
        done = False
        raw = 0
        speeds = []
        while not done and raw < env.max_episode_steps:
            conn.simulationStep()
            env._after_simulation_step()
            env._raw_steps += 1
            env._lifetime_raw_steps += 1
            env._record_histories()
            success, collision, off_route, max_time = env._events_after_step()
            raw += 1
            if ego_id in conn.vehicle.getIDList():
                speeds.append(conn.vehicle.getSpeed(ego_id))
            if success or collision or off_route or max_time:
                done = True
        succ += int(success)
        coll += int(collision)
        timeout += int(max_time)
        off += int(off_route)
        if ep == 0:
            first_profile = speeds
        if success:
            n = len(speeds)
            print(f"ep{ep} SUCCESS raw={raw} speed_profile:",
                  [round(s, 1) for s in speeds[::max(1, n // 24)]], flush=True)
        elif ep < 5 or (ep + 1) % 10 == 0:
            print(f"ep{ep} success={success} collision={collision} timeout={max_time} "
                  f"off_route={off_route} raw={raw}", flush=True)
    env.close()
    print(f"RESULT restore_mode={restore_mode}: success={succ}/{n_ep} collision={coll} "
          f"timeout={timeout} off_route={off}", flush=True)
    if first_profile:
        n = len(first_profile)
        print("  first-ep speed profile (每~20步采样):",
              [round(s, 1) for s in first_profile[::max(1, n // 20)]])
        print(f"  min={round(min(first_profile),2)} max={round(max(first_profile),2)} n_raw={n}")


if __name__ == "__main__":
    main()


def _run_maxspeed(max_speed: float, n_ep: int = 30):
    overlay_root = _THIS_DIR / "_p2_verify_overlay"
    env = base.make_env_factory("base", overlay_root)(base._environment_namespace(), evaluation=False)
    succ = coll = 0
    prof = []
    for ep in range(n_ep):
        env._traffic_episode_index = ep
        env._traffic_roll = None
        obs, info = env.reset(seed=base.SEED_START["mst_slt"] + ep)
        conn = env._connection
        ego_id = env.specification.ego_id
        conn.vehicle.setSpeedMode(ego_id, 31)
        conn.vehicle.setMaxSpeed(ego_id, max_speed)
        done = False
        raw = 0
        speeds = []
        while not done and raw < env.max_episode_steps:
            conn.simulationStep()
            env._after_simulation_step()
            env._raw_steps += 1
            env._record_histories()
            success, collision, off_route, max_time = env._events_after_step()
            raw += 1
            if ego_id in conn.vehicle.getIDList():
                speeds.append(conn.vehicle.getSpeed(ego_id))
            if success or collision or off_route or max_time:
                done = True
        succ += int(success)
        coll += int(collision)
        if ep == 0:
            prof = speeds
    env.close()
    n = len(prof)
    print(f"maxSpeed={max_speed}: success={succ}/{n_ep} collision={coll} "
          f"v=[{', '.join(f'{s:.1f}' for s in prof[::max(1,n//16)])}]", flush=True)
