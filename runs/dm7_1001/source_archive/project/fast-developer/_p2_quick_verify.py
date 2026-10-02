"""P2 快速验证：恢复 ego speed mode 后，满速期望能否让 SUMO 自动让行。

用法: python fast-developer/_p2_quick_verify.py [mode] [expect] [n_ep]
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
    mode = int(sys.argv[1]) if len(sys.argv) > 1 else 31
    expect = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    overlay_root = _THIS_DIR / "_p2_verify_overlay"
    env = base.make_env_factory("base", overlay_root)(base._environment_namespace(), evaluation=False)
    succ = coll = timeout = 0
    for ep in range(n_ep):
        env._traffic_episode_index = ep
        env._traffic_roll = None
        obs, info = env.reset(seed=base.SEED_START["mst_slt"] + ep)
        conn = env._connection
        ego_id = env.specification.ego_id
        conn.vehicle.setSpeedMode(ego_id, mode)
        done = False
        steps = 0
        while not done and steps < 250:
            obs, r, term, trunc, info = env.step(np.array([expect, 0.0], dtype=np.float32))
            done = term or trunc
            steps += 1
        ok = bool(info.get("is_success"))
        is_coll = bool(info.get("collision"))
        is_to = bool(info.get("max_time"))
        succ += int(ok)
        coll += int(is_coll)
        timeout += int(is_to)
        print(f"ep{ep} ok={ok} coll={is_coll} timeout={is_to} steps={steps}", flush=True)
    env.close()
    print(f"RESULT mode={mode} expect={expect}: success={succ}/{n_ep} collision={coll} timeout={timeout}", flush=True)


if __name__ == "__main__":
    main()
