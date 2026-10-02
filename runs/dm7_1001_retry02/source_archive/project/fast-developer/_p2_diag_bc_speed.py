"""P2 诊断：打印 BC 模型在若干 episode 的 ego 速度曲线与动作，看让行减速是否发生。"""
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
from algos.sb3_torch.sac import SceneRepresentationSAC


def main():
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _THIS_DIR / "artifacts" / "mst_slt_bc_pretrained.zip"
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    overlay_root = _THIS_DIR / "_p2_verify_overlay"
    env = base.make_env_factory("base", overlay_root)(base._environment_namespace(), evaluation=False)
    model = SceneRepresentationSAC.load(str(model_path), env=env, device="cpu", buffer_size=32)

    for ep in range(n_ep):
        env._traffic_episode_index = ep
        env._traffic_roll = None
        obs, info = env.reset(seed=base.SEED_START["mst_slt"] + ep)
        conn = env._connection
        ego_id = env.specification.ego_id
        done = False
        steps = 0
        speeds = []
        acts = []
        while not done and steps < 250:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
            if ego_id in conn.vehicle.getIDList():
                speeds.append(conn.vehicle.getSpeed(ego_id))
            acts.append(float(action[0]))
        n = len(speeds)
        print(f"ep{ep} ok={bool(info.get('is_success'))} coll={bool(info.get('collision'))} "
              f"steps={steps} v=[{', '.join(f'{s:.1f}' for s in speeds[::max(1,n//16)])}]", flush=True)
        print(f"      action[0]=[{', '.join(f'{a:.2f}' for a in acts[::max(1,len(acts)//16)])}]", flush=True)
    env.close()


if __name__ == "__main__":
    main()
