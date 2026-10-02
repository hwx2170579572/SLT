"""P2 快速评估：BC 预训练后的 mst_slt actor 能否直接让行（不微调）。"""
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
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    deterministic = len(sys.argv) > 3 and sys.argv[3] == "stoch"

    overlay_root = _THIS_DIR / "_p2_verify_overlay"
    env = base.make_env_factory("base", overlay_root)(base._environment_namespace(), evaluation=False)
    model = SceneRepresentationSAC.load(str(model_path), env=env, device="cpu", buffer_size=32)

    succ = coll = timeout = 0
    for ep in range(n_ep):
        env._traffic_episode_index = ep
        env._traffic_roll = None
        obs, info = env.reset(seed=base.SEED_START["mst_slt"] + ep)
        done = False
        steps = 0
        while not done and steps < 250:
            action, _ = model.predict(obs, deterministic=not deterministic)
            obs, r, term, trunc, info = env.step(action)
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
    print(f"RESULT success={succ}/{n_ep} collision={coll} timeout={timeout}", flush=True)


if __name__ == "__main__":
    main()
