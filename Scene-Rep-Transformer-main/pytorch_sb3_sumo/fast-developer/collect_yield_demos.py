"""P2 采集「让行」示范轨迹（SUMO 默认让行，不干预 ego）。

原理：reset 后恢复 ego 的 speed mode 为默认 31（含 right-of-way bit），此后**不 setSpeed**，
让 SUMO 的 Krauss + route 追踪自动驾驶 ego。ego 在路口前的实际速度序列即「让行速度调节」
（加速接近 -> 路口前减速 ~7m/s -> 择机加速穿过 -> 满速驶离）。只保留 success 的 episode，
记录每个决策步的 (observation, action=clip(actual_speed/5-1, -1, 1)) 作为 BC 示范。

动作对齐：RL 训练时 ego speed mode=0，action[0]=(speed/5-1) 即「目标速度」；示范里用
「决策步开始时 ego 的实际速度」反推 action，等价地教会 actor「在 obs 下该输出什么速度」。

用法::

    python fast-developer/collect_yield_demos.py --n-episodes 200 --out fast-developer/artifacts/yield_demos.pkl
"""
from __future__ import annotations

import argparse
import copy
import pickle
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

RESTORE_MODE = 31          # 默认全开（含 right-of-way bit2）
SPEED_SCALE = 5.0          # adapt_action: speed = (action[0]+1)*5
MAX_ACTION_SPEED = 10.0    # action 上限 10 m/s


def _copy_obs(obs: dict) -> dict:
    return {k: np.asarray(v).copy() for k, v in obs.items()}


def collect(env, episode_index: int, max_decision_steps: int, max_speed: float = 10.0):
    """采集一个 episode 的让行示范；非 success 返回 None。

    ``max_speed`` 限制 ego 的 SUMO 自动驾驶上限：设为 10 时满速段 = action 上限 10，
    与 RL 动作空间一致，示范的「满速 1.0 -> 减速 ~0.5」对比更清晰、可学。
    """
    env._traffic_episode_index = episode_index
    env._traffic_roll = None
    obs, info = env.reset(seed=base.SEED_START["mst_slt"] + episode_index)
    conn = env._connection
    ego_id = env.specification.ego_id
    conn.vehicle.setSpeedMode(ego_id, RESTORE_MODE)
    conn.vehicle.setMaxSpeed(ego_id, max_speed)

    obs_list: list[dict] = []
    act_list: list[np.ndarray] = []
    done = False
    decision_step = 0
    success = collision = False
    while not done and decision_step < max_decision_steps:
        v = float(conn.vehicle.getSpeed(ego_id))
        action = np.array(
            [np.clip(v / SPEED_SCALE - 1.0, -1.0, 1.0), 0.0], dtype=np.float32
        )
        obs_list.append(_copy_obs(obs))
        act_list.append(action)

        for _ in range(3):  # action_repeat
            conn.simulationStep()
            env._after_simulation_step()
            env._raw_steps += 1
            env._lifetime_raw_steps += 1
            env._record_histories()
            success, collision, off_route, max_time = env._events_after_step()
            if success or collision or off_route or max_time:
                done = True
                break
        if done:
            break
        obs = _copy_obs(env._make_observation())
        decision_step += 1

    if not success:
        return None
    return {"observations": obs_list, "actions": np.asarray(act_list, dtype=np.float32)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--out", type=Path, default=_THIS_DIR / "artifacts" / "yield_demos.pkl")
    parser.add_argument("--target-demos", type=int, default=40, help="采满即停")
    args = parser.parse_args(argv)

    overlay_root = _THIS_DIR / "_p2_verify_overlay"
    env = base.make_env_factory("base", overlay_root)(base._environment_namespace(), evaluation=False)

    demos = []
    n_success = 0
    ep = 0
    while ep < args.n_episodes and len(demos) < args.target_demos:
        demo = collect(env, ep, max_decision_steps=250)
        ep += 1
        if demo is None:
            continue
        n_success += 1
        demos.append(demo)
        n = len(demo["actions"])
        print(f"ep{ep - 1} SUCCESS: {n} decision-steps  (累计 {len(demos)}/{args.target_demos})",
              flush=True)

    env.close()
    total_steps = sum(len(d["actions"]) for d in demos)
    print(f"采集完成: {len(demos)} 条成功轨迹 / {ep} 集尝试, 共 {total_steps} 决策步", flush=True)

    if demos:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as f:
            pickle.dump(
                {
                    "demos": demos,
                    "n_episodes_attempted": ep,
                    "restore_mode": RESTORE_MODE,
                    "action_repeat": base.ACTION_REPEAT,
                    "seed_start": base.SEED_START["mst_slt"],
                },
                f,
            )
        print(f"已保存 {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
