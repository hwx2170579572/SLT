"""P2 BC 预训练 mst_slt 的 actor（示范暖启动）。

用 ``collect_yield_demos.py`` 采集的「让行」示范 (observation, action) 对，监督训练
mst_slt 的连续 actor，让它先学会「加速接近 -> 路口前减速 -> 择机加速穿过」的让行时序，
再交给 reward-shaping 微调。BC 只优化 actor 的完整参数（含与 critic 共享的 scene
extractor），把特征一并拉到能区分让行状态的方向。

用法::

    python fast-developer/bc_pretrain_actor.py --demos fast-developer/artifacts/yield_demos.pkl \
        --out fast-developer/artifacts/mst_slt_bc_pretrained.zip --epochs 200
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch as th

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import train_intersection_hold35k_mst_fixed as base
from configs.sb3_configs import make_model


def load_demos(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["demos"]


def build_model(overlay_root: Path, device: str):
    env = base.make_env_factory("base", overlay_root)(
        base._environment_namespace(), evaluation=False
    )
    model = make_model(
        "scene_rep",
        env,
        scenario=base.SCENARIO,
        learning_rate=base.LEARNING_RATE["mst_slt"],
        batch_size=base.BATCH_SIZE,
        discount=base.DISCOUNT,
        learning_starts=base.LEARNING_STARTS,
        buffer_size=base.BUFFER_SIZE,
        action_repeat=base.ACTION_REPEAT,
        seed=base.SEED,
        device=device,
        tensorboard_log=None,
        verbose=0,
    )
    return model, env


def stack_obs(obs_list: list[dict], device: th.device) -> dict[str, th.Tensor]:
    keys = list(obs_list[0].keys())
    return {
        k: th.as_tensor(np.stack([o[k] for o in obs_list]), dtype=th.float32).to(device)
        for k in keys
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, default=_THIS_DIR / "artifacts" / "yield_demos.pkl")
    parser.add_argument("--out", type=Path, default=_THIS_DIR / "artifacts" / "mst_slt_bc_pretrained.zip")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--reweight", type=float, default=4.0,
        help="减速段样本权重放大系数：weight = 1 + reweight * (1 - action[0])，"
             "让 BC 在路口前更倾向减速让行（保守），而非把满速/减速平均成中速。",
    )
    args = parser.parse_args(argv)

    th.set_num_threads(1)
    demos = load_demos(args.demos)
    n_demos = len(demos)
    total_steps = sum(len(d["actions"]) for d in demos)
    print(f"示范: {n_demos} 条轨迹, {total_steps} 决策步", flush=True)

    # 展平为单样本列表（obs 保持为 numpy dict，避免一次性物化所有 tensor）
    obs_list: list[dict] = []
    act_list: list[np.ndarray] = []
    for d in demos:
        obs_list.extend(d["observations"])
        act_list.append(np.asarray(d["actions"], dtype=np.float32))
    actions = np.concatenate(act_list, axis=0)
    n = len(obs_list)
    print(f"样本数 {n}, action 形状 {actions.shape}", flush=True)

    model, env = build_model(_THIS_DIR / "_p2_verify_overlay", args.device)
    device = th.device(args.device)
    model.actor.train()

    # BC 优化整个 actor（含共享 scene extractor）
    optimizer = th.optim.NAdam(model.actor.parameters(), lr=args.lr, eps=1e-7)
    act_tensor = th.as_tensor(actions, dtype=th.float32).to(device)
    # 减速段（action[0] 低）样本加权，避免 MSE 把「满速/减速」双峰平均成中速。
    w = 1.0 + args.reweight * (1.0 - actions[:, 0])
    w_tensor = th.as_tensor(w, dtype=th.float32).to(device)

    for epoch in range(args.epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            obs_batch = stack_obs([obs_list[j] for j in idx], device)
            target = act_tensor[idx]
            mean, _, _ = model.actor.get_action_dist_params(obs_batch)
            sq = (mean - target) ** 2
            loss = (sq * w_tensor[idx].unsqueeze(-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.actor.parameters(), model.max_grad_norm)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{args.epochs}  mse={epoch_loss / n_batches:.6f}",
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    env.close()
    print(f"已保存 BC 预训练模型 {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
