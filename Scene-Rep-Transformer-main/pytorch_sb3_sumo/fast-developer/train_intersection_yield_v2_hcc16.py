"""D4 样本采样消融：sac_mlp / hsac_mlp_base + HCC-16（视界校正 16 步回报）。

单变量测试「关键样本 credit 传递」这条能力腿：把两个 MLP 基线的 n-step 回报从
``n_steps=4`` 换成 HCC-16（``HorizonCorrectDictNStepReplayBufferV47``：n=16 +
``gamma ** 实际视界`` bootstrap），其余（编码器 / 动作头 / 熵 / 调参 / 环境契约）
完全不动，看成功率 / 平均回报是否相对父基线提升。

这是 NEW method（``sac_mlp_hcc16`` / ``hsac_mlp_base_hcc16``），各自继承父基线的
env 契约与 seed 块，与已有方法区分开，不改任何已有文件。

用法::

    python fast-developer/train_intersection_yield_v2_hcc16.py --smoke
    python fast-developer/train_intersection_yield_v2_hcc16.py              # 两臂短训练 20000 raw 步
    python fast-developer/train_intersection_yield_v2_hcc16.py --method sac_mlp_hcc16
    python fast-developer/train_intersection_yield_v2_hcc16.py --max-steps 50000
    python fast-developer/train_intersection_yield_v2_hcc16.py --eval-only \
        --method sac_mlp_hcc16 \
        --model-path fast-developer/sac_mlp_hcc16__intersection_yield_v2/final_model.zip
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_intersection_yield_v2 as base

SCENARIO = base.SCENARIO
RESULT_ROOT = base.RESULT_ROOT
RUN_SUFFIX = base.RUN_SUFFIX

# 父方法映射：hcc16 方法继承其父基线的 env 契约 / seed 块 / 动作头（连续 vs 混合）。
PARENT = {
    "sac_mlp_hcc16": "sac_mlp",
    "hsac_mlp_base_hcc16": "hsac_mlp_base",
}
METHODS = tuple(PARENT)

# HCC-16：n=16 步回报 + gamma ** 实际视界 bootstrap（与 hold35k 的 v4_7 完全一致）。
N_STEP = 16

DEFAULT_SHORT_STEPS = 20_000
EVAL_EPISODES_TOTAL = base.EVAL_EPISODES_TOTAL


def _env_adapter_hcc16(method: str) -> str:
    """hcc16 方法的环境契约：继承父基线的 adapter（sac_mlp=base，hsac_mlp_base=v4_base）。"""
    parent = PARENT[method]
    if parent == "hsac_mlp_base":
        return "v4_base"
    return "base"  # sac_mlp（连续头，base 契约）


def _replace_replay_buffer_hcc16(model, *, discount: float, action_repeat: int) -> None:
    """把模型刚初始化的空 n-step buffer 换成 horizon-correct 16 步版本。

    照搬 v4_7 的 ``_replace_empty_replay_buffer`` 替换逻辑（hold35k 已验证的接线）：
    只改回报估计契约（n_steps 4→16、bootstrap γ^4→γ^实际视界），不动存储 / 采样 /
    超时语义 / episode 尾重复。
    """
    from algos.sb3_torch.replay_buffer import DictNStepReplayBuffer
    from algos.sb3_torch.replay_buffer_v4_7 import (
        HorizonCorrectDictNStepReplayBufferV47,
    )

    parent = model.replay_buffer
    if not isinstance(parent, DictNStepReplayBuffer):
        raise TypeError("HCC-16 期望父 buffer 为 DictNStepReplayBuffer")
    model.n_steps = N_STEP
    model.replay_buffer_class = HorizonCorrectDictNStepReplayBufferV47
    model.replay_buffer_kwargs = {
        "n_steps": N_STEP,
        "gamma": float(discount),
        "duplicate_episode_end_transition": True,
        "source_action_repeat": int(action_repeat),
    }
    model.replay_buffer = HorizonCorrectDictNStepReplayBufferV47(
        model.buffer_size,
        model.observation_space,
        model.action_space,
        device=model.device,
        n_envs=model.n_envs,
        optimize_memory_usage=model.optimize_memory_usage,
        **model.replay_buffer_kwargs,
    )


def _build_mlp_model_hcc16(method: str, env, *, learning_starts: int):
    """复用父基线的模型构建，再把 n-step 回报换成 HCC-16。"""
    parent = PARENT[method]
    if parent in ("hsac_mlp", "hsac_mlp_base"):
        from algos.hybrid_action.hsac import make_hsac_model

        model = make_hsac_model(
            "hsac_mlp",
            env,
            scenario=SCENARIO,
            learning_rate=base.LEARNING_RATE[parent],
            batch_size=base.BATCH_SIZE,
            discount=base.DISCOUNT,
            learning_starts=learning_starts,
            buffer_size=base.BUFFER_SIZE,
            action_repeat=base.ACTION_REPEAT,
            seed=base.SEED,
            device="cuda",
            verbose=0,
        )
    else:  # sac_mlp（连续头）
        model = base._make_sac_mlp_model(
            env,
            learning_rate=base.LEARNING_RATE[parent],
            batch_size=base.BATCH_SIZE,
            discount=base.DISCOUNT,
            learning_starts=learning_starts,
            buffer_size=base.BUFFER_SIZE,
            action_repeat=base.ACTION_REPEAT,
            seed=base.SEED,
            device="cuda",
        )
    _replace_replay_buffer_hcc16(model, discount=base.DISCOUNT, action_repeat=base.ACTION_REPEAT)
    return model


def run_training_hcc16(method: str, run_dir: Path, smoke: bool, max_steps: int) -> Path:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor

    from algos.sb3_torch.callbacks import RawStepControlCallback

    torch.set_num_threads(1)

    raw_budget = 300 if smoke else max_steps
    frequency = 100 if smoke else base.CHECKPOINT_FREQUENCY
    learning_starts = 60 if smoke else base.LEARNING_STARTS
    overlay_root = run_dir / "overlays"
    namespace = "sm" if smoke else "tr"
    env_adapter = _env_adapter_hcc16(method)

    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = None
    model = None

    base._write_json_atomic(
        run_dir / "status.json",
        dict(status="training", pid=os.getpid(), smoke=smoke, started_at=started),
    )

    try:
        env = Monitor(
            base.make_env_factory(env_adapter, overlay_root / f"ns_{namespace}")(
                base._environment_namespace(), evaluation=False
            ),
            filename=str(run_dir / "train_monitor.csv"),
            info_keywords=(
                "raw_simulation_steps",
                "is_success",
                "collision",
                "off_route",
                "max_time",
            ),
        )
        model = _build_mlp_model_hcc16(method, env, learning_starts=learning_starts)

        base._write_json_atomic(
            run_dir / "arguments.json",
            dict(
                method=method,
                parent=PARENT[method],
                scenario=SCENARIO,
                raw_budget=raw_budget,
                checkpoint_frequency=frequency,
                learning_starts_raw_steps=learning_starts,
                batch_size=base.BATCH_SIZE,
                learning_rate=base.LEARNING_RATE[PARENT[method]],
                buffer_size=base.BUFFER_SIZE,
                discount=base.DISCOUNT,
                action_repeat=base.ACTION_REPEAT,
                seed=base.SEED,
                device="cuda",
                density=base.DENSITY,
                smoke=smoke,
                env_contract=env_adapter,
                reward_shaping=base.REWARD,
                ablation="n_step_4_to_hcc16_horizon_correct",
                n_step=N_STEP,
                note=(
                    f"与父基线 {PARENT[method]} 唯一差异：n-step 回报 4→16，"
                    "bootstrap γ^4→γ^实际视界（HCC-16）。编码器/动作头/熵/调参/环境契约完全一致。"
                ),
            ),
        )

        class Progress(BaseCallback):
            def _on_step(self):
                if self.n_calls % 100 == 0:
                    base._write_json_atomic(
                        run_dir / "progress.json",
                        dict(
                            raw_steps=getattr(self.model, "_raw_steps_seen", None),
                            updates=self.model._n_updates,
                            updated_at=time.time(),
                        ),
                    )
                return True

        model.learn(
            total_timesteps=raw_budget,
            callback=CallbackList(
                [
                    RawStepControlCallback(
                        raw_step_budget=raw_budget,
                        checkpoint_frequency=frequency,
                        checkpoint_path=run_dir / "checkpoints",
                        checkpoint_prefix="ckpt",
                    ),
                    Progress(),
                ]
            ),
        )
        if model._raw_steps_seen != raw_budget:
            raise AssertionError(
                f"raw-step budget mismatch: {model._raw_steps_seen} != {raw_budget}"
            )

        final = run_dir / "final_model.zip"
        model.save(final)
        base._write_json_atomic(
            run_dir / "training_complete.json",
            dict(
                smoke=smoke,
                checkpoint_sha256=base._sha256(final),
                raw_steps=raw_budget,
                updates=model._n_updates,
                replay_size=model.replay_buffer.size(),
                wall_seconds=time.time() - started,
            ),
        )
        base._write_json_atomic(
            run_dir / "status.json",
            dict(status="trained", smoke=smoke, wall_seconds=time.time() - started),
        )
        return final
    except BaseException as exc:
        base._write_json_atomic(
            run_dir / "status.json",
            dict(
                status="failed",
                error=repr(exc),
                smoke=smoke,
                raw_steps=getattr(model, "_raw_steps_seen", None),
                wall_seconds=time.time() - started,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


def _evaluate_saved(method: str, model_path: Path, run_dir: Path, n_ep: int) -> dict:
    """进程内评估（对齐父基线的部署语义：连续头直接 load，混合头 + use_actor_only）。"""
    import torch

    from algos.sb3_torch.evaluation import evaluate_model_detailed
    from algos.sb3_torch.sac import SceneRepresentationSAC

    torch.set_num_threads(1)
    parent = PARENT[method]
    env_adapter = _env_adapter_hcc16(method)
    overlay_root = run_dir / "overlays" / "eval"
    env = base.make_env_factory(env_adapter, overlay_root)(
        base._environment_namespace(), evaluation=True
    )
    loaded = SceneRepresentationSAC.load(str(model_path), env=env, device="cpu", buffer_size=32)
    if parent == "hsac_mlp_base":
        from tools.v48_stability_v4.model import use_actor_only

        model = use_actor_only(loaded)
    else:  # sac_mlp（连续头，标准 SAC 前向，无融合门控）
        model = loaded

    seed_start = base.SEED_START[parent]
    records = []
    try:
        for index in range(n_ep):
            episode_seed = seed_start + index
            np.random.seed(episode_seed + 600_000)
            torch.manual_seed(episode_seed + 600_000)
            env.unwrapped._traffic_episode_index, env.unwrapped._traffic_roll = (
                index,
                None,
            )
            detailed = evaluate_model_detailed(
                model,
                env,
                episodes=1,
                seed=episode_seed,
                deterministic=True,
                sumo_step_seconds=0.1,
                policy_action_hold=1,
            )
            rec = detailed.episode_records[0].to_dict()
            rec["episode"] = index
            records.append(rec)
    finally:
        env.close()

    n = len(records)
    return dict(
        episodes=n,
        success_rate=sum(int(r["success"]) for r in records) / n if n else 0.0,
        collision_rate=sum(int(r["collision"]) for r in records) / n if n else 0.0,
        off_route_rate=sum(int(r["off_route"]) for r in records) / n if n else 0.0,
        timeout_rate=sum(int(r["timeout"]) for r in records) / n if n else 0.0,
        mean_return=float(np.mean([r["episode_return"] for r in records])) if n else None,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_SHORT_STEPS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args(argv)

    if args.eval_only:
        method = args.method or METHODS[0]
        run_dir = RESULT_ROOT / f"{method}__{SCENARIO}{RUN_SUFFIX}"
        model_path = Path(args.model_path) if args.model_path else (run_dir / "final_model.zip")
        if not model_path.is_file():
            raise FileNotFoundError(f"待评估模型不存在：{model_path}")
        summary = _evaluate_saved(method, model_path, run_dir, EVAL_EPISODES_TOTAL)
        print(json.dumps(dict(method=method, summary=summary), ensure_ascii=False, indent=2))
        return 0

    if not args.smoke and not base._cuda_available():
        raise RuntimeError("CUDA unavailable; training has no CPU fallback")

    methods = [args.method] if args.method else list(METHODS)
    for method in methods:
        run_dir = RESULT_ROOT / f"{method}__{SCENARIO}{RUN_SUFFIX}"
        if not args.smoke and run_dir.exists():
            raise RuntimeError(f"output dir already exists (remove first): {run_dir}")

        final = run_training_hcc16(method, run_dir, smoke=args.smoke, max_steps=args.max_steps)
        eval_episodes = 8 if args.smoke else EVAL_EPISODES_TOTAL
        summary = _evaluate_saved(method, final, run_dir, eval_episodes)

        manifest = dict(
            method=method,
            parent=PARENT[method],
            scenario=SCENARIO,
            run_dir=str(run_dir),
            final_model=str(final),
            evaluation=summary,
            smoke=args.smoke,
            raw_training_steps=(300 if args.smoke else args.max_steps),
            note=(
                "D4 样本采样消融：n-step 回报 4→16 + horizon-correct bootstrap（HCC-16），"
                "判断「关键样本 credit 传递」是否让行场景的学习瓶颈。"
            ),
            ablation="n_step_4_to_hcc16_horizon_correct",
        )
        base._write_json_atomic(run_dir / "experiment_manifest.json", manifest)
        print(json.dumps(dict(method=method, summary=summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
