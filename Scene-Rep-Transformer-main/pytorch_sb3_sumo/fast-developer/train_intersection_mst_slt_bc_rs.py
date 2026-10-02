"""P2 微调：加载 BC 预训练的 mst_slt actor + reward shaping 继续 RL 训练。

P2 链路：``collect_yield_demos.py`` 采集「SUMO 默认让行」示范 -> ``bc_pretrain_actor.py``
BC 预训练 actor -> 本脚本加载 BC 模型，叠加 P1 的 RewardShapingWrapper 微调，让 mst_slt
在好的初始化上把「让行时序」最终收敛到 success。

与 ``train_intersection_mst_slt_rs.py``（P1）唯一区别：起点是 BC 预训练模型而非随机
初始化，且 raw_learning_starts 置 0（BC 已暖启动，不再随机探索 5000 步）。

用法::

    # 正式微调（CUDA，5w raw 步，每 200 步 checkpoint）
    python fast-developer/train_intersection_mst_slt_bc_rs.py --bc-model fast-developer/artifacts/mst_slt_bc_pretrained.zip

    # 只评估 BC 预训练模型（不微调），看 BC 本身能否让行
    python fast-developer/train_intersection_mst_slt_bc_rs.py --bc-model fast-developer/artifacts/mst_slt_bc_pretrained.zip --eval-only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import train_intersection_hold35k_mst_fixed as base
import train_intersection_mst_slt_rs as rs  # 复用 shaped_env_factory / 常量

METHOD = "mst_slt"
SCENARIO = base.SCENARIO
RESULT_ROOT = base.RESULT_ROOT
RUN_SUFFIX = "_bc_rs"
RUN_DIR = RESULT_ROOT / f"{METHOD}__{SCENARIO}{RUN_SUFFIX}"


def _monitor_kwargs():
    return dict(
        filename=str(RUN_DIR / "train_monitor.csv"),
        info_keywords=(
            "raw_simulation_steps",
            "is_success",
            "collision",
            "off_route",
            "max_time",
        ),
    )


def run_finetune(bc_model: Path, overlay_root: Path, smoke: bool) -> Path:
    import torch

    torch.set_num_threads(1)

    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.monitor import Monitor

    from algos.sb3_torch.callbacks import (
        BestTrainingSuccessCallback,
        RawStepControlCallback,
    )
    from algos.sb3_torch.sac import SceneRepresentationSAC

    max_steps = 300 if smoke else base.RAW_TRAINING_STEPS
    frequency = 100 if smoke else base.CHECKPOINT_FREQUENCY

    RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
    shaped_factory = rs.shaped_env_factory(overlay_root)
    env = shaped_factory(base._environment_namespace(), evaluation=False)
    monitored_env = Monitor(env, **_monitor_kwargs())

    model = SceneRepresentationSAC.load(str(bc_model), env=monitored_env, device="cuda")
    # BC 已暖启动 actor，微调立即用策略动作并立即训练（跳过 5000 步随机探索）。
    model.raw_learning_starts = 0

    callbacks = CallbackList(
        [
            RawStepControlCallback(
                raw_step_budget=max_steps,
                checkpoint_frequency=frequency,
                checkpoint_path=RUN_DIR / "checkpoints",
                checkpoint_prefix="scene_rep",
            ),
            BestTrainingSuccessCallback(RUN_DIR / "best_training_success_model"),
        ]
    )
    model.learn(total_timesteps=max_steps, callback=callbacks, progress_bar=False)
    final = RUN_DIR / "final_model.zip"
    model.save(str(final))
    monitored_env.close()
    if not final.is_file():
        raise FileNotFoundError(f"finetune did not produce {final}")
    base._write_json_atomic(
        RUN_DIR / "status.json",
        dict(
            status="trained",
            smoke=smoke,
            method=METHOD,
            bc_model=str(bc_model),
            reward_shaping=dict(
                timeout_penalty=rs.TIMEOUT_PENALTY,
                step_cost=rs.STEP_COST,
                progress_scale=rs.PROGRESS_SCALE,
            ),
        ),
    )
    return final


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-model", type=Path, default=_THIS_DIR / "artifacts" / "mst_slt_bc_pretrained.zip")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args(argv)

    if not args.bc_model.is_file():
        raise FileNotFoundError(f"BC 模型不存在: {args.bc_model}")

    if args.eval_only:
        result = base.run_evaluation(METHOD, RUN_DIR, args.bc_model, smoke=args.smoke)
        base.plot_training_curves(METHOD, RUN_DIR)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0

    if not args.smoke and not base._cuda_available():
        raise RuntimeError("CUDA unavailable; training has no CPU fallback")
    if not args.smoke and RUN_DIR.exists():
        raise RuntimeError(f"output dir already exists (remove first): {RUN_DIR}")

    final = run_finetune(args.bc_model, RUN_DIR / "overlays" / "seed_0", smoke=args.smoke)

    result = base.run_evaluation(METHOD, RUN_DIR, final, smoke=args.smoke)
    curve = base.plot_training_curves(METHOD, RUN_DIR)

    manifest = dict(
        method=METHOD,
        scenario=SCENARIO,
        run_dir=str(RUN_DIR),
        final_model=str(final),
        bc_model=str(args.bc_model),
        evaluation=result["summary"],
        training_curve=str(curve),
        smoke=args.smoke,
        reward_shaping=dict(
            timeout_penalty=rs.TIMEOUT_PENALTY,
            step_cost=rs.STEP_COST,
            progress_scale=rs.PROGRESS_SCALE,
        ),
    )
    base._write_json_atomic(RUN_DIR / "experiment_manifest.json", manifest)
    print(json.dumps(dict(method=METHOD, summary=result["summary"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
