"""mst_slt + reward shaping 在 intersection 上的训练（P1 根因修复）。

背景：修复超参后的 mst_slt 仍坍缩为「停车等超时」零回报吸收盆地（成功率 1%），
根因见 ``fast-developer/analysis_mst_slt_fix_failure_rootcause_v2.md``——环境原始
reward 为稀疏三元 ``raw_reward = float(success) - float(collision)``，timeout=0 使
「停车等到超时」成为零风险吸收策略，严格支配早期价值为负的抢行策略。

本脚本只训练 **mst_slt**（连续动作头，修复超参 lr=1e-4 / batch=32 /
learning_starts=5000），并在环境层包裹 ``RewardShapingWrapper``（见
``reward_shaping_wrapper.py``）改写 reward 三处：

  (a) timeout -> -1（与 collision 同罚），消除「0 > -1」的支配；
  (b) 稠密进度奖励（沿 route 累计行驶距离增量）；
  (c) 小步长生活成本（-step_cost/决策步），抑制原地停留。

不改 observation / termination / truncated / info，因此评估（读
info["is_success"] / info["collision"] / info["max_time"]）不受影响。

评估复用 ``train_intersection_hold35k_mst_fixed.py`` 的 ``run_evaluation`` /
``plot_training_curves``：其 eval-worker 通过 ``__file__`` 派发到 fixed 脚本，
用**无 wrapper** 的原始环境评估，保证 success/collision/timeout 判定与 _fixed 版
同源可比。结果目录后缀 ``_rs``，与 ``_fixed`` / 加速版区分。

用法::

    # 正式训练（CUDA，5w raw 步，每 200 步 checkpoint，6-worker 评估 100 回合）
    python fast-developer/train_intersection_mst_slt_rs.py

    # 快速 smoke（几分钟内验证全链路）
    python fast-developer/train_intersection_mst_slt_rs.py --smoke
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

# 复用修复版脚本的常量与评估/绘图/文件工具（其 run_evaluation 的 eval-worker
# 通过 __file__ 派发到该脚本，评估环境无 shaping wrapper）。
import train_intersection_hold35k_mst_fixed as base

from reward_shaping_wrapper import RewardShapingWrapper

METHOD = "mst_slt"
SCENARIO = base.SCENARIO  # "intersection"
RESULT_ROOT = base.RESULT_ROOT  # fast-developer
RUN_SUFFIX = "_rs"
RUN_DIR = RESULT_ROOT / f"{METHOD}__{SCENARIO}{RUN_SUFFIX}"

# --------------------------------------------------------------------------- #
# reward shaping 超参（P1；单一变量，其余训练超参与 _fixed 版完全一致）
# --------------------------------------------------------------------------- #
TIMEOUT_PENALTY = 1.0      # (a) timeout -> -1，与 collision 同罚
STEP_COST = 0.005          # (c) 每决策步生活成本
PROGRESS_SCALE = 0.01      # (b) 进度奖励系数（元/米，全程约 +0.7~+1.4）


def shaped_env_factory(overlay_root: Path):
    """返回 ``make_environment(args, *, evaluation=False)``，训练环境外包 RewardShapingWrapper。

    shaping 仅在训练侧（evaluation=False）生效：train_sb3 训练结束时的最终评估
    （``make_injected_evaluation_env`` -> ``env_factory(args, evaluation=True)``）
    需要 ``PaperSumoSceneEnv`` 类型以满足 paper evaluation contract，故 evaluation
    环境不包 wrapper（其 success/collision 判定读 info，本就不受 reward 影响）。
    """
    base_factory = base.make_env_factory("base", overlay_root)

    def factory(args, *, evaluation=False):
        env = base_factory(args, evaluation=evaluation)
        if evaluation:
            return env
        return RewardShapingWrapper(
            env,
            timeout_penalty=TIMEOUT_PENALTY,
            step_cost=STEP_COST,
            progress_scale=PROGRESS_SCALE,
        )

    return factory


def run_training(overlay_root: Path, smoke: bool) -> Path:
    import torch

    torch.set_num_threads(1)

    from tools import train_sb3

    max_steps = 300 if smoke else base.RAW_TRAINING_STEPS
    learning_starts = 60 if smoke else base.LEARNING_STARTS
    frequency = 100 if smoke else base.CHECKPOINT_FREQUENCY
    eval_episodes = 1 if smoke else 100

    RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
    env_factory = shaped_env_factory(overlay_root)

    argv = [
        "--algo", "scene_rep",
        "--scenario", SCENARIO,
        "--max-steps", str(max_steps),
        "--learning-starts", str(learning_starts),
        "--checkpoint-freq", str(frequency),
        "--eval-freq", "0",
        "--eval-episodes", str(eval_episodes),
        "--seed", str(base.SEED),
        "--device", "cuda",
        "--batch-size", str(base.BATCH_SIZE),
        "--learning-rate", str(base.LEARNING_RATE["mst_slt"]),
        "--discount", str(base.DISCOUNT),
        "--buffer-size", str(base.BUFFER_SIZE),
        "--action-repeat", str(base.ACTION_REPEAT),
        "--output-dir", str(RUN_DIR.parent.resolve()),
        "--model-name", RUN_DIR.name,
        "--ego-control-profile", "direct",
        "--episode-limit-profile", "source",
        "--traffic-protocol", "frozen_80_20",
    ]
    train_sb3.main(
        argv,
        env_factory=env_factory,
        default_output_dir=RUN_DIR.parent,
        require_paper_evaluation_contract=True,
        tensorboard_log_root=RESULT_ROOT / "tb",
    )
    final = RUN_DIR / "final_model.zip"
    if not final.is_file():
        raise FileNotFoundError(f"train_sb3 did not produce {final}")
    base._write_json_atomic(
        RUN_DIR / "status.json",
        dict(
            status="trained",
            smoke=smoke,
            method=METHOD,
            reward_shaping=dict(
                timeout_penalty=TIMEOUT_PENALTY,
                step_cost=STEP_COST,
                progress_scale=PROGRESS_SCALE,
            ),
        ),
    )
    return final


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="缩小规模快速验证全链路")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    if args.eval_only:
        run_dir = Path(args.output_dir) if args.output_dir else RUN_DIR
        model = Path(args.model_path) if args.model_path else (RUN_DIR / "final_model.zip")
        result = base.run_evaluation(METHOD, run_dir, model, smoke=args.smoke)
        base.plot_training_curves(METHOD, run_dir)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0

    if not args.smoke and not base._cuda_available():
        raise RuntimeError("CUDA unavailable; training has no CPU fallback")

    if not args.smoke and RUN_DIR.exists():
        raise RuntimeError(f"output dir already exists (remove first): {RUN_DIR}")

    final = run_training(RUN_DIR / "overlays" / "seed_0", smoke=args.smoke)

    result = base.run_evaluation(METHOD, RUN_DIR, final, smoke=args.smoke)
    curve = base.plot_training_curves(METHOD, RUN_DIR)

    manifest = dict(
        method=METHOD,
        scenario=SCENARIO,
        run_dir=str(RUN_DIR),
        final_model=str(final),
        evaluation=result["summary"],
        training_curve=str(curve),
        smoke=args.smoke,
        reward_shaping=dict(
            timeout_penalty=TIMEOUT_PENALTY,
            step_cost=STEP_COST,
            progress_scale=PROGRESS_SCALE,
        ),
        raw_training_steps=base.RAW_TRAINING_STEPS,
        checkpoint_frequency=base.CHECKPOINT_FREQUENCY,
        hyperparams=dict(
            batch_size=base.BATCH_SIZE,
            learning_rate=base.LEARNING_RATE["mst_slt"],
            learning_starts=base.LEARNING_STARTS,
        ),
    )
    base._write_json_atomic(RUN_DIR / "experiment_manifest.json", manifest)
    print(
        json.dumps(
            dict(method=METHOD, summary=result["summary"]),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
