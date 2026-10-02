"""P4 课程学习：mst_slt + reward shaping，低密度 traffic 起步逐步加密回原始密度。

背景：intersection 原始密度（1 veh/s 左转）下「让行」上界仅 ~22-30%（SUMO Krauss
极限，见 analysis_p1_reward_shaping_result.md 与密度验证 _p4_density_verify.py：
scale=1.0→22-30%、2.0→23%、4.0→70%）。RL 在稀疏三元奖励下无法突破该上界。

本脚本做课程学习：先在 scale=4.0（gap 4-8s，success 不再稀疏）下用 reward shaping
训练 mst_slt 学会让行，再把该模型作为下一档的起点，逐步加密 traffic（4.0→2.0→1.0），
每档继续 RL 微调。最终在原始密度下评估，看课程学到的让行策略能否保持 / 超过 Krauss
上界。这直接回答「mst_slt 能否学会让行」这个核心问题。

实现：不使用 train_sb3.main（无 resume），改用 SceneRepresentationSAC.load +
model.learn 多阶段连续训练；每档用 monkey-patch ``_partitioned_traffic_paths`` 注入
该档的低密度 traffic（新文件放在 fast-developer/_p4_lowdensity_*/，不改现有文件）。

用法::

    # smoke（几分钟验证全链路）
    python fast-developer/train_intersection_mst_slt_curriculum.py --smoke

    # 正式训练（CUDA；4.0→2.0→1.0 各 5w raw 步）
    python fast-developer/train_intersection_mst_slt_curriculum.py

    # 只评估某个已完成档位
    python fast-developer/train_intersection_mst_slt_curriculum.py --eval-only \
        --stage 4.0 --model-path fast-developer/mst_slt__intersection_curriculum/stage_4p0/final_model.zip
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import train_intersection_hold35k_mst_fixed as base
from reward_shaping_wrapper import RewardShapingWrapper

METHOD = "mst_slt"
SCENARIO = base.SCENARIO  # "intersection"
RESULT_ROOT = base.RESULT_ROOT  # fast-developer
RUN_DIR = RESULT_ROOT / f"{METHOD}__{SCENARIO}_curriculum"

# 课程档位（scale=1.0 即原始密度；>1 表示 depart 时间放大、密度下降）。
# 密度-成功率（「不干预」让行）：4.0→70%、3.0→63%、2.0→23%、1.0→22-30%。
# 临界密度在 ~2.5（gap≈ego 穿过时间 2.7s），故在 2.5 附近加密。
CURRICULUM = [4.0, 3.0, 2.5, 2.0, 1.0]
STEPS_PER_STAGE = base.RAW_TRAINING_STEPS  # 每档 5w raw 步

# reward shaping 超参（复用 P1，单一变量）
TIMEOUT_PENALTY = 1.0
STEP_COST = 0.005
PROGRESS_SCALE = 0.01

SOURCE_TRAFFIC = (
    _PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1" / "intersection" / "traffic"
)


def _scale_token(scale: float) -> str:
    return str(scale).replace(".", "p")


def _traffic_paths_for_scale(scale: float) -> tuple[Path, ...]:
    if scale == 1.0:
        return tuple(sorted(SOURCE_TRAFFIC.glob("traffic_*.rou.xml")))
    out_dir = _THIS_DIR / f"_p4_lowdensity_s{_scale_token(scale)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for src in sorted(SOURCE_TRAFFIC.glob("traffic_*.rou.xml")):
        tree = ET.parse(src)
        root = tree.getroot()
        for veh in root.findall("vehicle"):
            depart = float(veh.attrib.get("depart", "0"))
            veh.attrib["depart"] = f"{depart * scale:.3f}"
        dst = out_dir / src.name
        tree.write(dst, encoding="UTF-8", xml_declaration=True)
        paths.append(dst)
    return tuple(paths)


def _curriculum_env_factory(overlay_root: Path, traffic_paths: tuple[Path, ...]):
    """训练环境外包 RewardShapingWrapper，且 traffic 注入当前档的低密度文件。"""
    base_factory = base.make_env_factory("base", overlay_root)

    def factory(args, *, evaluation=False):
        env = base_factory(args, evaluation=evaluation)
        env._partitioned_traffic_paths = lambda spec: traffic_paths
        if evaluation:
            return env
        return RewardShapingWrapper(
            env,
            timeout_penalty=TIMEOUT_PENALTY,
            step_cost=STEP_COST,
            progress_scale=PROGRESS_SCALE,
        )

    return factory


def _monitor_kwargs(stage_dir: Path):
    return dict(
        filename=str(stage_dir / "train_monitor.csv"),
        info_keywords=(
            "raw_simulation_steps",
            "is_success",
            "collision",
            "off_route",
            "max_time",
        ),
    )


def _evaluate(model, env_factory, n_ep: int) -> dict:
    """用当前档 traffic 的原始环境（无 wrapper）评估成功率。"""
    env = env_factory(base._environment_namespace(), evaluation=True)
    succ = coll = timeout = 0
    for ep in range(n_ep):
        env._traffic_episode_index = ep
        env._traffic_roll = None
        obs, info = env.reset(seed=base.SEED_START["mst_slt"] + ep)
        done = False
        steps = 0
        while not done and steps < 250:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        succ += int(bool(info.get("is_success")))
        coll += int(bool(info.get("collision")))
        timeout += int(bool(info.get("max_time")))
    env.close()
    return dict(success=succ, collision=coll, timeout=timeout, n=n_ep)


def run_stage(scale: float, traffic_paths, prev_model, stage_dir: Path, smoke: bool) -> Path:
    import torch

    torch.set_num_threads(1)

    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.monitor import Monitor

    from algos.sb3_torch.callbacks import (
        BestTrainingSuccessCallback,
        RawStepControlCallback,
    )
    from algos.sb3_torch.sac import SceneRepresentationSAC
    from configs.sb3_configs import make_model

    max_steps = 300 if smoke else STEPS_PER_STAGE
    frequency = 100 if smoke else base.CHECKPOINT_FREQUENCY
    eval_episodes = 3 if smoke else 30

    stage_dir.parent.mkdir(parents=True, exist_ok=True)
    overlay_root = stage_dir / "overlays" / "seed_0"
    env_factory = _curriculum_env_factory(overlay_root, traffic_paths)
    shaped_env = env_factory(base._environment_namespace(), evaluation=False)
    monitored_env = Monitor(shaped_env, **_monitor_kwargs(stage_dir))

    if prev_model is None:
        model = make_model(
            "scene_rep",
            monitored_env,
            scenario=SCENARIO,
            learning_rate=base.LEARNING_RATE["mst_slt"],
            batch_size=base.BATCH_SIZE,
            discount=base.DISCOUNT,
            learning_starts=(60 if smoke else base.LEARNING_STARTS),
            buffer_size=base.BUFFER_SIZE,
            action_repeat=base.ACTION_REPEAT,
            seed=base.SEED,
            device="cuda",
            tensorboard_log=None,
            verbose=0,
        )
    else:
        model = SceneRepresentationSAC.load(str(prev_model), env=monitored_env, device="cuda")
        # raw_learning_starts 同时控制「随机探索窗口」与「训练开始」，且 load 后 replay
        # buffer 为空：设 0 会在 buffer 未满 n_steps 时训练而崩溃。设一个小窗口先填满
        # buffer（>= n_steps/batch）再训练；策略侧 raw_learning_starts 越小越早收敛到
        # load 的策略动作。
        model.raw_learning_starts = 100 if smoke else 500

    callbacks = CallbackList(
        [
            RawStepControlCallback(
                raw_step_budget=max_steps,
                checkpoint_frequency=frequency,
                checkpoint_path=stage_dir / "checkpoints",
                checkpoint_prefix="scene_rep",
            ),
            BestTrainingSuccessCallback(stage_dir / "best_training_success_model"),
        ]
    )
    model.learn(total_timesteps=max_steps, callback=callbacks, progress_bar=False)
    final = stage_dir / "final_model.zip"
    model.save(str(final))
    monitored_env.close()

    eval_result = _evaluate(model, env_factory, eval_episodes)
    print(f"  [stage {scale}] eval: success={eval_result['success']}/{eval_episodes} "
          f"collision={eval_result['collision']} timeout={eval_result['timeout']}", flush=True)

    base._write_json_atomic(
        stage_dir / "status.json",
        dict(
            status="trained",
            scale=scale,
            smoke=smoke,
            prev_model=str(prev_model) if prev_model else None,
            final_model=str(final),
            evaluation=eval_result,
            reward_shaping=dict(
                timeout_penalty=TIMEOUT_PENALTY,
                step_cost=STEP_COST,
                progress_scale=PROGRESS_SCALE,
            ),
        ),
    )
    return final


def run_curriculum(smoke: bool) -> Path:
    prev_model = None
    final = None
    for scale in CURRICULUM:
        token = _scale_token(scale)
        stage_dir = RUN_DIR / f"stage_{token}"
        traffic_paths = _traffic_paths_for_scale(scale)
        print(f"== 课程阶段 scale={scale}（{len(traffic_paths)} traffic）-> {stage_dir}", flush=True)
        final = run_stage(scale, traffic_paths, prev_model, stage_dir, smoke)
        prev_model = final
    return final


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--stage", type=float)
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args(argv)

    if args.eval_only:
        from algos.sb3_torch.sac import SceneRepresentationSAC

        if args.stage is None:
            raise SystemExit("--eval-only 需要 --stage")
        token = _scale_token(args.stage)
        stage_dir = RUN_DIR / f"stage_{token}"
        model_path = args.model_path or (stage_dir / "final_model.zip")
        traffic_paths = _traffic_paths_for_scale(args.stage)
        env_factory = _curriculum_env_factory(stage_dir / "overlays" / "eval", traffic_paths)
        env = env_factory(base._environment_namespace(), evaluation=True)
        model = SceneRepresentationSAC.load(str(model_path), env=env, device="cuda")
        result = _evaluate(model, env_factory, 30)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if not args.smoke and not base._cuda_available():
        raise RuntimeError("CUDA unavailable; training has no CPU fallback")
    if not args.smoke and RUN_DIR.exists():
        raise RuntimeError(f"output dir already exists (remove first): {RUN_DIR}")

    final = run_curriculum(smoke=args.smoke)

    manifest = dict(
        method=METHOD,
        scenario=SCENARIO,
        run_dir=str(RUN_DIR),
        final_model=str(final),
        curriculum=[float(s) for s in CURRICULUM],
        steps_per_stage=STEPS_PER_STAGE,
        smoke=args.smoke,
        reward_shaping=dict(
            timeout_penalty=TIMEOUT_PENALTY,
            step_cost=STEP_COST,
            progress_scale=PROGRESS_SCALE,
        ),
    )
    base._write_json_atomic(RUN_DIR / "experiment_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
