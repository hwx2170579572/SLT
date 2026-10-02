"""hold35k attention-aggregation ablation on intersection (yield v2).

Single-variable test of the hold35k collapse hypothesis: replace the vehicle
GCN's Gaussian distance weight (``vehicle_sigma = 20``) with learned attention,
keep everything else identical to hold35k, and check whether the timeout
collapse disappears.

This is a NEW method (``hold35k_attn``) with its own output dir and its own
feature extractor / config files -- no existing file is modified.  Reuses only
the environment factory and file utilities from ``train_intersection_yield_v2``.

Usage::

    python fast-developer/train_intersection_yield_v2_attention.py --smoke
    python fast-developer/train_intersection_yield_v2_attention.py              # 短训练 20000 raw 步
    python fast-developer/train_intersection_yield_v2_attention.py --max-steps 50000
    python fast-developer/train_intersection_yield_v2_attention.py --eval-only \
        --model-path fast-developer/hold35k_attn__intersection_yield_v2/final_model.zip
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

METHOD = "hold35k_attn"
SCENARIO = base.SCENARIO
RESULT_ROOT = base.RESULT_ROOT
RUN_DIR = RESULT_ROOT / f"{METHOD}__{SCENARIO}{base.RUN_SUFFIX}"

# 短训练预算：足以越过 learning_starts(5000) 并积累足够更新观察坍缩与否；
# 正式对比仍应跑满 RAW_TRAINING_STEPS(50000)。
DEFAULT_SHORT_STEPS = 20_000
EVAL_EPISODES_TOTAL = base.EVAL_EPISODES_TOTAL
SEED_START = base.SEED_START["hold35k"]


def _evaluate_saved(model_path: Path, run_dir: Path, n_ep: int) -> dict:
    """进程内评估（对齐 hold35k 的 StabilitySAC + use_actor_only 评估方式）。"""
    import torch

    from algos.sb3_torch.evaluation import evaluate_model_detailed
    from tools.v48_stability_v4.model import StabilitySAC, use_actor_only

    torch.set_num_threads(1)
    overlay_root = run_dir / "overlays" / "eval"
    env = base.make_env_factory("v4_8", overlay_root)(
        base._environment_namespace(), evaluation=True
    )
    loaded = StabilitySAC.load(str(model_path), env=env, device="cpu", buffer_size=32)
    model = use_actor_only(loaded)
    records = []
    try:
        for index in range(n_ep):
            episode_seed = SEED_START + index
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


def run_training_attention(run_dir: Path, smoke: bool, max_steps: int) -> Path:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor

    from algos.sb3_torch.callbacks import RawStepControlCallback
    from configs.sb3_configs_v4_8_attention import make_model_v4_8_attention
    from tools.phase2_model_factory_v1 import verify_optimizer_settings
    from tools.v48_stability_v4.common import CANDIDATES
    from tools.v48_stability_v4.model import StabilitySAC, learning_rate as _lr_at

    torch.set_num_threads(1)

    raw_budget = 300 if smoke else max_steps
    frequency = 100 if smoke else base.CHECKPOINT_FREQUENCY
    learning_starts = 60 if smoke else base.LEARNING_STARTS
    overlay_root = run_dir / "overlays"
    namespace = "sm" if smoke else "tr"

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
            base.make_env_factory("v4_8", overlay_root / f"ns_{namespace}")(
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

        original_config = dict(CANDIDATES[base.HOLD35K_CANDIDATE])
        config = dict(original_config)
        config.update(batch_size=base.BATCH_SIZE)

        model = make_model_v4_8_attention(
            env,
            learning_rate=config["learning_rate"],
            scenario=SCENARIO,
            batch_size=config["batch_size"],
            learning_starts=learning_starts,
            buffer_size=config["buffer_size"],
            action_repeat=base.ACTION_REPEAT,
            seed=base.SEED,
            device="cuda",
            verbose=0,
        )
        model.tau = config["tau"]
        verify_optimizer_settings(
            model, learning_rate=config["learning_rate"], tau=config["tau"]
        )
        model.__class__ = StabilitySAC
        model.stability_config = config
        model.stability_trace_path = str(run_dir / "optimization_trace.jsonl")

        base._write_json_atomic(
            run_dir / "arguments.json",
            dict(
                method=METHOD,
                scenario=SCENARIO,
                candidate=base.HOLD35K_CANDIDATE,
                original_candidate_config=original_config,
                accelerated_config=config,
                raw_budget=raw_budget,
                checkpoint_frequency=frequency,
                learning_starts_raw_steps=learning_starts,
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                tau=config["tau"],
                buffer_size=config["buffer_size"],
                discount=base.DISCOUNT,
                action_repeat=base.ACTION_REPEAT,
                seed=base.SEED,
                device="cuda",
                density=base.DENSITY,
                smoke=smoke,
                env_contract="v4_8",
                env_class="YieldConflictIndependentV2EnvV4V1 (改法 1+2+3)",
                reward_shaping=base.REWARD,
                ablation="vehicle_gcn_distance_weight_to_learned_attention",
                note=(
                    "与 hold35k 唯一差异：TopoTemporalGraphExtractorV2 的 VehicleGraphLayer"
                    "（高斯距离权重 σ=20）换成 VehicleGraphAttentionLayer（可学习 attention）。"
                    "其余（因子化熵/混合头/16步horizon replay/Graph-SLT）完全一致。"
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
        if model._n_updates != raw_budget - learning_starts + 1:
            raise AssertionError(
                f"update-count mismatch: {model._n_updates} != "
                f"{raw_budget - learning_starts + 1}"
            )
        rate = _lr_at(config, raw_budget)
        verify_optimizer_settings(model, learning_rate=rate, tau=config["tau"])
        model.audit_parameters()

        final = run_dir / "final_model.zip"
        model.save(final)
        base._write_json_atomic(
            run_dir / "training_diagnostics.json", model.training_diagnostics()
        )
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_SHORT_STEPS)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args(argv)

    if args.eval_only:
        model_path = Path(args.model_path) if args.model_path else (RUN_DIR / "final_model.zip")
        if not model_path.is_file():
            raise FileNotFoundError(f"待评估模型不存在：{model_path}")
        summary = _evaluate_saved(model_path, RUN_DIR, EVAL_EPISODES_TOTAL)
        print(json.dumps(dict(method=METHOD, summary=summary), ensure_ascii=False, indent=2))
        return 0

    if not args.smoke and not base._cuda_available():
        raise RuntimeError("CUDA unavailable; training has no CPU fallback")
    if not args.smoke and RUN_DIR.exists():
        raise RuntimeError(f"output dir already exists (remove first): {RUN_DIR}")

    final = run_training_attention(RUN_DIR, smoke=args.smoke, max_steps=args.max_steps)

    eval_episodes = 8 if args.smoke else EVAL_EPISODES_TOTAL
    summary = _evaluate_saved(final, RUN_DIR, eval_episodes)

    manifest = dict(
        method=METHOD,
        scenario=SCENARIO,
        run_dir=str(RUN_DIR),
        final_model=str(final),
        evaluation=summary,
        smoke=args.smoke,
        raw_training_steps=(300 if args.smoke else args.max_steps),
        note=(
            "hold35k attention 消融：vehicle GCN 高斯距离权重 → 可学习 attention；"
            "短训练（默认 20000 raw 步）判断是否不再坍缩。"
        ),
        ablation="vehicle_gcn_distance_weight_to_learned_attention",
    )
    base._write_json_atomic(RUN_DIR / "experiment_manifest.json", manifest)
    print(json.dumps(dict(method=METHOD, summary=summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
