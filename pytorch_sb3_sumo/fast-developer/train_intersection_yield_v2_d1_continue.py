"""续训 train_intersection_yield_v2_d1 编排下的方法到目标 raw 步数（默认 100k）。

背景
----
收敛判断（``_conv_checkpoints.json`` 确定性评估）显示：
  * mst_slt 未收敛（35k=20% → 50k=51%，collision 仍 49%），值得延长观察上限；
  * d1 家族（sac_mlp_d1_st / _attn / _rt）与 sac_mlp 基线 50k 已基本收敛；
  * sac_mlp_d1_st_rt 是方法本质负增益，延长步数救不回（但脚本仍支持统一续训）。

本脚本把每个**已训练出 final_model** 的方法，从 final_model 续训到 ``--target-steps``
（默认 100000）raw 步，产物落在 ``<run_dir>/continue_<global_raw_steps>/``，与
``train_intersection_yield_v2.py`` 的 ``run_continue_training`` 约定一致（续训子目录、
continuation_*.json、checkpoint 局部步数、保存前写回全局 ``_raw_steps_seen``）。

覆盖 ``train_intersection_yield_v2_d1.py`` 的 DISPATCH 全部方法：
  * d1 家族（11）：SceneRepresentationSAC / SceneRepresentationSACV2(use_slots=True)
  * yv2 家族（6）：StabilitySAC(hold35k) / SceneRepresentationSAC(其余)
  * legacy 家族（2）：StabilitySAC(hold35k) / SceneRepresentationSAC(mst_slt)
（legacy 在 depart 场景暂无训练数据，运行时会因 final_model 不存在而自然报错。）

load 机制复用 yield_v2 的 ``run_continue_training``：SB3 ``save()`` 不持久化 replay
buffer，续训时 buffer 为空，故把 ``raw_learning_starts`` 设为小窗口（500，smoke 100）
先填满 buffer（>= n_steps/batch）再训练；续训期间 ``_raw_steps_seen`` 为局部计数，
保存前写回全局步数使模型可再次续训。评估统一走 ``d1.evaluate``（正确处理 depart
后缀与 d1 / yv2 / legacy 三分派）。

用法::

    # 单方法续训到 100k（默认 depart4.0）
    python fast-developer/train_intersection_yield_v2_d1_continue.py --method mst_slt
    # 指定目标步数 / 密度
    python fast-developer/train_intersection_yield_v2_d1_continue.py \\
        --method sac_mlp_d1_st --target-steps 80000 --depart-scale 4.0
    # 从中间 checkpoint 续训（默认 <run_dir>/final_model.zip）
    python fast-developer/train_intersection_yield_v2_d1_continue.py \\
        --method sac_mlp_d1_st --model-path <run_dir>/checkpoints/ckpt_raw_35000_steps.zip
    # 全部已训练方法并行续训（2 CUDA worker，跳过无 final_model / 已达标的方法）
    python fast-developer/train_intersection_yield_v2_d1_continue.py --all
    # smoke：额外 300 raw 步验证全链路
    python fast-developer/train_intersection_yield_v2_d1_continue.py --method mst_slt --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_intersection_yield_v2_d1 as d1

DEFAULT_TARGET_STEPS = 100_000
CONTINUE_WORKERS = 2


def _load_spec(global_name: str):
    """返回 (load 类, env adapter, env factory, namespace fn, is_stability)。"""
    from algos.sb3_torch.sac import SceneRepresentationSAC
    from algos.sb3_torch.sac_v2 import SceneRepresentationSACV2

    family, mod, method = d1.DISPATCH[global_name]
    if family == "d1":
        use_slots = bool(d1.D1_CONFIG[method]["use_slots"])
        load_cls = SceneRepresentationSACV2 if use_slots else SceneRepresentationSAC
        return (
            load_cls,
            d1._env_adapter_d1(method),
            d1.base.make_env_factory,
            d1.base._environment_namespace,
            False,
        )
    if family == "legacy":
        is_stability = method == "hold35k"
        load_cls = None if is_stability else SceneRepresentationSAC  # stability 动态 import
        return (
            load_cls,
            "v4_8" if is_stability else "base",
            mod.make_env_factory,
            mod._environment_namespace,
            is_stability,
        )
    # yv2
    is_stability = method == "hold35k"
    load_cls = None if is_stability else SceneRepresentationSAC
    return (
        load_cls,
        d1.base._env_adapter(method),
        d1.base.make_env_factory,
        d1.base._environment_namespace,
        is_stability,
    )


def run_continue(
    global_name: str,
    *,
    target_steps: int,
    depart_scale: float,
    smoke: bool,
    model_path: Path | None = None,
) -> Path:
    """从已保存模型续训到 target_steps raw 步，输出到 <run_dir>/continue_<global>/。"""
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor

    from algos.sb3_torch.callbacks import RawStepControlCallback

    torch.set_num_threads(1)

    family, mod, method = d1.DISPATCH[global_name]
    run_dir = d1._run_dir(global_name, depart_scale)
    model_path = Path(model_path) if model_path else (run_dir / "final_model.zip")
    if not model_path.is_file():
        raise FileNotFoundError(f"待续训模型不存在（先跑正式训练）：{model_path}")

    base_raw_steps = d1.base._read_raw_steps_from_zip(model_path)
    extra_steps = int(target_steps) - base_raw_steps
    if extra_steps <= 0:
        raise ValueError(
            f"{global_name} 已训练 {base_raw_steps} raw 步 >= 目标 {target_steps}，无需续训"
        )
    if smoke:
        extra_steps = min(extra_steps, 300)
    global_raw_steps = base_raw_steps + extra_steps
    cont_dir = run_dir / f"continue_{global_raw_steps}"
    cont_dir.mkdir(parents=True, exist_ok=True)

    load_cls, adapter, env_factory, namespace_fn, is_stability = _load_spec(global_name)
    frequency = 100 if smoke else d1.base.CHECKPOINT_FREQUENCY
    namespace = "sm" if smoke else "ct"

    started = time.time()
    env = None
    model = None

    env = Monitor(
        env_factory(adapter, cont_dir / "overlays" / f"ns_{namespace}")(
            namespace_fn(), evaluation=False
        ),
        filename=str(cont_dir / "train_monitor.csv"),
        info_keywords=(
            "raw_simulation_steps",
            "is_success",
            "collision",
            "off_route",
            "max_time",
        ),
    )

    try:
        if is_stability:
            from tools.v48_stability_v4.model import StabilitySAC

            model = StabilitySAC.load(model_path, env=env, device="cuda")
            # stability_trace_path 不在 save 清单内，续训需重指到新 trace 文件。
            model.stability_trace_path = str(cont_dir / "optimization_trace.jsonl")
        else:
            model = load_cls.load(model_path, env=env, device="cuda")

        if int(model._raw_steps_seen) != base_raw_steps:
            raise AssertionError(
                f"加载后 _raw_steps_seen({model._raw_steps_seen}) 与 zip 记录 "
                f"({base_raw_steps}) 不一致"
            )

        # load 后 replay buffer 为空：设小窗口先填满 buffer（>= n_steps/batch）再训练，
        # 避免「空 buffer 上 train()」崩溃；越小越早收敛到 load 的策略动作。
        model.raw_learning_starts = 100 if smoke else 500

        d1.base._write_json_atomic(
            cont_dir / "status.json",
            dict(
                status="training",
                pid=os.getpid(),
                smoke=smoke,
                base_model=str(model_path),
                base_raw_steps=base_raw_steps,
                extra_raw_steps=extra_steps,
                global_raw_steps=global_raw_steps,
                started_at=started,
            ),
        )
        d1.base._write_json_atomic(
            cont_dir / "arguments.json",
            dict(
                method=global_name,
                family=family,
                source_method=method,
                scenario=d1.NEW_SCENARIO,
                depart_scale=depart_scale,
                base_model=str(model_path),
                base_raw_steps=base_raw_steps,
                extra_raw_steps=extra_steps,
                global_raw_steps=global_raw_steps,
                checkpoint_frequency=frequency,
                smoke=smoke,
                env_contract=adapter,
            ),
        )

        class Progress(BaseCallback):
            def _on_step(self):
                if self.n_calls % 100 == 0:
                    d1.base._write_json_atomic(
                        cont_dir / "progress.json",
                        dict(
                            raw_steps=getattr(self.model, "_raw_steps_seen", None),
                            updates=self.model._n_updates,
                            updated_at=time.time(),
                        ),
                    )
                return True

        model.learn(
            total_timesteps=extra_steps,
            callback=CallbackList(
                [
                    RawStepControlCallback(
                        raw_step_budget=extra_steps,
                        checkpoint_frequency=frequency,
                        checkpoint_path=cont_dir / "checkpoints",
                        checkpoint_prefix="ckpt",
                    ),
                    Progress(),
                ]
            ),
        )
        if model._raw_steps_seen != extra_steps:
            raise AssertionError(
                f"续训 raw-step 不符：{model._raw_steps_seen} != {extra_steps}"
            )

        # 续训期间计数为局部值（extra_steps）；写回全局步数，使保存后的模型可再次续训。
        model._raw_steps_seen = global_raw_steps

        final = cont_dir / "final_model.zip"
        model.save(final)
        d1.base._write_json_atomic(
            cont_dir / "training_complete.json",
            dict(
                smoke=smoke,
                checkpoint_sha256=d1.base._sha256(final),
                base_raw_steps=base_raw_steps,
                extra_raw_steps=extra_steps,
                global_raw_steps=global_raw_steps,
                updates=model._n_updates,
                replay_size=model.replay_buffer.size(),
                wall_seconds=time.time() - started,
            ),
        )
        d1.base._write_json_atomic(
            cont_dir / "status.json",
            dict(status="trained", smoke=smoke, wall_seconds=time.time() - started),
        )

        # 评估统一走 d1.evaluate（正确处理 depart 后缀与 d1 / yv2 / legacy 三分派）。
        result = d1.evaluate(global_name, cont_dir, final, smoke=smoke)

        manifest = dict(
            method=global_name,
            family=family,
            scenario=d1.NEW_SCENARIO,
            depart_scale=depart_scale,
            run_dir=str(run_dir),
            base_model=str(model_path),
            base_raw_steps=base_raw_steps,
            extra_raw_steps=extra_steps,
            global_raw_steps=global_raw_steps,
            continuation_dir=str(cont_dir),
            final_model=str(final),
            evaluation=result["summary"],
            smoke=smoke,
            note=(
                "续训：加载 base_model 权重/优化器/计数器；replay buffer 从空重新累积；"
                "raw_learning_starts 设 500(smoke 100) 先填满 buffer；checkpoint 为局部步数，"
                "保存前把 _raw_steps_seen 写回全局步数。"
            ),
        )
        d1.base._write_json_atomic(
            run_dir / f"continuation_{global_raw_steps}.json", manifest
        )
        print(
            json.dumps(
                dict(method=global_name, summary=result["summary"]),
                ensure_ascii=False,
                indent=2,
            )
        )
        return final
    except BaseException as exc:
        d1.base._write_json_atomic(
            cont_dir / "status.json",
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


def run_all(
    *, target_steps: int, depart_scale: float, smoke: bool, workers: int
) -> Path:
    """并行续训所有已训练（有 final_model）且未达 target 的方法。"""
    if not smoke and not d1.base._cuda_available():
        raise RuntimeError("CUDA unavailable; training has no CPU fallback")

    root = d1.base.RESULT_ROOT
    root.mkdir(parents=True, exist_ok=True)

    todo: list[str] = []
    for name in d1.ALL_METHODS:
        model_path = d1._run_dir(name, depart_scale) / "final_model.zip"
        if not model_path.is_file():
            print(f"[skip] {name}: no final_model", flush=True)
            continue
        base_raw = d1.base._read_raw_steps_from_zip(model_path)
        if base_raw >= target_steps:
            print(f"[skip] {name}: already {base_raw} >= {target_steps}", flush=True)
            continue
        todo.append(name)
    if not todo:
        print("nothing to continue-train", flush=True)
        return root

    script = str(Path(__file__).resolve())

    def run_one(name: str) -> tuple[str, int]:
        cmd = [
            sys.executable,
            script,
            "--method",
            name,
            "--depart-scale",
            str(depart_scale),
            "--target-steps",
            str(target_steps),
        ]
        if smoke:
            cmd.append("--smoke")
        log = root / (
            f"{name}__{d1.NEW_SCENARIO}_depart{d1._depart_token(depart_scale)}"
            "__continue.log"
        )
        with log.open("w", encoding="utf-8") as handle:
            code = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).returncode
        return name, code

    codes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, name): name for name in todo}
        for future in as_completed(futures):
            name, code = future.result()
            codes[name] = code
            print(f"[continue-launcher] {name} exit {code}", flush=True)

    failed = {n: c for n, c in codes.items() if c != 0}
    if failed:
        raise RuntimeError(
            f"continue-train worker(s) failed: {failed}; inspect "
            f"{root}/*{d1.NEW_SCENARIO}_depart*__continue.log"
        )
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=d1.ALL_METHODS)
    parser.add_argument(
        "--all",
        action="store_true",
        help="并行续训所有已训练（有 final_model）且未达 target 的方法",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=DEFAULT_TARGET_STEPS,
        help="续训目标 raw 步数（默认 100k）",
    )
    parser.add_argument(
        "--depart-scale",
        type=float,
        default=d1.DEFAULT_DEPART_SCALE,
        help="发车间隔缩放（默认 4.0，与 d1 编排一致）",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--workers", type=int, default=CONTINUE_WORKERS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    d1._apply_patch(args.depart_scale)
    d1.ensure_sorted_scenario()

    if args.all:
        run_all(
            target_steps=args.target_steps,
            depart_scale=args.depart_scale,
            smoke=args.smoke,
            workers=args.workers,
        )
        return 0

    if args.method is None:
        parser.error("需要 --method 或 --all")
    run_continue(
        args.method,
        target_steps=args.target_steps,
        depart_scale=args.depart_scale,
        smoke=args.smoke,
        model_path=args.model_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
