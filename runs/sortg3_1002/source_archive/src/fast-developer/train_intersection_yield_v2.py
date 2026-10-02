"""hold35k + MST+SLT 在 intersection 上的「obs 修复 + reward shaping」训练脚本（yield v2）。

相对同目录下 ``train_intersection_hold35k_mst_fixed.py``（修复超参版）的差异，全部在
环境层，不改任何项目文件：

  * hold35k（v4_8 混合动作头）：``YieldConflictIndependentV2EnvV4V1``
    = 改法 1+2（ego map 穿过 internal 弧线、从 ego 车道出发）+ 改法 3（neighbor
    冲突相关性选车，TTC 排序）。见 ``yield_obs_env.py`` / ``yield_conflict_env.py``。
  * mst_slt（base 连续动作头）：``YieldObsIndependentV2EnvV1`` = 仅改法 1+2，
    neighbor 保持父类纯欧氏距离（即「改法 3 只对 hold35k 生效」的消融对照）。
  * 两个方法都包 ``GeneralizedRewardShapingWrapper``（可泛化 reward shaping v2），
    见 ``reward_shaping_v2.py``：success +10 / collision -10 / off_route -10 /
    timeout -5 + progress 0.02/米 + step_cost 0.01，打破「停车等超时=0」的吸收盆地。

训练规模与超参沿用 fixed 版（5w raw 步、每 200 步存一次、batch 32、lr hold35k=5e-5
深地板 / mst_slt=1e-4、learning_starts=5000、action_repeat=3），仅环境不同。

用法::

    python fast-developer/train_intersection_yield_v2.py --smoke   # 快速全链路 smoke
    python fast-developer/train_intersection_yield_v2.py           # 正式训练 + 评估 + 绘图

    # 从已保存模型续训（--extra-steps 为续训 raw 步数，可任意指定）：
    python fast-developer/train_intersection_yield_v2.py continue-train \
        --method sac_mlp --extra-steps 50000
    python fast-developer/train_intersection_yield_v2.py continue-train \
        --method hsac_mlp_base --extra-steps 50000
    # 指定从哪个 checkpoint/中间模型续训（默认 <run_dir>/final_model.zip）：
    python fast-developer/train_intersection_yield_v2.py continue-train \
        --method sac_mlp --extra-steps 50000 \
        --model-path fast-developer/sac_mlp__intersection_yield_v2/final_model.zip
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# 实验配置
# --------------------------------------------------------------------------- #
SCENARIO = "intersection"
METHODS = ("hold35k", "mst_slt", "hsac_mlp", "sac_mlp", "sac_mlp_v48", "hsac_mlp_base")

RAW_TRAINING_STEPS = 50_000
CHECKPOINT_FREQUENCY = 200
LEARNING_STARTS = 5_000
BATCH_SIZE = 32
LEARNING_RATE = {
    "hold35k": 5e-5,
    "mst_slt": 1e-4,
    "hsac_mlp": 1e-4,
    "sac_mlp": 1e-4,
    "sac_mlp_v48": 1e-4,
    "hsac_mlp_base": 1e-4,
}
BUFFER_SIZE = 20_000
DISCOUNT = 0.99
ACTION_REPEAT = 3
SEED = 0

EVAL_EPISODES_TOTAL = 100
EVAL_WORKERS = 6
TRAIN_WORKERS = 2
SEED_START = {
    "hold35k": 420_000,
    "mst_slt": 10_000,
    "hsac_mlp": 420_000,
    "sac_mlp": 10_000,
    # sac_mlp_v48 对齐 hsac_mlp（都 v4_8 契约），保证「动作头维度」hsac_mlp vs sac_mlp_v48
    # 在同一 SUMO seed / traffic 变体序列上评估，消除评估侧噪音。
    "sac_mlp_v48": 420_000,
    # hsac_mlp_base 对齐 sac_mlp（都 base 契约，改法 1+2 欧氏选车），保证 base 契约下的
    # 「动作头维度」sac_mlp vs hsac_mlp_base 在同一 seed / traffic 序列上评估。
    "hsac_mlp_base": 10_000,
}

HOLD35K_CANDIDATE = "tau0025_floor2e5_hold35k"

RESULT_ROOT = PROJECT_ROOT / "fast-developer"

# yield v2 版 run 目录后缀：与 fixed 版（``_fixed``）区分，避免覆盖。
RUN_SUFFIX = "_yield_v2"

# 本实验用 depart×scale 降密度（见 DEPART_SCALE），不用 vehicle_scale 加密，故保持 1.0
#（no-op：build_high_density_overlay 生成空 overlay，跑基础 traffic 文件本身）。
DENSITY = {
    "vehicle_scale": 1.0,
    "pedestrian_scale": 1.0,
    "clone_depart_jitter_seconds": (0.0, 0.0),
}

# depart×scale：用户指定 2.0（发车间隔放大 2 倍 = 密度减半）。密度-成功率映射
#（_p4_density_verify.py，不干预让行）：4.0→70%、3.0→63%、2.0→23%、1.0→22-30%。
# 把原始 traffic 每个 vehicle 的 depart 时间 ×scale，生成低密度 traffic 变体，经
# monkey-patch `_partitioned_traffic_paths` 注入环境（训练与评估同密度）。设为 None
# 或 1.0 表示用原始密度 traffic。
DEPART_SCALE = 2.0
EVAL_TRAFFIC_SPLIT = "validation"

SOURCE_TRAFFIC = (
    PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1" / "intersection" / "traffic"
)


def _traffic_paths_for_scale(scale: float) -> tuple[Path, ...]:
    """生成 depart×scale 的低密度 traffic 变体（发车间隔放大），返回文件路径元组。

    scale==1.0 直接返回原始 traffic；否则把每个 vehicle 的 depart 时间 ×scale 写到
    ``_p4_lowdensity_s{scale}/``（与 _p4_density_verify.py 同机制，不改项目文件）。
    """
    import xml.etree.ElementTree as ET

    if scale == 1.0:
        return tuple(sorted(SOURCE_TRAFFIC.glob("traffic_*.rou.xml")))
    token = str(scale).replace(".", "p")
    # 输出目录按源场景名区分：不同脚本复用本函数并覆盖 SOURCE_TRAFFIC（如
    # intersection_sorted）时各自写独立目录，避免互相覆盖同名 traffic 文件导致
    # source_sha256 漂移。
    source_token = SOURCE_TRAFFIC.parent.name
    out_dir = RESULT_ROOT / f"_p4_lowdensity_s{token}__{source_token}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for src in sorted(SOURCE_TRAFFIC.glob("traffic_*.rou.xml")):
        dst = out_dir / src.name
        # 幂等：已生成则直接复用。训练阶段已生成一次，评估阶段的多个 eval-worker
        # 子进程会并发调用本函数；若每个都无条件 os.replace，已加载该文件的 SUMO
        # 实例会锁住目标文件，导致 Windows 下 os.replace 报 WinError 5。
        if dst.is_file():
            paths.append(dst)
            continue
        tree = ET.parse(src)
        root = tree.getroot()
        for veh in root.findall("vehicle"):
            depart = float(veh.attrib.get("depart", "0"))
            veh.attrib["depart"] = f"{depart * scale:.3f}"
        # 原子写（临时文件 + os.replace），避免并发写时读到中间状态。
        tmp = dst.with_name(f".t{os.getpid()}.tmp")
        tree.write(tmp, encoding="UTF-8", xml_declaration=True)
        os.replace(tmp, dst)
        paths.append(dst)
    return tuple(paths)

# 可泛化 reward shaping v2 的数值（与 reward_shaping_v2.py 默认值一致，集中在此便于审查）。
REWARD = dict(
    success_reward=10.0,
    collision_reward=-10.0,
    off_route_reward=-10.0,
    timeout_reward=-5.0,
    progress_scale=0.02,
    step_cost=0.01,
)


# --------------------------------------------------------------------------- #
# 环境工厂：注入自定义环境类（obs 修复）+ reward shaping wrapper
# --------------------------------------------------------------------------- #
def _environment_namespace() -> argparse.Namespace:
    return argparse.Namespace(
        scenario=SCENARIO,
        history_steps=10,
        neighbors=5,
        path_length=10,
        action_repeat=ACTION_REPEAT,
        discount=DISCOUNT,
        ego_control_profile="direct",
        episode_limit_profile="source",
        gui=False,
        # This is the existing high-density contract partition. Random-flow
        # train/validation/test namespaces are set separately below.
        evaluation_split="validation",
    )


def _configure_cli_scene(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Apply the optional scene/profile flags without changing legacy defaults."""
    from envs.sumo.random_intersection import (
        ASSET_ROOT,
        RANDOM_INTERSECTION_SCENARIOS,
        ensure_random_intersection_assets,
        is_random_intersection_scenario,
    )

    global SCENARIO, SOURCE_TRAFFIC, DEPART_SCALE, EVAL_TRAFFIC_SPLIT
    scenario = str(args.scenario or SCENARIO)
    random_profile = is_random_intersection_scenario(scenario)
    if args.depart_scale is None:
        scale = 1.0 if random_profile else DEPART_SCALE
    else:
        scale = float(args.depart_scale)
    if random_profile:
        if scale != 1.0:
            parser.error("random intersection flows require --depart-scale 1.0")
        ensure_random_intersection_assets(scenario)
        SOURCE_TRAFFIC = ASSET_ROOT / scenario / "traffic"
    elif scenario == "intersection":
        if args.eval_traffic_split != "validation":
            parser.error("--eval-traffic-split applies only to random intersection profiles")
        SOURCE_TRAFFIC = (
            PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1"
            / "intersection" / "traffic"
        )
    else:
        parser.error(f"unsupported base trainer scenario: {scenario}")
    SCENARIO = scenario
    DEPART_SCALE = scale
    EVAL_TRAFFIC_SPLIT = str(args.eval_traffic_split)


def make_env_factory(adapter: str, overlay_root: Path):
    """返回 ``make_environment(args, *, evaluation=False)``。

    ``adapter`` 三选一（都包 ``GeneralizedRewardShapingWrapper``）：
    - "v4_8"：改法 1+2+3（TTC 冲突选车），v4 观察契约（含 lane_action_mask）
    - "base"：改法 1+2（欧氏选车），连续头观察契约（无 lane_action_mask）
    - "v4_base"：改法 1+2（欧氏选车），v4 观察契约（含 lane_action_mask）——供混合头在
      base 契约语义下训练（混合头需要 lane_action_mask 做离散车道 mask，故用 v4 观察
      契约类 ``YieldObsIndependentV2EnvV4V1``，但走 baseline 分支拿 evaluation split，
      与 sac_mlp 的 base 契约同 traffic 变体）。
    """
    from reward_shaping_v2 import GeneralizedRewardShapingWrapper
    from envs.sumo.random_intersection import is_random_intersection_scenario
    from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory
    from yield_conflict_env import YieldConflictIndependentV2EnvV4V1
    from yield_obs_env import YieldObsIndependentV2EnvV1, YieldObsIndependentV2EnvV4V1

    if adapter == "v4_8":
        inner_adapter = "v4_8"
        baseline_class = YieldObsIndependentV2EnvV1
        v4_class = YieldConflictIndependentV2EnvV4V1
    elif adapter == "v4_base":
        inner_adapter = "base"
        baseline_class = YieldObsIndependentV2EnvV4V1
        v4_class = YieldConflictIndependentV2EnvV4V1
    else:  # "base"
        inner_adapter = "base"
        baseline_class = YieldObsIndependentV2EnvV1
        v4_class = YieldConflictIndependentV2EnvV4V1

    base_factory = _make_environment_factory(
        adapter=inner_adapter,
        density=DENSITY,
        overlay_root=overlay_root,
        baseline_environment_class=baseline_class,
        v4_environment_class=v4_class,
    )

    # depart×scale 低密度 traffic（DEPART_SCALE=None/1.0 时用原始密度，不注入）。
    # 训练与评估共用同一 traffic 变体，保证一致性（对应同一密度档）。
    traffic_paths = (
        None
        if DEPART_SCALE is None or DEPART_SCALE == 1.0
        else _traffic_paths_for_scale(DEPART_SCALE)
    )

    def make_environment(args, *, evaluation=False):
        env = base_factory(args, evaluation=evaluation)
        if is_random_intersection_scenario(SCENARIO):
            traffic_split = EVAL_TRAFFIC_SPLIT if evaluation else "train"
            setter = getattr(env, "set_traffic_split", None)
            if not callable(setter):
                raise TypeError(
                    "Random intersection environment must expose set_traffic_split()"
                )
            setter(traffic_split)
        elif traffic_paths is not None:
            # 必须在包 wrapper 前 patch 底层 env 的实例方法（gym.Wrapper 1.2.3 无
            # __getattr__）。_sumo_command 经 _selected_traffic_path 读它，配合
            # vehicle_scale=1.0（空 overlay）即以 depart×scale 低密度跑。
            env._partitioned_traffic_paths = lambda spec: traffic_paths
        return GeneralizedRewardShapingWrapper(
            env,
            success_reward=REWARD["success_reward"],
            collision_reward=REWARD["collision_reward"],
            off_route_reward=REWARD["off_route_reward"],
            timeout_reward=REWARD["timeout_reward"],
            progress_scale=REWARD["progress_scale"],
            step_cost=REWARD["step_cost"],
        )

    return make_environment


def _env_adapter(method: str) -> str:
    """各方法在 yield_v2 训练/评估中使用的环境契约（env class + 改法 3 是否生效）。

    与 ``run_training_mlp`` / ``run_eval_worker`` 内的分支保持一致：
    - "v4_8"：改法 1+2+3（TTC 冲突选车），v4 观察契约
    - "v4_base"：改法 1+2（欧氏选车），v4 观察契约（混合头需要 lane_action_mask）
    - "base"：改法 1+2（欧氏选车），连续头观察契约
    """
    if method in ("hsac_mlp", "sac_mlp_v48", "hold35k"):
        return "v4_8"
    if method == "hsac_mlp_base":
        return "v4_base"
    return "base"  # sac_mlp / mst_slt


# --------------------------------------------------------------------------- #
# 通用文件工具
# --------------------------------------------------------------------------- #
def _write_json_atomic(path: Path, payload) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    retry_delays = (0.05, 0.10, 0.20)
    for attempt in range(len(retry_delays) + 1):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError as exc:
            if attempt < len(retry_delays):
                time.sleep(retry_delays[attempt])
                continue
            if path.name == "progress.json":
                logging.getLogger(__name__).warning(
                    "Could not publish progress telemetry after %d attempts; "
                    "keeping the existing file and complete temporary snapshot at %s: %s",
                    len(retry_delays) + 1,
                    tmp,
                    exc,
                )
                return False
            raise


def _guard_random_evaluation_output(
    run_dir: Path, *, scenario: str, traffic_split: str
) -> None:
    """Prevent a randomized-split eval from replacing evidence from another split."""
    from envs.sumo.random_intersection import is_random_intersection_scenario

    if not is_random_intersection_scenario(scenario):
        return
    result_path = Path(run_dir) / "evaluation_results.json"
    if not result_path.is_file():
        return
    try:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Refusing to overwrite unreadable randomized evaluation results: {result_path}"
        ) from exc
    identity = existing.get("identity") or {}
    prior_scenario = identity.get("scenario")
    prior_split = identity.get("eval_traffic_split")
    if prior_scenario != scenario or prior_split != traffic_split:
        raise FileExistsError(
            "Randomized evaluation output already contains different or unlabelled "
            f"scenario/split evidence ({prior_scenario!r}/{prior_split!r}); "
            f"write {scenario!r}/{traffic_split!r} to a separate --output-dir."
        )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _behavior_diagnostics_metadata(method: str, phase: str, run_dir: Path, **extra) -> dict:
    """Return provenance for the exact traffic pool exposed to this environment."""
    from envs.sumo.random_intersection import (
        get_random_intersection_config,
        is_random_intersection_scenario,
    )

    if DEPART_SCALE is None or DEPART_SCALE == 1.0:
        traffic_paths = tuple(sorted(SOURCE_TRAFFIC.glob("traffic_*.rou.xml")))
    else:
        traffic_paths = _traffic_paths_for_scale(DEPART_SCALE)
    files = [
        {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
        for path in traffic_paths
    ]
    paths = [item["path"] for item in files]
    metadata = {
        "source_script": Path(__file__).name,
        "method": method,
        "phase": phase,
        "run_dir": str(Path(run_dir).resolve()),
        "scenario": SCENARIO,
        "depart_scale": DEPART_SCALE,
        "source_traffic_dir": str(SOURCE_TRAFFIC.resolve()),
        "effective_traffic_files": files,
        "train_traffic_files": paths,
        "eval_traffic_files": paths,
        "traffic_partition": "same_complete_effective_pool_for_train_and_eval; no holdout",
        "traffic_variants_shared_between_train_and_eval": True,
        "training_seed": SEED,
        "action_repeat": ACTION_REPEAT,
        "reward_shaping": REWARD,
    }
    if is_random_intersection_scenario(SCENARIO):
        traffic_config = get_random_intersection_config(SCENARIO)
        traffic_split = "train" if phase == "train" else EVAL_TRAFFIC_SPLIT
        metadata.update(
            {
                "traffic_protocol": traffic_config["protocol"],
                "traffic_config": traffic_config,
                "traffic_split": traffic_split,
                "traffic_seed_domain": traffic_config["traffic_seed_domains"][
                    traffic_split
                ],
                "traffic_partition": "disjoint_simulation_seed_domains; shared_static_flow_specification",
                "traffic_variants_shared_between_train_and_eval": False,
                "traffic_specification_shared_between_train_and_eval": True,
                "train_traffic_files": paths,
                "eval_traffic_files": paths,
            }
        )
    metadata.update(extra)
    return metadata


def _wrap_behavior_diagnostics_env(env, run_dir: Path, phase: str, method: str, **metadata):
    from behavior_diagnostics import get_behavior_recorder, wrap_behavior_diagnostics

    wrapped = wrap_behavior_diagnostics(
        env,
        output_dir=run_dir,
        phase=phase,
        method=method,
        metadata=_behavior_diagnostics_metadata(method, phase, run_dir, **metadata),
    )
    return wrapped, get_behavior_recorder(wrapped)


def _make_behavior_optimization_callback(recorder, raw_interval: int = 1000):
    """Sample the latest scalar learner metrics without duplicating policy records."""
    from stable_baselines3.common.callbacks import BaseCallback

    class BehaviorOptimizationCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.last_raw_steps = 0

        def _record(self):
            raw_steps = int(getattr(self.model, "_raw_steps_seen", 0))
            decision_steps = int(getattr(self.model, "num_timesteps", 0))
            updates = int(getattr(self.model, "_n_updates", 0))
            metrics = dict(getattr(getattr(self.model, "logger", None), "name_to_value", {}))
            recorder.record_optimization(raw_steps, decision_steps, updates, metrics)
            self.last_raw_steps = raw_steps

        def _on_step(self):
            raw_steps = int(getattr(self.model, "_raw_steps_seen", 0))
            if raw_steps - self.last_raw_steps >= raw_interval:
                self._record()
            return True

        def _on_training_end(self):
            if int(getattr(self.model, "_raw_steps_seen", 0)) > self.last_raw_steps:
                self._record()

    return BehaviorOptimizationCallback()


# --------------------------------------------------------------------------- #
# 训练：hold35k（TASAC, v4_8）
# --------------------------------------------------------------------------- #
def run_training_hold35k(run_dir: Path, smoke: bool, max_steps: int | None = None) -> Path:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor

    from algos.sb3_torch.callbacks import (
        BestTrainingSuccessCallback,
        RawStepControlCallback,
        RewardBranchProgressCallback,
    )
    from reward_shaping_v2 import REWARD_BRANCH_KEYS
    from tools.phase2_model_factory_v1 import make_phase2_model, verify_optimizer_settings
    from tools.v48_stability_v4.common import CANDIDATES
    from tools.v48_stability_v4.model import StabilitySAC, learning_rate as _lr_at

    torch.set_num_threads(1)

    if max_steps is None:
        max_steps = RAW_TRAINING_STEPS
    raw_budget = 300 if smoke else max_steps
    frequency = 100 if smoke else CHECKPOINT_FREQUENCY
    learning_starts = 60 if smoke else LEARNING_STARTS
    overlay_root = run_dir / "overlays"
    namespace = "sm" if smoke else "tr"

    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = None
    model = None

    _write_json_atomic(
        run_dir / "status.json",
        dict(status="training", pid=os.getpid(), smoke=smoke, started_at=started),
    )

    try:
        env = Monitor(
            make_env_factory("v4_8", overlay_root / f"ns_{namespace}")(
                _environment_namespace(), evaluation=False
            ),
            filename=str(run_dir / "train_monitor.csv"),
            info_keywords=(
                "raw_simulation_steps",
                "is_success",
                "collision",
                "off_route",
                "max_time",
            )
            + REWARD_BRANCH_KEYS,
        )

        original_config = dict(CANDIDATES[HOLD35K_CANDIDATE])
        config = dict(original_config)
        config.update(batch_size=BATCH_SIZE)
        model = make_phase2_model(
            "v4_8",
            env,
            learning_rate=config["learning_rate"],
            tau=config["tau"],
            scenario=SCENARIO,
            batch_size=config["batch_size"],
            learning_starts=learning_starts,
            buffer_size=config["buffer_size"],
            action_repeat=ACTION_REPEAT,
            seed=SEED,
            device="cuda",
            verbose=0,
        )
        model.__class__ = StabilitySAC
        model.stability_config = config
        model.stability_trace_path = str(run_dir / "optimization_trace.jsonl")

        _write_json_atomic(
            run_dir / "arguments.json",
            dict(
                method="hold35k",
                scenario=SCENARIO,
                candidate=HOLD35K_CANDIDATE,
                original_candidate_config=original_config,
                accelerated_config=config,
                raw_budget=raw_budget,
                checkpoint_frequency=frequency,
                learning_starts_raw_steps=learning_starts,
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                tau=config["tau"],
                buffer_size=config["buffer_size"],
                discount=DISCOUNT,
                action_repeat=ACTION_REPEAT,
                seed=SEED,
                device="cuda",
                density=DENSITY,
                smoke=smoke,
                env_contract="v4_8",
                env_class="YieldConflictIndependentV2EnvV4V1 (改法 1+2+3)",
                reward_shaping=REWARD,
            ),
        )

        class Progress(BaseCallback):
            def _on_step(self):
                if self.n_calls % 100 == 0:
                    _write_json_atomic(
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
                    BestTrainingSuccessCallback(run_dir / "best_training_success_model"),
                    RewardBranchProgressCallback(run_dir / "reward_branches.json"),
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
        # 训练结束后 lr 已按 StabilitySAC 调度衰减到 floor（正式：2e-5，smoke 未触发衰减
        # 仍为 5e-5）。用训练结束时刻的期望 lr 校验，而非初始 learning_rate（后者在
        # decay 后与优化器实际 lr 不一致，会抛 ValueError: Optimizer learning-rate mismatch）。
        rate = _lr_at(config, raw_budget)
        verify_optimizer_settings(model, learning_rate=rate, tau=config["tau"])
        model.audit_parameters()

        final = run_dir / "final_model.zip"
        model.save(final)
        _write_json_atomic(
            run_dir / "training_diagnostics.json", model.training_diagnostics()
        )
        _write_json_atomic(
            run_dir / "training_complete.json",
            dict(
                smoke=smoke,
                checkpoint_sha256=_sha256(final),
                raw_steps=raw_budget,
                updates=model._n_updates,
                replay_size=model.replay_buffer.size(),
                wall_seconds=time.time() - started,
            ),
        )
        _write_json_atomic(
            run_dir / "status.json",
            dict(status="trained", smoke=smoke, wall_seconds=time.time() - started),
        )
        return final
    except BaseException as exc:
        _write_json_atomic(
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


# --------------------------------------------------------------------------- #
# 训练：mst_slt（MST+SLT, base）
# --------------------------------------------------------------------------- #
def run_training_mst_slt(
    run_dir: Path,
    smoke: bool,
    max_steps: int | None = None,
    behavior_diagnostics: bool = False,
) -> Path:
    import torch

    torch.set_num_threads(1)

    from tools import train_sb3

    if max_steps is None:
        max_steps = RAW_TRAINING_STEPS
    max_steps = 300 if smoke else max_steps
    learning_starts = 60 if smoke else LEARNING_STARTS
    frequency = 100 if smoke else CHECKPOINT_FREQUENCY
    eval_episodes = 1 if smoke else 100

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    base_env_factory = make_env_factory("base", run_dir / "overlays" / "seed_0")
    factory_calls = 0
    factory_roles: list[str] = []
    diagnostic_recorder = None

    def env_factory(args, *, evaluation=False):
        nonlocal factory_calls, diagnostic_recorder
        factory_calls += 1
        if behavior_diagnostics:
            expected_calls = (
                (1, False, "validation"),
                (2, False, "training"),
                (3, True, "final_evaluation"),
            )
            if factory_calls > len(expected_calls):
                raise RuntimeError(
                    "MST+SLT behavior diagnostics expected validation, training, "
                    "then final evaluation environments; train_sb3 factory call "
                    f"{factory_calls} was unexpected"
                )
            expected_index, expected_evaluation, role = expected_calls[factory_calls - 1]
            if bool(evaluation) != expected_evaluation:
                raise RuntimeError(
                    "MST+SLT behavior diagnostics detected changed environment "
                    f"factory order at call {factory_calls}: expected {role} "
                    f"(evaluation={expected_evaluation}), got "
                    f"evaluation={bool(evaluation)}"
                )
            if expected_index != factory_calls:
                raise AssertionError("internal MST environment factory role mismatch")
            factory_roles.append(role)
        env = base_env_factory(args, evaluation=evaluation)
        if behavior_diagnostics and factory_roles[-1:] == ["training"]:
            env, diagnostic_recorder = _wrap_behavior_diagnostics_env(
                env,
                run_dir,
                "train",
                "mst_slt",
                raw_step_budget=max_steps,
                warmup_raw_steps=learning_starts,
                checkpoint_frequency_raw_steps=frequency,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE["mst_slt"],
                buffer_size=BUFFER_SIZE,
                discount=DISCOUNT,
                env_contract="base",
                device="cuda",
                smoke=smoke,
                learner_diagnostics_path=str((run_dir / "training_diagnostics.json").resolve()),
                learner_diagnostics_source="train_sb3 model.training_diagnostics aggregate",
            )
        return env

    argv = [
        "--algo", "scene_rep",
        "--scenario", SCENARIO,
        "--max-steps", str(max_steps),
        "--learning-starts", str(learning_starts),
        "--checkpoint-freq", str(frequency),
        "--eval-freq", "0",
        "--eval-episodes", str(eval_episodes),
        "--seed", str(SEED),
        "--device", "cuda",
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LEARNING_RATE["mst_slt"]),
        "--discount", str(DISCOUNT),
        "--buffer-size", str(BUFFER_SIZE),
        "--action-repeat", str(ACTION_REPEAT),
        "--output-dir", str(run_dir.parent.resolve()),
        "--model-name", run_dir.name,
        "--ego-control-profile", "direct",
        "--episode-limit-profile", "source",
        "--traffic-protocol", "frozen_80_20",
    ]
    train_sb3.main(
        argv,
        env_factory=env_factory,
        default_output_dir=run_dir.parent,
        # False：本实验的 reward 已被 GeneralizedRewardShapingWrapper 改写为 ±10 量级，
        # 不再是 paper 原始 ±1，不满足 paper 评估合约；且 wrapper 不是
        # PaperSumoSceneEnv 子类，True 会在 train_sb3 末尾的 build_evaluation_provenance
        # 触发 isinstance 严格类型检查 TypeError。False 走简化 provenance，仍完整跑
        # evaluate_model_detailed 并写 final_evaluation.json。
        require_paper_evaluation_contract=False,
        tensorboard_log_root=RESULT_ROOT / "tb",
    )
    if behavior_diagnostics:
        if factory_calls != 3 or factory_roles != [
            "validation",
            "training",
            "final_evaluation",
        ]:
            raise RuntimeError(
                "MST+SLT behavior diagnostics expected validation, training, "
                "and final evaluation environments in order; observed "
                f"calls={factory_calls}, roles={factory_roles}"
            )
    final = run_dir / "final_model.zip"
    if not final.is_file():
        raise FileNotFoundError(f"train_sb3 did not produce {final}")
    _write_json_atomic(
        run_dir / "status.json",
        dict(status="trained", smoke=smoke, method="mst_slt",
             behavior_diagnostics=behavior_diagnostics),
    )
    return final


# --------------------------------------------------------------------------- #
# 训练：HSAC-MLP（混合头 + MLP）与 SAC-MLP（连续头 + MLP）
# --------------------------------------------------------------------------- #
def _make_sac_mlp_model(
    env, *, learning_rate, batch_size, discount, learning_starts,
    buffer_size, action_repeat, seed, device,
):
    """纯 SAC-MLP：``SceneRepSACPolicy`` 连续头 + ``SimpleMlpLstmExtractor(mlp)``，关 SLT。

    与 HSAC-MLP（``make_hsac_model``，``DecisionAlignedSACPolicy`` 混合头）唯一区别是
    动作头；两者同为 MLP 编码器、同 ``representation_coef=0.0``、同 lr=1e-4，构成
    「混合头 vs 连续头」的干净消融（都关 Scene-Rep 表示）。sac_mlp 用 base 契约，
    sac_mlp_v48 用 v4_8 契约（``SimpleMlpLstmExtractor`` 只读 trajectory/map、
    忽略 ``lane_action_mask``，连续头无需 mask，故契约只影响改法 3 的 neighbor 选车）。
    """
    import torch
    from algos.hybrid_action.hsac.simple_encoder import SimpleMlpLstmExtractor
    from algos.sb3_torch import (
        DictNStepReplayBuffer,
        SceneRepSACPolicy,
        SceneRepresentationSAC,
    )

    feature_kwargs = {
        "features_dim": 128,
        "hidden_dim": 128,
        "backbone": "mlp",
        "random_augmentation": SCENARIO != "cross",
        "carla_contract": SCENARIO == "carla",
    }
    policy_kwargs = {
        "features_extractor_class": SimpleMlpLstmExtractor,
        "features_extractor_kwargs": feature_kwargs,
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False,
        "net_arch": {"pi": [128, 32], "qf": [128, 32]},
        "optimizer_class": torch.optim.NAdam,
        "optimizer_kwargs": {"eps": 1e-7},
        "n_critics": 2,
        "action_embedding_dim": 64,
    }
    return SceneRepresentationSAC(
        SceneRepSACPolicy,
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=0,
        batch_size=batch_size,
        tau=5e-3,
        gamma=discount,
        train_freq=(1, "step"),
        gradient_steps=action_repeat,
        n_steps=4,
        replay_buffer_class=DictNStepReplayBuffer,
        replay_buffer_kwargs={
            "n_steps": 4,
            "gamma": discount,
            "source_action_repeat": action_repeat,
        },
        ent_coef="auto_0.2",
        target_entropy="auto",
        policy_kwargs=policy_kwargs,
        representation_coef=0.0,
        action_embedding_dim=64,
        max_grad_norm=5.0,
        raw_learning_starts=learning_starts,
        seed=seed,
        device=device,
        verbose=0,
    )


def _build_mlp_model(method: str, env, *, learning_starts: int):
    """按 method 构造 MLP 编码器的模型：hsac_mlp/hsac_mlp_base=混合头，sac_mlp/sac_mlp_v48=连续头。"""
    if method in ("hsac_mlp", "hsac_mlp_base"):
        from algos.hybrid_action.hsac import make_hsac_model

        return make_hsac_model(
            "hsac_mlp",
            env,
            scenario=SCENARIO,
            learning_rate=LEARNING_RATE[method],
            batch_size=BATCH_SIZE,
            discount=DISCOUNT,
            learning_starts=learning_starts,
            buffer_size=BUFFER_SIZE,
            action_repeat=ACTION_REPEAT,
            seed=SEED,
            device="cuda",
            verbose=0,
        )
    if method in ("sac_mlp", "sac_mlp_v48"):
        return _make_sac_mlp_model(
            env,
            learning_rate=LEARNING_RATE[method],
            batch_size=BATCH_SIZE,
            discount=DISCOUNT,
            learning_starts=learning_starts,
            buffer_size=BUFFER_SIZE,
            action_repeat=ACTION_REPEAT,
            seed=SEED,
            device="cuda",
        )
    raise ValueError(f"not an MLP method: {method}")


def run_training_mlp(
    method: str,
    run_dir: Path,
    smoke: bool,
    max_steps: int | None = None,
    behavior_diagnostics: bool = False,
) -> Path:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor

    from algos.sb3_torch.callbacks import (
        BestTrainingSuccessCallback,
        RawStepControlCallback,
        RewardBranchProgressCallback,
    )
    from reward_shaping_v2 import REWARD_BRANCH_KEYS

    torch.set_num_threads(1)

    if max_steps is None:
        max_steps = RAW_TRAINING_STEPS
    raw_budget = 300 if smoke else max_steps
    frequency = 100 if smoke else CHECKPOINT_FREQUENCY
    learning_starts = 60 if smoke else LEARNING_STARTS
    overlay_root = run_dir / "overlays"
    namespace = "sm" if smoke else "tr"
    if method in ("hsac_mlp", "sac_mlp_v48"):
        env_adapter = "v4_8"
    elif method == "hsac_mlp_base":
        env_adapter = "v4_base"
    else:  # sac_mlp
        env_adapter = "base"

    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = None
    model = None
    diagnostic_recorder = None

    _write_json_atomic(
        run_dir / "status.json",
        dict(status="training", pid=os.getpid(), smoke=smoke, started_at=started),
    )

    try:
        env_base = make_env_factory(env_adapter, overlay_root / f"ns_{namespace}")(
            _environment_namespace(), evaluation=False
        )
        if behavior_diagnostics:
            env_base, diagnostic_recorder = _wrap_behavior_diagnostics_env(
                env_base,
                run_dir,
                "train",
                method,
                raw_step_budget=raw_budget,
                warmup_raw_steps=learning_starts,
                checkpoint_frequency_raw_steps=frequency,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE[method],
                buffer_size=BUFFER_SIZE,
                discount=DISCOUNT,
                env_contract=env_adapter,
                device="cuda",
                smoke=smoke,
            )
        env = Monitor(
            env_base,
            filename=str(run_dir / "train_monitor.csv"),
            info_keywords=(
                "raw_simulation_steps",
                "is_success",
                "collision",
                "off_route",
                "max_time",
            )
            + REWARD_BRANCH_KEYS,
        )
        model = _build_mlp_model(method, env, learning_starts=learning_starts)

        _write_json_atomic(
            run_dir / "arguments.json",
            dict(
                method=method,
                scenario=SCENARIO,
                raw_budget=raw_budget,
                checkpoint_frequency=frequency,
                learning_starts_raw_steps=learning_starts,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE[method],
                buffer_size=BUFFER_SIZE,
                discount=DISCOUNT,
                action_repeat=ACTION_REPEAT,
                seed=SEED,
                device="cuda",
                density=DENSITY,
                smoke=smoke,
                behavior_diagnostics=behavior_diagnostics,
                run_root=str(Path(run_dir).parent.resolve()),
                env_contract=env_adapter,
                env_class=(
                    "YieldConflictIndependentV2EnvV4V1 (改法 1+2+3)"
                    if env_adapter == "v4_8"
                    else "YieldObsIndependentV2EnvV4V1 (改法 1+2, v4 观察)"
                    if env_adapter == "v4_base"
                    else "YieldObsIndependentV2EnvV1 (改法 1+2)"
                ),
                reward_shaping=REWARD,
            ),
        )

        class Progress(BaseCallback):
            def _on_step(self):
                if self.n_calls % 100 == 0:
                    _write_json_atomic(
                        run_dir / "progress.json",
                        dict(
                            raw_steps=getattr(self.model, "_raw_steps_seen", None),
                            updates=self.model._n_updates,
                            updated_at=time.time(),
                        ),
                    )
                return True

        training_callbacks = [
            RawStepControlCallback(
                raw_step_budget=raw_budget,
                checkpoint_frequency=frequency,
                checkpoint_path=run_dir / "checkpoints",
                checkpoint_prefix="ckpt",
            ),
            BestTrainingSuccessCallback(run_dir / "best_training_success_model"),
            RewardBranchProgressCallback(run_dir / "reward_branches.json"),
            Progress(),
        ]
        if diagnostic_recorder is not None:
            training_callbacks.append(_make_behavior_optimization_callback(diagnostic_recorder))

        model.learn(
            total_timesteps=raw_budget,
            callback=CallbackList(training_callbacks),
        )
        if model._raw_steps_seen != raw_budget:
            raise AssertionError(
                f"raw-step budget mismatch: {model._raw_steps_seen} != {raw_budget}"
            )

        final = run_dir / "final_model.zip"
        model.save(final)
        _write_json_atomic(
            run_dir / "training_complete.json",
            dict(
                smoke=smoke,
                checkpoint_sha256=_sha256(final),
                raw_steps=raw_budget,
                updates=model._n_updates,
                replay_size=model.replay_buffer.size(),
                behavior_diagnostics=behavior_diagnostics,
                wall_seconds=time.time() - started,
            ),
        )
        _write_json_atomic(
            run_dir / "status.json",
            dict(status="trained", smoke=smoke, wall_seconds=time.time() - started),
        )
        return final
    except BaseException as exc:
        _write_json_atomic(
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


def _read_raw_steps_from_zip(model_path: Path) -> int:
    """不加载整个模型，只从 SB3 zip 的 ``data`` 读 ``_raw_steps_seen``（续训基准 raw 步）。"""
    import zipfile

    with zipfile.ZipFile(model_path) as archive:
        data = json.loads(archive.read("data").decode("utf-8"))
    if "_raw_steps_seen" not in data:
        raise KeyError(f"{model_path} 的 data 里没有 _raw_steps_seen（非 yield_v2 模型？）")
    return int(data["_raw_steps_seen"])


# --------------------------------------------------------------------------- #
# 继续训练：加载已保存模型，续训 extra_steps raw 步
# --------------------------------------------------------------------------- #
def run_continue_training(
    method: str,
    extra_steps: int,
    model_path: Path | None,
    smoke: bool,
) -> Path:
    """从已保存模型续训。

    加载 ``model_path``（默认 ``<run_dir>/final_model.zip``）的权重/优化器/计数器，
    用与原始训练完全一致的环境契约（env class + 改法 3 + reward wrapper + 密度）再
    续训 ``extra_steps`` raw 步，输出到 ``<run_dir>/continue_<global_raw_steps>/``。

    * replay buffer 不随 SB3 ``save()`` 持久化（原训练也未单独保存），续训时 buffer
      为空。沿用 ``train_intersection_mst_slt_curriculum.py`` 的做法：把
      ``raw_learning_starts`` 设为一个小窗口（500，smoke 100）先填满 buffer
      （>= n_steps/batch）再训练，避免「空 buffer 上 train()」崩溃。
    * 续训结束把 ``model._raw_steps_seen`` 写成全局步数（base + extra），保存后
      的模型可直接再次续训（下次读到的 base 即全局步数）。
    * 支持所有 6 个方法：hold35k 走 ``StabilitySAC.load``，其余走
      ``SceneRepresentationSAC.load``。
    """
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor

    from algos.sb3_torch.callbacks import RawStepControlCallback

    torch.set_num_threads(1)

    run_dir = RESULT_ROOT / f"{method}__{SCENARIO}{RUN_SUFFIX}"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir 不存在（先跑正式训练）：{run_dir}")

    model_path = Path(model_path) if model_path else (run_dir / "final_model.zip")
    if not model_path.is_file():
        raise FileNotFoundError(f"待续训模型不存在：{model_path}")

    adapter = _env_adapter(method)
    frequency = 100 if smoke else CHECKPOINT_FREQUENCY
    namespace = "sm" if smoke else "ct"

    base_raw_steps = _read_raw_steps_from_zip(model_path)
    global_raw_steps = base_raw_steps + int(extra_steps)
    cont_dir = run_dir / f"continue_{global_raw_steps}"
    cont_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    env = None
    model = None

    # 先建环境（load 需要与模型 observation/action space 匹配）；Monitor 日志与
    # 续训产物统一落在 cont_dir 下，避免覆盖原始 run_dir 的 train_monitor.csv。
    env = Monitor(
        make_env_factory(adapter, cont_dir / "overlays" / f"ns_{namespace}")(
            _environment_namespace(), evaluation=False
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
        if method == "hold35k":
            from tools.v48_stability_v4.model import StabilitySAC

            model = StabilitySAC.load(model_path, env=env, device="cuda")
        else:  # mst_slt / hsac_mlp / sac_mlp / sac_mlp_v48 / hsac_mlp_base
            from algos.sb3_torch.sac import SceneRepresentationSAC

            model = SceneRepresentationSAC.load(model_path, env=env, device="cuda")

        if int(model._raw_steps_seen) != base_raw_steps:
            raise AssertionError(
                f"加载后 _raw_steps_seen({model._raw_steps_seen}) 与 zip 记录 "
                f"({base_raw_steps}) 不一致"
            )

        if method == "hold35k":
            # stability_trace_path 不在 save 清单内，续训需重指到新 trace 文件。
            model.stability_trace_path = str(cont_dir / "optimization_trace.jsonl")

        # raw_learning_starts 同时控制「随机探索窗口」与「训练开始」，且 load 后 replay
        # buffer 为空：设 0 会在 buffer 未满 n_steps 时训练而崩溃。沿用 curriculum
        # 脚本做法，设一个小窗口先填满 buffer（>= n_steps/batch）再训练；策略侧
        # raw_learning_starts 越小越早收敛到 load 的策略动作。
        model.raw_learning_starts = 100 if smoke else 500

        _write_json_atomic(
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

        _write_json_atomic(
            cont_dir / "arguments.json",
            dict(
                method=method,
                scenario=SCENARIO,
                base_model=str(model_path),
                base_raw_steps=base_raw_steps,
                extra_raw_steps=extra_steps,
                global_raw_steps=global_raw_steps,
                checkpoint_frequency=frequency,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE[method],
                buffer_size=BUFFER_SIZE,
                discount=DISCOUNT,
                action_repeat=ACTION_REPEAT,
                seed=SEED,
                device="cuda",
                density=DENSITY,
                smoke=smoke,
                env_contract=adapter,
                reward_shaping=REWARD,
            ),
        )

        class Progress(BaseCallback):
            def _on_step(self):
                if self.n_calls % 100 == 0:
                    _write_json_atomic(
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
        _write_json_atomic(
            cont_dir / "training_complete.json",
            dict(
                smoke=smoke,
                checkpoint_sha256=_sha256(final),
                base_raw_steps=base_raw_steps,
                extra_raw_steps=extra_steps,
                global_raw_steps=global_raw_steps,
                updates=model._n_updates,
                replay_size=model.replay_buffer.size(),
                wall_seconds=time.time() - started,
            ),
        )
        _write_json_atomic(
            cont_dir / "status.json",
            dict(status="trained", smoke=smoke, wall_seconds=time.time() - started),
        )

        result = run_evaluation(method, cont_dir, final, smoke=smoke)
        curve = plot_training_curves(method, cont_dir, raw_steps=global_raw_steps)

        manifest = dict(
            method=method,
            scenario=SCENARIO,
            run_dir=str(run_dir),
            base_model=str(model_path),
            base_raw_steps=base_raw_steps,
            extra_raw_steps=extra_steps,
            global_raw_steps=global_raw_steps,
            continuation_dir=str(cont_dir),
            final_model=str(final),
            evaluation=result["summary"],
            training_curve=str(curve) if curve else None,
            smoke=smoke,
            reward_shaping=REWARD,
            note=(
                "续训：加载 base_model 的权重/优化器/计数器；replay buffer 从空重新累积；"
                "raw_learning_starts 设为 500(smoke 100) 先填满 buffer；"
                "checkpoint 名为局部步数，保存前把 _raw_steps_seen 写回全局步数。"
            ),
        )
        _write_json_atomic(
            run_dir / f"continuation_{global_raw_steps}.json", manifest
        )
        print(
            json.dumps(
                dict(method=method, summary=result["summary"]),
                ensure_ascii=False,
                indent=2,
            )
        )
        return final
    except BaseException as exc:
        _write_json_atomic(
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


# --------------------------------------------------------------------------- #
# 评估（6 worker 子进程）
# --------------------------------------------------------------------------- #
def run_eval_worker(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    from algos.sb3_torch.evaluation import evaluate_model_detailed

    torch.set_num_threads(1)
    method = args.method
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_root = output_dir / "overlays" / f"eval_w{args.worker_id}"
    start = args.start_episode
    end = args.end_episode
    n = end - start
    model_path = str(args.model_path)
    behavior_diagnostics = bool(getattr(args, "behavior_diagnostics", False))
    eval_phase = f"eval_worker_{int(args.worker_id):02d}"
    eval_metadata = dict(
        evaluation_seed_start=SEED_START[method] + start,
        evaluation_episode_start=start,
        evaluation_episode_end=end,
        evaluation_episodes=n,
        deterministic=True,
        checkpoint=str(Path(model_path).resolve()),
        device="cpu",
        smoke=bool(getattr(args, "smoke", False)),
    )

    if method in ("hold35k", "hsac_mlp", "hsac_mlp_base"):
        from tools.v48_stability_v4.model import StabilitySAC, use_actor_only

        eval_adapter = "v4_base" if method == "hsac_mlp_base" else "v4_8"
        env = make_env_factory(eval_adapter, overlay_root)(
            _environment_namespace(), evaluation=True
        )
        if behavior_diagnostics:
            env, _ = _wrap_behavior_diagnostics_env(
                env, output_dir, eval_phase, method,
                env_contract=eval_adapter, **eval_metadata
            )
        if method == "hold35k":
            loaded = StabilitySAC.load(model_path, env=env, device="cpu", buffer_size=32)
        else:  # hsac_mlp / hsac_mlp_base
            from algos.sb3_torch.sac import SceneRepresentationSAC

            loaded = SceneRepresentationSAC.load(model_path, env=env, device="cpu", buffer_size=32)
        model = use_actor_only(loaded)
        records = []
        try:
            for index in range(start, end):
                episode_seed = SEED_START[method] + index
                np.random.seed(episode_seed + 600_000)
                torch.manual_seed(episode_seed + 600_000)
                # 写到底层 env：GeneralizedRewardShapingWrapper 无 __getattr__，
                # 写到 wrapper 只会落在 wrapper 自身、底层 env 读不到，traffic seed 失效。
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
    else:  # mst_slt / sac_mlp / sac_mlp_v48（连续头，按 method 决定环境契约）
        from algos.sb3_torch.sac import SceneRepresentationSAC

        env_adapter = "v4_8" if method == "sac_mlp_v48" else "base"
        env = make_env_factory(env_adapter, overlay_root)(
            _environment_namespace(), evaluation=True
        )
        if behavior_diagnostics:
            env, _ = _wrap_behavior_diagnostics_env(
                env, output_dir, eval_phase, method,
                env_contract=env_adapter, **eval_metadata
            )
        model = SceneRepresentationSAC.load(
            model_path, env=env, device="cpu", buffer_size=32
        )
        records = []
        try:
            # 写到底层 env（同上，wrapper 无 __getattr__ 会吞掉这些属性）。
            env.unwrapped._traffic_episode_index = start
            env.unwrapped._traffic_roll = None
            detailed = evaluate_model_detailed(
                model,
                env,
                episodes=n,
                seed=SEED_START[method] + start,
                deterministic=True,
                sumo_step_seconds=0.1,
                policy_action_hold=1,
            )
            for i, rec in enumerate(detailed.episode_records):
                r = rec.to_dict()
                r["episode"] = start + i
                records.append(r)
        finally:
            env.close()

    _write_json_atomic(
        output_dir / f"eval_worker_{args.worker_id:02d}.json",
        dict(
            worker_id=args.worker_id,
            method=method,
            start_episode=start,
            end_episode=end,
            model_path=model_path,
            records=records,
        ),
    )
    print(
        json.dumps(
            dict(
                worker_id=args.worker_id,
                method=method,
                episodes=len(records),
                start=start,
                end=end,
            )
        ),
        flush=True,
    )


def summarize_records(records: list[dict]) -> dict:
    import numpy as np
    from algos.sb3_torch.evaluation import (
        EVALUATION_RETURN_PROTOCOL_VERSION,
        RAW_RETURN_PROTOCOL_VERSION,
        REWARD_COMPONENT_KEYS,
    )

    n = len(records)
    returns = [r["episode_return"] for r in records]
    raw_returns = [
        float(r["raw_episode_return"])
        for r in records
        if r.get("raw_episode_return") is not None
    ]
    reward_component_means = {}
    for key in REWARD_COMPONENT_KEYS:
        values = [float(r[key]) for r in records if r.get(key) is not None]
        reward_component_means[key] = float(np.mean(values)) if values else None
    complete_component_records = [
        r for r in records
        if all(r.get(key) is not None for key in REWARD_COMPONENT_KEYS)
    ]
    reconciliation_errors = [
        abs(
            float(r["episode_return"])
            - sum(float(r[key]) for key in REWARD_COMPONENT_KEYS)
        )
        for r in complete_component_records
    ]
    return_protocols = {
        r.get("evaluation_return_protocol_version")
        for r in records
        if r.get("evaluation_return_protocol_version") is not None
    }
    component_protocols = {
        r.get("reward_component_protocol_version")
        for r in records
        if r.get("reward_component_protocol_version") is not None
    }
    raw_sources = {
        r.get("raw_return_source")
        for r in records
        if r.get("raw_return_source") is not None
    }
    raw_source_protocol = (
        "mixed_info_and_environment_reward_fallback"
        if len(raw_sources) > 1
        else next(iter(raw_sources), "not_available")
    )
    raw_lengths = [r["raw_steps"] for r in records]
    decision_lengths = [r["decision_steps"] for r in records]
    completion_times = [
        r["completion_time_seconds"]
        for r in records
        if r.get("completion_time_seconds") is not None
    ]
    successes = sum(int(r["success"]) for r in records)
    return dict(
        episodes=n,
        success_rate=successes / n if n else 0.0,
        collision_rate=sum(int(r["collision"]) for r in records) / n if n else 0.0,
        off_route_rate=sum(int(r["off_route"]) for r in records) / n if n else 0.0,
        timeout_rate=sum(int(r["timeout"]) for r in records) / n if n else 0.0,
        mean_return=float(np.mean(returns)) if n else None,
        std_return=float(np.std(returns)) if n else None,
        raw_mean_return=float(np.mean(raw_returns)) if raw_returns else None,
        raw_std_return=float(np.std(raw_returns)) if raw_returns else None,
        reward_component_means=reward_component_means,
        reward_component_coverage=len(complete_component_records),
        reward_component_reconciliation_max_abs_error=(
            max(reconciliation_errors) if reconciliation_errors else None
        ),
        evaluation_return_protocol_version=(
            next(iter(return_protocols))
            if len(return_protocols) == 1
            else (
                "mixed"
                if return_protocols
                else EVALUATION_RETURN_PROTOCOL_VERSION if n == 0 else None
            )
        ),
        raw_return_protocol_version=(
            f"{RAW_RETURN_PROTOCOL_VERSION};source={raw_source_protocol}"
        ),
        reward_component_protocol_version=(
            next(iter(component_protocols))
            if len(component_protocols) == 1
            and len(complete_component_records) == n
            else (
                "mixed"
                if component_protocols and len(component_protocols) > 1
                else None
            )
        ),
        mean_decision_steps=float(np.mean(decision_lengths)) if n else None,
        mean_raw_steps=float(np.mean(raw_lengths)) if n else None,
        mean_success_completion_time_seconds=(
            float(np.mean(completion_times)) if completion_times else None
        ),
        std_success_completion_time_seconds=(
            float(np.std(completion_times)) if completion_times else None
        ),
        successful_episodes=successes,
    )


def run_evaluation(
    method: str,
    run_dir: Path,
    final_model: Path,
    smoke: bool,
    behavior_diagnostics: bool = False,
) -> dict:
    # A single diagnostic writer owns each method's evaluation stream. This also
    # keeps the paired final evaluations bounded after two concurrent trainers.
    workers = 1 if behavior_diagnostics else (2 if smoke else EVAL_WORKERS)
    episodes = 8 if smoke else EVAL_EPISODES_TOTAL
    device = "cpu"

    ranges: list[tuple[int, int]] = []
    for worker_id in range(workers):
        start = worker_id * episodes // workers
        end = (worker_id + 1) * episodes // workers
        if end > start:
            ranges.append((start, end))

    log_dir = run_dir / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[str, subprocess.Popen] = {}
    stop_requested = threading.Event()
    started = time.time()

    def execute(worker_id: int, start: int, end: int):
        if stop_requested.is_set():
            return dict(worker_id=worker_id, exit_code=-1, cancelled=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log = log_dir / f"worker_{worker_id:02d}_{stamp}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "eval-worker",
            "--method",
            method,
            "--scenario",
            SCENARIO,
            "--depart-scale",
            str(DEPART_SCALE),
            "--eval-traffic-split",
            EVAL_TRAFFIC_SPLIT,
            "--worker-id",
            str(worker_id),
            "--start-episode",
            str(start),
            "--end-episode",
            str(end),
            "--model-path",
            str(final_model),
            "--output-dir",
            str(run_dir),
        ]
        if behavior_diagnostics:
            command.append("--behavior-diagnostics")
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            running[str(worker_id)] = process
            returncode = process.wait()
            running.pop(str(worker_id), None)
        return dict(
            worker_id=worker_id,
            exit_code=returncode,
            log=str(log),
            completed_at=time.time(),
        )

    done: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(execute, wid, start, end): wid
            for wid, (start, end) in enumerate(ranges)
        }
        try:
            for future in as_completed(futures):
                wid = futures[future]
                done[wid] = future.result()
                print(
                    f"[{method}] eval worker {wid}: exit {done[wid]['exit_code']}",
                    flush=True,
                )
        except BaseException:
            stop_requested.set()
            for future in futures:
                future.cancel()
            for process in list(running.values()):
                try:
                    import psutil

                    parent = psutil.Process(process.pid)
                    for child in parent.children(recursive=True):
                        child.terminate()
                    parent.terminate()
                except Exception:
                    pass
            raise

    failed = [wid for wid, result in done.items() if result["exit_code"] != 0]
    if failed:
        raise RuntimeError(f"evaluation workers failed: {failed}; inspect {log_dir}")

    records: list[dict] = []
    for wid in sorted(done):
        path = run_dir / f"eval_worker_{wid:02d}.json"
        records.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    records.sort(key=lambda r: r["episode"])
    summary = summarize_records(records)

    result = dict(
        identity=dict(
            method=method,
            scenario=SCENARIO,
            depart_scale=DEPART_SCALE,
            eval_traffic_split=EVAL_TRAFFIC_SPLIT,
            checkpoint=str(final_model),
            checkpoint_sha256=_sha256(final_model),
            episodes=episodes,
            workers=workers,
            episodes_per_worker=[end - start for start, end in ranges],
            device=device,
            seed_start=SEED_START[method],
            deployment="actor_deterministic",
            smoke=smoke,
            behavior_diagnostics=behavior_diagnostics,
            reward_shaping=REWARD,
            evaluation_return_protocol_version="environment_step_reward_v2",
            note=(
                "episode_return/mean_return sum the reward returned by env.step "
                "(yield-v2 shaped reward when its wrapper is active; protocol "
                "environment_step_reward_v2). raw_episode_return/raw_mean_return "
                "sum info['undiscounted_reward'] when exposed; each episode records "
                "raw_return_source when it must fall back to env.step reward. "
                "Terminal outcome rates remain event flags."
            ),
        ),
        summary=summary,
        episode_records=records,
        wall_seconds=time.time() - started,
    )
    _write_json_atomic(run_dir / "evaluation_results.json", result)
    return result


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def plot_training_curves(
    method: str, run_dir: Path, raw_steps: int | None = None
) -> Optional[Path]:
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    monitor_csv = run_dir / "train_monitor.csv"
    if not monitor_csv.is_file():
        raise FileNotFoundError(monitor_csv)

    with monitor_csv.open(newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
        rows = list(csv.DictReader(lines))
    if not rows:
        # smoke 训练步数太少时（如混合头 warmup 期间 300 raw 步内无 episode 结束），
        # train_monitor.csv 只有表头没有数据行。跳过绘图并返回 None（调用方按 None 处理）。
        return None

    episode = np.arange(1, len(rows) + 1)
    returns = np.asarray([float(r["r"]) for r in rows], dtype=float)
    successes = np.asarray(
        [
            1.0 if str(r.get("is_success", "")).strip().lower() == "true" else 0.0
            for r in rows
        ],
        dtype=float,
    )

    def ema(values: np.ndarray, span: int) -> np.ndarray:
        span = max(1, min(span, len(values)))
        alpha = 2.0 / (span + 1)
        out = np.empty_like(values, dtype=float)
        acc = values[0]
        for i, v in enumerate(values):
            acc = alpha * v + (1.0 - alpha) * acc
            out[i] = acc
        return out

    _TITLES = {
        "hold35k": "hold35k",
        "mst_slt": "MST+SLT",
        "hsac_mlp": "HSAC-MLP",
        "sac_mlp": "SAC-MLP",
        "sac_mlp_v48": "SAC-MLP (v4_8)",
        "hsac_mlp_base": "HSAC-MLP (base)",
    }
    title = _TITLES[method]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(
        episode, returns, color="#1f77b4", alpha=0.35, lw=1.0, label="episode return"
    )
    axes[0].plot(episode, ema(returns, 10), color="#1f77b4", lw=2.0, label="return (EMA 10)")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("shaped episode return")
    axes[0].set_title(f"{title} on intersection (yield v2) — episode return")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    window = 10
    if len(rows) >= window:
        success_rate = np.convolve(successes, np.ones(window) / window, mode="valid")
        axes[1].plot(
            episode[window - 1 :], success_rate, color="#ff7f0e", lw=2.0,
            label=f"success rate (window {window})",
        )
    else:
        axes[1].plot(
            episode, successes, color="#ff7f0e", marker="o", lw=1.5,
            label="success (per episode)",
        )
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("success rate")
    axes[1].set_title(f"{title} on intersection (yield v2) — success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"{title} / intersection / yield v2 / "
        f"{raw_steps if raw_steps is not None else RAW_TRAINING_STEPS} raw steps",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = run_dir / "training_curve.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_train_method(
    method: str, smoke: bool, behavior_diagnostics: bool = False
) -> Path:
    run_dir = RESULT_ROOT / f"{method}__{SCENARIO}{RUN_SUFFIX}"

    if method == "hold35k":
        final = run_training_hold35k(run_dir, smoke=smoke)
    elif method == "mst_slt":
        final = run_training_mst_slt(
            run_dir, smoke=smoke, behavior_diagnostics=behavior_diagnostics
        )
    else:  # hsac_mlp / sac_mlp / sac_mlp_v48 / hsac_mlp_base
        final = run_training_mlp(
            method, run_dir, smoke=smoke,
            behavior_diagnostics=behavior_diagnostics,
        )

    result = run_evaluation(
        method, run_dir, final, smoke=smoke,
        behavior_diagnostics=behavior_diagnostics,
    )
    curve = plot_training_curves(method, run_dir)

    manifest = dict(
        method=method,
        scenario=SCENARIO,
        depart_scale=DEPART_SCALE,
        eval_traffic_split=EVAL_TRAFFIC_SPLIT,
        run_dir=str(run_dir),
        final_model=str(final),
        evaluation=result["summary"],
        training_curve=str(curve) if curve else None,
        smoke=smoke,
        raw_training_steps=RAW_TRAINING_STEPS,
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
        train_workers=TRAIN_WORKERS,
        eval_episodes_total=EVAL_EPISODES_TOTAL,
        eval_workers=EVAL_WORKERS,
        hyperparams=dict(
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            learning_starts=LEARNING_STARTS,
        ),
        reward_shaping=REWARD,
        env_class=(
            "YieldConflictIndependentV2EnvV4V1 (改法 1+2+3)"
            if method in ("hold35k", "hsac_mlp", "sac_mlp_v48")
            else "YieldObsIndependentV2EnvV4V1 (改法 1+2, v4 观察)"
            if method == "hsac_mlp_base"
            else "YieldObsIndependentV2EnvV1 (改法 1+2)"
        ),
    )
    _write_json_atomic(run_dir / "experiment_manifest.json", manifest)
    print(
        json.dumps(
            dict(method=method, summary=result["summary"]),
            ensure_ascii=False,
            indent=2,
        )
    )
    return run_dir


def run_full(smoke: bool, behavior_diagnostics: bool = False) -> Path:
    if not smoke and not _cuda_available():
        raise RuntimeError("CUDA unavailable; training has no CPU fallback")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    if not smoke:
        existing = [
            m for m in METHODS if (RESULT_ROOT / f"{m}__{SCENARIO}{RUN_SUFFIX}").exists()
        ]
        if existing:
            raise RuntimeError(
                f"output dirs already exist (remove first): {existing}"
            )

    script = str(Path(__file__).resolve())
    max_attempts = 1 if smoke else 2
    codes: dict[str, int] = {}
    for attempt in range(1, max_attempts + 1):
        todo = [m for m in METHODS if m not in codes or codes[m] != 0]
        if not todo:
            break
        if attempt > 1:
            import shutil

            print(f"[launcher] attempt {attempt}: retrying failed methods {todo}")
            for m in todo:
                shutil.rmtree(RESULT_ROOT / f"{m}__{SCENARIO}{RUN_SUFFIX}", ignore_errors=True)
        procs: dict[str, subprocess.Popen] = {}
        for method in todo:
            cmd = [
                sys.executable, script, "train-method", "--method", method,
                "--scenario", SCENARIO,
                "--depart-scale", str(DEPART_SCALE),
                "--eval-traffic-split", EVAL_TRAFFIC_SPLIT,
            ]
            if smoke:
                cmd.append("--smoke")
            if behavior_diagnostics:
                cmd.append("--behavior-diagnostics")
            log = RESULT_ROOT / f"{method}{RUN_SUFFIX}__train_launcher.log"
            with log.open("w", encoding="utf-8") as handle:
                procs[method] = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            print(f"[launcher] {method} attempt {attempt} (pid {procs[method].pid})")
        for method in todo:
            codes[method] = procs[method].wait()
            print(f"[launcher] {method} attempt {attempt} exit {codes[method]}")

    per_method: dict = {}
    for method in METHODS:
        manifest_path = (
            RESULT_ROOT / f"{method}__{SCENARIO}{RUN_SUFFIX}" / "experiment_manifest.json"
        )
        if manifest_path.is_file():
            per_method[method] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )

    failed = {m: c for m, c in codes.items() if c != 0}
    _write_json_atomic(
        RESULT_ROOT / "experiment_manifest.json",
        dict(
            scenario=SCENARIO,
            depart_scale=DEPART_SCALE,
            eval_traffic_split=EVAL_TRAFFIC_SPLIT,
            methods=per_method,
            exit_codes=codes,
            failed=list(failed),
            raw_training_steps=RAW_TRAINING_STEPS,
            checkpoint_frequency=CHECKPOINT_FREQUENCY,
            train_workers=TRAIN_WORKERS,
            reward_shaping=REWARD,
            behavior_diagnostics=behavior_diagnostics,
            smoke=smoke,
        ),
    )
    print(json.dumps(per_method, ensure_ascii=False, indent=2))
    if failed:
        raise RuntimeError(
            f"training worker(s) failed after {max_attempts} attempt(s): {failed}; "
            f"inspect {RESULT_ROOT}/*__train_launcher.log"
        )
    return RESULT_ROOT


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("eval-worker", "train-method", "continue-train"),
        default=None,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--behavior-diagnostics",
        action="store_true",
        help="Record bounded route/lane behavior telemetry during normal train/eval",
    )
    parser.add_argument("--method", choices=METHODS)
    from envs.sumo.random_intersection import RANDOM_INTERSECTION_SCENARIOS

    parser.add_argument(
        "--scenario",
        choices=("intersection", *RANDOM_INTERSECTION_SCENARIOS),
        default=None,
        help="SUMO scenario/profile (default preserves the existing intersection run)",
    )
    parser.add_argument(
        "--depart-scale",
        type=float,
        default=None,
        help="Legacy explicit-vehicle departure scaling; random profiles require 1.0",
    )
    parser.add_argument(
        "--eval-traffic-split",
        choices=("validation", "test"),
        default="validation",
        help="Random-flow evaluation seed domain; test is a held-out domain",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--start-episode", type=int)
    parser.add_argument("--end-episode", type=int)
    parser.add_argument("--extra-steps", type=int)
    args = parser.parse_args(argv)
    _configure_cli_scene(parser, args)

    if args.command == "eval-worker":
        if (
            args.method is None
            or args.worker_id is None
            or args.start_episode is None
            or args.end_episode is None
        ):
            parser.error(
                "eval-worker requires --method, --worker-id, --start-episode, --end-episode"
            )
        if args.model_path is None or args.output_dir is None:
            parser.error("eval-worker requires --model-path and --output-dir")
        run_eval_worker(args)
        return 0

    if args.command == "train-method":
        if args.method is None:
            parser.error("train-method requires --method")
        run_train_method(
            args.method,
            smoke=args.smoke,
            behavior_diagnostics=args.behavior_diagnostics,
        )
        return 0

    if args.command == "continue-train":
        if args.method is None:
            parser.error("continue-train requires --method")
        if args.extra_steps is None or args.extra_steps <= 0:
            parser.error("continue-train requires --extra-steps > 0")
        run_continue_training(
            args.method, args.extra_steps, args.model_path, smoke=args.smoke
        )
        return 0

    if args.eval_only:
        if args.method is None or args.model_path is None or args.output_dir is None:
            parser.error("--eval-only requires --method, --model-path and --output-dir")
        run_dir = Path(args.output_dir)
        _guard_random_evaluation_output(
            run_dir, scenario=SCENARIO, traffic_split=EVAL_TRAFFIC_SPLIT
        )
        result = run_evaluation(
            args.method,
            run_dir,
            Path(args.model_path),
            smoke=args.smoke,
            behavior_diagnostics=args.behavior_diagnostics,
        )
        plot_training_curves(args.method, run_dir)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0

    run_full(smoke=args.smoke, behavior_diagnostics=args.behavior_diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
