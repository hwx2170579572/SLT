"""在「depart 排序版」intersection 副本场景上训练两个脚本里的所有方法。

背景
----
原 ``intersection`` 场景的 traffic 文件按**流向**分组，组间 depart 时间大幅回退，SUMO
报「Route file should be sorted by departure time, ignoring ...」后把左右手车流静默
忽略，实际只有北→南一股车流在跑。``intersection_sorted`` 副本把 traffic 按
``(depart, departLane)`` 升序重排，三股车流都真正参与仿真。

本脚本在 ``intersection_sorted`` 上训练三类方法（发车间隔可自定义，默认 depart×4.0
降密度），复用既有脚本的**训练 / 评估 / 绘图**底层函数，并把它们的 ``SCENARIO`` 常量
指向新场景：

  yield_v2 家族（``train_intersection_yield_v2``：DEPART_SCALE 降密度 + reward v2 +
  改法 1+2/3）: hold35k / mst_slt / hsac_mlp / sac_mlp / sac_mlp_v48 / hsac_mlp_base
  legacy 家族（``train_intersection_hold35k_mst_fixed``：原始密度 + 纸面 reward +
  改法 1+2）: hold35k_legacy / mst_slt_legacy
  d1 家族（本文件 D1 表征消融，父基线 sac_mlp / hsac_mlp_base）: sac_mlp_d1_st /
  sac_mlp_d1_st_attn / sac_mlp_d1_st_rt / sac_mlp_d1_st_rt_topo / sac_mlp_d1_full /
  sac_mlp_d1_st_rt_topo_3slot /
  hsac_mlp_base_d1_st / hsac_mlp_base_d1_st_rt / hsac_mlp_base_d1_st_rt_topo /
  hsac_mlp_base_d1_full

默认训练 ``mst_slt`` 与 ``sac_mlp_d1_st_rt`` 两个方法（两 CUDA worker 并行）。

结果目录 ``fast-developer/{方法名}__intersection_sorted_depart{scale}``。

用法::

    python fast-developer/train_intersection_yield_v2_d1.py --make-scenario-only
    python fast-developer/train_intersection_yield_v2_d1.py --smoke   # 默认两方法 smoke
    python fast-developer/train_intersection_yield_v2_d1.py           # 默认两方法正式训练
    python fast-developer/train_intersection_yield_v2_d1.py --method mst_slt --smoke
    python fast-developer/train_intersection_yield_v2_d1.py --method sac_mlp_d1_st_rt \
        --depart-scale 3.0
    python fast-developer/train_intersection_yield_v2_d1.py --eval-only \
        --method sac_mlp_d1_st_rt --model-path <run_dir>/final_model.zip \
        --output-dir <run_dir> --depart-scale 4.0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_intersection_yield_v2 as base
import train_intersection_hold35k_mst_fixed as legacy
from envs.sumo.random_intersection import (
    ASSET_ROOT as RANDOM_INTERSECTION_ASSET_ROOT,
    RANDOM_INTERSECTION_SCENARIOS,
    ensure_random_intersection_assets,
    get_random_intersection_config,
    is_random_intersection_scenario,
)

# 新副本场景（depart 排序版 intersection）与 traffic 源。
NEW_SCENARIO = "intersection_sorted"
NEW_SOURCE_TRAFFIC = (
    PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1"
    / "intersection_sorted" / "traffic"
)
SUPPORTED_SCENARIOS = ("intersection_sorted", *RANDOM_INTERSECTION_SCENARIOS)
EVAL_TRAFFIC_SPLIT = "validation"
_LEGACY_MAKE_ENV_FACTORY = legacy.make_env_factory

# 发车间隔缩放（>1.0 降密度）。可自定义，默认 4.0（三股流下理论可完成档）。
DEFAULT_DEPART_SCALE = 4.0
DEPART_SCALE = DEFAULT_DEPART_SCALE

# d1 训练 raw 步数（--max-steps 覆盖）。
MAX_STEPS = base.RAW_TRAINING_STEPS

# d1 方法当前场景（模块级副本；_apply_patch 时同步为 NEW_SCENARIO，供
# _build_model_d1 / run_training_d1 判断 random_augmentation / carla_contract 用）。
SCENARIO = base.SCENARIO
RESULT_ROOT = base.RESULT_ROOT

# 父方法映射：D1 方法继承其父基线的 env 契约 / seed 块 / 动作头（连续 vs 混合）。
PARENT = {
    "sac_mlp_d1_st": "sac_mlp",
    "sac_mlp_d1_st_attn": "sac_mlp",
    "sac_mlp_d1_st_rt": "sac_mlp",
    "sac_mlp_d1_st_rt_gate": "sac_mlp",
    "sac_mlp_d1_st_rt_late": "sac_mlp",
    "sac_mlp_d1_st_rt_ego": "sac_mlp",
    "sac_mlp_d1_st_rt_edge": "sac_mlp",
    "sac_mlp_d1_st_rt_topo": "sac_mlp",
    "sac_mlp_d1_st_rt_topo_3slot": "sac_mlp",
    "sac_mlp_d1_full": "sac_mlp",
    "hsac_mlp_base_d1_st": "hsac_mlp_base",
    "hsac_mlp_base_d1_st_rt": "hsac_mlp_base",
    "hsac_mlp_base_d1_st_rt_topo": "hsac_mlp_base",
    "hsac_mlp_base_d1_full": "hsac_mlp_base",
}
METHODS = tuple(PARENT)

# 每个方法的三个表征开关（时空交互恒开，见 IncrementalTopoEncoder）。
D1_CONFIG = {
    "sac_mlp_d1_st": dict(use_route=False, use_topology=False, use_slots=False),
    "sac_mlp_d1_st_attn": dict(use_route=False, use_topology=False, use_slots=False),
    "sac_mlp_d1_st_rt": dict(use_route=True, use_topology=False, use_slots=False),
    "sac_mlp_d1_st_rt_gate": dict(use_route=True, use_topology=False, use_slots=False, route_gate=True),
    "sac_mlp_d1_st_rt_late": dict(use_route=True, use_topology=False, use_slots=False, route_late=True),
    "sac_mlp_d1_st_rt_ego": dict(use_route=True, use_topology=False, use_slots=False, route_ego_only=True),
    "sac_mlp_d1_st_rt_edge": dict(use_route=True, use_topology=False, use_slots=False, route_cond_edge=True),
    "sac_mlp_d1_st_rt_topo": dict(use_route=True, use_topology=True, use_slots=False),
    "sac_mlp_d1_st_rt_topo_3slot": dict(
        use_route=True,
        use_topology=True,
        use_slots=True,
        use_incremental_slots=True,
        use_graph_slt=False,
        use_sbs=False,
        representation_coef=0.0,
        slot_balance_coef=0.0,
    ),
    "sac_mlp_d1_full": dict(use_route=True, use_topology=True, use_slots=True),
    "hsac_mlp_base_d1_st": dict(use_route=False, use_topology=False, use_slots=False),
    "hsac_mlp_base_d1_st_rt": dict(use_route=True, use_topology=False, use_slots=False),
    "hsac_mlp_base_d1_st_rt_topo": dict(use_route=True, use_topology=True, use_slots=False),
    "hsac_mlp_base_d1_full": dict(use_route=True, use_topology=True, use_slots=True),
}

# Keep the legacy full method's learned objective unchanged while representing
# slots, Graph-SLT and SBS as separate switches for new controlled comparisons.
for _d1_cfg in D1_CONFIG.values():
    _d1_cfg.setdefault("use_incremental_slots", False)
    _d1_cfg.setdefault("use_graph_slt", bool(_d1_cfg["use_slots"]))
    _d1_cfg.setdefault("use_sbs", bool(_d1_cfg["use_slots"]))
    _d1_cfg.setdefault(
        "representation_coef", 1.0 if _d1_cfg["use_graph_slt"] else 0.0
    )
    _d1_cfg.setdefault(
        "slot_balance_coef", 0.01 if _d1_cfg["use_sbs"] else 0.0
    )

DEFAULT_STEPS = base.RAW_TRAINING_STEPS  # 50_000 满预算
EVAL_EPISODES_TOTAL = base.EVAL_EPISODES_TOTAL
DEFAULT_SMOKE_EVAL_EPISODES = 8
SMOKE_EVAL_EPISODES = DEFAULT_SMOKE_EVAL_EPISODES


def _env_adapter_d1(method: str) -> str:
    """D1 方法的环境契约：继承父基线（sac_mlp=base，hsac_mlp_base=v4_base）。"""
    parent = PARENT[method]
    if parent == "hsac_mlp_base":
        return "v4_base"
    return "base"


def _build_topology_graph(env):
    """从环境 specification 构建 M1 静态车道图（所有 D1 臂都需传入编码器）。"""
    from envs.sumo.topology_graph import MAX_TOPO_EDGES, MAX_TOPO_NODES
    from envs.sumo.topology_graph_v2 import build_topology_graph_v2

    raw_env = env.unwrapped
    specification = getattr(raw_env, "specification", None)
    if specification is None or not hasattr(specification, "network_path"):
        raise TypeError("D1 topology method requires specification.network_path")
    max_nodes = int(getattr(specification, "topology_max_nodes", MAX_TOPO_NODES))
    max_edges = int(getattr(specification, "topology_max_edges", MAX_TOPO_EDGES))
    topology_graph, _ = build_topology_graph_v2(
        specification.network_path,
        max_nodes=max_nodes,
        max_edges=max_edges,
        coordinate_offset=tuple(
            getattr(specification, "coordinate_offset", (0.0, 0.0))
        ),
        return_info=True,
    )
    return topology_graph


def _build_model_d1(method: str, env, *, learning_starts: int):
    """按 method 构造 D1 表征消融模型。

    复用父基线的动作头（连续头 SceneRepSACPolicy / 混合头 DecisionAlignedSACPolicy）
    与全部 SAC 调参，仅把 features_extractor 换成 IncrementalTopoEncoder 并叠加对应
    开关。Graph-SLT/SBS 切换 SACV2；单独三槽输出仍用普通 SceneRepresentationSAC。
    """
    import torch

    from algos.sb3_torch import DictNStepReplayBuffer, SceneRepSACPolicy
    from algos.sb3_torch.hybrid_policy_v4 import DecisionAlignedSACPolicy
    from algos.sb3_torch.incremental_topo_encoder import IncrementalTopoEncoder
    from algos.sb3_torch.incremental_topo_encoder_attn import IncrementalTopoEncoderAttn
    from algos.sb3_torch.sac import SceneRepresentationSAC
    from algos.sb3_torch.sac_v2 import SceneRepresentationSACV2

    parent = PARENT[method]
    cfg = D1_CONFIG[method]
    learning_rate = base.LEARNING_RATE[parent]

    topology_graph = _build_topology_graph(env)
    feature_kwargs = {
        "topology_graph": topology_graph,
        "use_route": cfg["use_route"],
        "use_topology": cfg["use_topology"],
        "use_slots": cfg["use_slots"],
        "use_incremental_slots": cfg["use_incremental_slots"],
        "route_gate": cfg.get("route_gate", False),
        "route_late": cfg.get("route_late", False),
        "route_ego_only": cfg.get("route_ego_only", False),
        "route_cond_edge": cfg.get("route_cond_edge", False),
        "features_dim": 128,
        "hidden_dim": 128,
        "num_heads": 2,
        "random_augmentation": SCENARIO != "cross",
        "carla_contract": SCENARIO == "carla",
    }
    extractor_cls = (
        IncrementalTopoEncoderAttn if method.endswith("_attn") else IncrementalTopoEncoder
    )
    policy_kwargs = {
        "features_extractor_class": extractor_cls,
        "features_extractor_kwargs": feature_kwargs,
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False,
        "net_arch": {"pi": [128, 32], "qf": [128, 32]},
        "optimizer_class": torch.optim.NAdam,
        "optimizer_kwargs": {"eps": 1e-7},
        "n_critics": 2,
        "action_embedding_dim": 64,
    }

    # 动作头：连续头 vs 混合头（与父基线一致）。
    if parent == "hsac_mlp_base":
        policy_class = DecisionAlignedSACPolicy
    else:
        policy_class = SceneRepSACPolicy

    # SBS currently augments Graph-SLT in SACV2; reject unsupported SBS-only configs.
    use_graph_slt = bool(cfg["use_graph_slt"])
    use_sbs = bool(cfg["use_sbs"])
    if use_sbs and not use_graph_slt:
        raise ValueError("SBS currently requires Graph-SLT in SceneRepresentationSACV2")
    algorithm_class = (
        SceneRepresentationSACV2 if (use_graph_slt or use_sbs)
        else SceneRepresentationSAC
    )

    common_kwargs = dict(
        learning_rate=learning_rate,
        buffer_size=base.BUFFER_SIZE,
        learning_starts=0,
        batch_size=base.BATCH_SIZE,
        tau=5e-3,
        gamma=base.DISCOUNT,
        train_freq=(1, "step"),
        gradient_steps=base.ACTION_REPEAT,
        n_steps=4,
        replay_buffer_class=DictNStepReplayBuffer,
        replay_buffer_kwargs={
            "n_steps": 4,
            "gamma": base.DISCOUNT,
            "source_action_repeat": base.ACTION_REPEAT,
        },
        ent_coef="auto_0.2",
        target_entropy="auto",
        policy_kwargs=policy_kwargs,
        action_embedding_dim=64,
        max_grad_norm=5.0,
        raw_learning_starts=learning_starts,
        seed=base.SEED,
        device="cuda",
        verbose=0,
    )

    if use_graph_slt or use_sbs:
        # Legacy full path: keep Graph-SLT and SBS coefficients independently explicit.
        common_kwargs.update(
            structured_representation=use_graph_slt,
            representation_coef=float(cfg["representation_coef"]),
            representation_learning_rate=learning_rate,
            representation_heads=2,
            representation_separate_target_projector=False,
            representation_online_target_encoder=(SCENARIO != "cross"),
            slot_balance_coef=float(cfg["slot_balance_coef"]),
        )
    else:
        common_kwargs.update(representation_coef=float(cfg["representation_coef"]))

    return algorithm_class(policy_class, env, **common_kwargs)


def _encoder_optimizer_metadata(model) -> dict:
    """Static evidence about which encoder parameters each SAC optimizer owns."""
    critic = getattr(model, "critic", None)
    actor = getattr(model, "actor", None)
    extractor = getattr(critic, "features_extractor", None)
    actor_extractor = getattr(actor, "features_extractor", None)
    if extractor is None:
        return {"available": False, "reason": "critic feature extractor unavailable"}

    named = list(extractor.named_parameters())
    optimizer = getattr(critic, "optimizer", None)
    optimizer_ids = {
        id(parameter)
        for group in getattr(optimizer, "param_groups", [])
        for parameter in group.get("params", [])
    }

    def summarize(items):
        items = list(items)
        total = sum(parameter.numel() for _, parameter in items)
        trainable = sum(
            parameter.numel() for _, parameter in items if parameter.requires_grad
        )
        owned = sum(
            parameter.numel() for _, parameter in items if id(parameter) in optimizer_ids
        )
        return {
            "parameter_tensors": len(items),
            "parameter_elements": total,
            "trainable_elements": trainable,
            "critic_optimizer_elements": owned,
            "critic_optimizer_coverage_fraction": owned / total if total else None,
        }

    groups = {}
    group_factory = getattr(extractor, "diagnostic_parameter_groups", None)
    if callable(group_factory):
        for label, prefixes in group_factory().items():
            prefixes = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
            selected = [
                (name, parameter)
                for name, parameter in named
                if any(
                    prefix == ""
                    or name == prefix
                    or name.startswith(prefix + ".")
                    for prefix in prefixes
                )
            ]
            groups[str(label)] = summarize(selected)

    actor_is_detached = type(actor).__name__ == "DetachedSceneActor"
    critic_optimizer_elements = sum(
        parameter.numel()
        for group in getattr(optimizer, "param_groups", [])
        for parameter in group.get("params", [])
    )
    return {
        "available": True,
        "critic_extractor_class": type(extractor).__name__,
        "critic_extractor": summarize(named),
        "critic_parameter_groups": groups,
        "critic_optimizer_parameter_elements": critic_optimizer_elements,
        "actor_extractor_class": (
            type(actor_extractor).__name__ if actor_extractor is not None else None
        ),
        "actor_and_critic_share_extractor_object": (
            actor_extractor is extractor if actor_extractor is not None else None
        ),
        "actor_class": type(actor).__name__ if actor is not None else None,
        "actor_detaches_extracted_features": actor_is_detached,
        "representation_optimizer_present": (
            getattr(model, "representation_optimizer", None) is not None
        ),
    }


def run_training_d1(
    method: str,
    run_dir: Path,
    smoke: bool,
    max_steps: int,
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

    raw_budget = 300 if smoke else max_steps
    frequency = 100 if smoke else base.CHECKPOINT_FREQUENCY
    learning_starts = 60 if smoke else base.LEARNING_STARTS
    overlay_root = run_dir / "overlays"
    namespace = "sm" if smoke else "tr"
    env_adapter = _env_adapter_d1(method)

    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = None
    model = None
    diagnostic_recorder = None

    base._write_json_atomic(
        run_dir / "status.json",
        dict(status="training", pid=os.getpid(), smoke=smoke, started_at=started),
    )

    try:
        env_base = base.make_env_factory(env_adapter, overlay_root / f"ns_{namespace}")(
            base._environment_namespace(), evaluation=False
        )
        if behavior_diagnostics:
            env_base, diagnostic_recorder = base._wrap_behavior_diagnostics_env(
                env_base,
                run_dir,
                "train",
                method,
                raw_step_budget=raw_budget,
                warmup_raw_steps=learning_starts,
                checkpoint_frequency_raw_steps=frequency,
                batch_size=base.BATCH_SIZE,
                learning_rate=base.LEARNING_RATE[PARENT[method]],
                buffer_size=base.BUFFER_SIZE,
                discount=base.DISCOUNT,
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
        model = _build_model_d1(method, env, learning_starts=learning_starts)

        cfg = D1_CONFIG[method]
        if diagnostic_recorder is not None:
            encoder = model.critic.features_extractor
            configure = getattr(encoder, "configure_diagnostics", None)
            if not callable(configure):
                raise TypeError("D1 encoder does not expose configure_diagnostics()")
            configure(True, sample_every=256, source="train_replay")
            diagnostic_recorder.write_run_metadata(
                {
                    "representation_diagnostics": {
                        "enabled": True,
                        "source": "train_replay",
                        "sample_every_forwards": 256,
                        "note": "first enabled forward is sampled; later samples follow the interval",
                    },
                    "representation_method": {
                        "use_route": cfg["use_route"],
                        "use_topology": cfg["use_topology"],
                        "use_slots": cfg["use_slots"],
                        "use_incremental_slots": cfg["use_incremental_slots"],
                        "use_graph_slt": cfg["use_graph_slt"],
                        "use_sbs": cfg["use_sbs"],
                        "representation_coef": cfg["representation_coef"],
                        "slot_balance_coef": cfg["slot_balance_coef"],
                        "algorithm_class": type(model).__name__,
                    },
                    "encoder_optimizer_ownership": _encoder_optimizer_metadata(model),
                    "training_protocol": {
                        "fresh_model": True,
                        "resume": False,
                        "requested_raw_steps": raw_budget,
                        "warmup_raw_steps": learning_starts,
                        "training_seed": base.SEED,
                        "device_requested": "cuda",
                        "training_process_device_policy": "each method runs in its own worker; both may share cuda:0",
                        "evaluation_episodes": SMOKE_EVAL_EPISODES if smoke else EVAL_EPISODES_TOTAL,
                    },
                }
            )
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
                behavior_diagnostics=behavior_diagnostics,
                run_root=str(Path(run_dir).parent.resolve()),
                env_contract=env_adapter,
                reward_shaping=base.REWARD,
                ablation="d1_representation_incremental",
                use_route=cfg["use_route"],
                use_topology=cfg["use_topology"],
                use_slots=cfg["use_slots"],
                use_incremental_slots=cfg["use_incremental_slots"],
                use_graph_slt=cfg["use_graph_slt"],
                use_sbs=cfg["use_sbs"],
                structured_representation=cfg["use_graph_slt"],
                representation_coef=cfg["representation_coef"],
                slot_balance_coef=cfg["slot_balance_coef"],
                algorithm_class=type(model).__name__,
                note=(
                    "D1 IncrementalTopoEncoder config; SAC/action head and optimizer settings "
                    f"inherit parent {PARENT[method]}. Separate switches: "
                    f"route={cfg['use_route']}, topology={cfg['use_topology']}, "
                    f"slots={cfg['use_slots']}, incremental_slots={cfg['use_incremental_slots']}, "
                    f"Graph-SLT={cfg['use_graph_slt']}, SBS={cfg['use_sbs']}."
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

        class ModuleDiagnosticCallback(BaseCallback):
            def _drain(self):
                pop_events = getattr(self.model, "pop_module_diagnostic_events", None)
                if not callable(pop_events):
                    return
                for event in pop_events():
                    diagnostic_recorder.record_representation(
                        raw_steps=event.get("raw_steps", getattr(self.model, "_raw_steps_seen", 0)),
                        decision_steps=event.get("decision_steps", self.model.num_timesteps),
                        metrics=event.get("metrics", {}),
                        source=event.get("source", "unspecified"),
                        episode=None,
                        updates=event.get("updates"),
                        timing=event.get("timing"),
                    )

            def _on_step(self):
                self._drain()
                return True

            def _on_training_end(self):
                self._drain()

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
            training_callbacks.append(base._make_behavior_optimization_callback(diagnostic_recorder))
            training_callbacks.append(ModuleDiagnosticCallback())

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
        base._write_json_atomic(
            run_dir / "training_complete.json",
            dict(
                smoke=smoke,
                checkpoint_sha256=base._sha256(final),
                raw_steps=raw_budget,
                updates=model._n_updates,
                replay_size=model.replay_buffer.size(),
                behavior_diagnostics=behavior_diagnostics,
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


def _evaluate_saved(
    method: str,
    model_path: Path,
    run_dir: Path,
    n_ep: int,
    behavior_diagnostics: bool = False,
) -> dict:
    """进程内评估（对齐父基线部署语义：连续头直接 load，混合头 + use_actor_only）。"""
    import torch

    from algos.sb3_torch.evaluation import evaluate_model_detailed
    from algos.sb3_torch.sac import SceneRepresentationSAC
    from algos.sb3_torch.sac_v2 import SceneRepresentationSACV2

    torch.set_num_threads(1)
    parent = PARENT[method]
    cfg = D1_CONFIG[method]
    use_graph_slt = bool(cfg["use_graph_slt"])
    use_sbs = bool(cfg["use_sbs"])
    env_adapter = _env_adapter_d1(method)
    # 评估 overlay 目录保持极短（"oe"）：续训时 run_dir = <run>/c100000 多一层，
    # 长方法名（如 sac_mlp_d1_st_rt_late）会把 manifest 路径顶到 264 > MAX_PATH(260)，
    # os.replace 报 WinError 3。原 "overlays/eval"(12 字符) 减到 "oe"(2 字符) 后降到 254。
    overlay_root = run_dir / "oe"
    env = base.make_env_factory(env_adapter, overlay_root)(
        base._environment_namespace(), evaluation=True
    )
    if behavior_diagnostics:
        seed_start = base.SEED_START[parent]
        env, eval_recorder = base._wrap_behavior_diagnostics_env(
            env,
            run_dir,
            "eval",
            method,
            env_contract=env_adapter,
            evaluation_seed_start=seed_start,
            evaluation_episode_start=0,
            evaluation_episode_end=n_ep,
            evaluation_episodes=n_ep,
            deterministic=True,
            checkpoint=str(Path(model_path).resolve()),
            device="cpu",
            smoke=n_ep == SMOKE_EVAL_EPISODES,
        )
    load_cls = (
        SceneRepresentationSACV2 if (use_graph_slt or use_sbs)
        else SceneRepresentationSAC
    )
    loaded = load_cls.load(str(model_path), env=env, device="cpu", buffer_size=32)
    if parent == "hsac_mlp_base":
        from tools.v48_stability_v4.model import use_actor_only

        model = use_actor_only(loaded)
    else:  # sac_mlp（连续头，标准 SAC 前向，无融合门控）
        model = loaded

    representation_extractor = None
    if behavior_diagnostics:
        representation_extractor = getattr(
            getattr(model, "actor", None), "features_extractor", None
        )
        if representation_extractor is None:
            representation_extractor = getattr(
                getattr(model, "critic", None), "features_extractor", None
            )
        configure = getattr(representation_extractor, "configure_diagnostics", None)
        if not callable(configure):
            raise TypeError("D1 evaluation encoder does not expose configure_diagnostics()")
        configure(True, sample_every=1, source="eval_policy")
        if eval_recorder is not None:
            eval_recorder.write_run_metadata(
                {
                    "representation_diagnostics": {
                        "enabled": True,
                        "source": "eval_policy",
                        "sample_every_forwards": 1,
                        "sample_timing": "after existing model.predict and before env.step",
                        "training_checkpoint": str(Path(model_path).resolve()),
                    },
                    "representation_method": {
                        "use_route": cfg["use_route"],
                        "use_topology": cfg["use_topology"],
                        "use_slots": cfg["use_slots"],
                        "use_incremental_slots": cfg["use_incremental_slots"],
                        "use_graph_slt": cfg["use_graph_slt"],
                        "use_sbs": cfg["use_sbs"],
                        "representation_coef": cfg["representation_coef"],
                        "slot_balance_coef": cfg["slot_balance_coef"],
                        "algorithm_class": type(model).__name__,
                    },
                }
            )

    class RepresentationDiagnosticPredictor:
        """Transparent evaluation proxy: record existing policy forwards only."""
        def __init__(self, wrapped_model, encoder, recorder):
            self.wrapped_model = wrapped_model
            self.encoder = encoder
            self.recorder = recorder
            self.last_sample_index = 0

        def __getattr__(self, name):
            return getattr(self.wrapped_model, name)

        def predict(self, observation, *args, **kwargs):
            result = self.wrapped_model.predict(observation, *args, **kwargs)
            if self.encoder is None or self.recorder is None:
                return result
            values = self.encoder.diagnostic_values()
            metrics = {}
            for key, value in values.items():
                if value is None:
                    metrics[str(key)] = None
                    continue
                if hasattr(value, "numel") and value.numel() != 1:
                    continue
                if hasattr(value, "detach"):
                    value = value.detach().cpu().item()
                if isinstance(value, (int, float, np.number)):
                    numeric = float(value)
                    metrics[str(key)] = numeric if np.isfinite(numeric) else None
            sample_index = metrics.get("diagnostic_sample_index")
            if sample_index is None:
                return result
            sample_index = int(sample_index)
            if sample_index <= self.last_sample_index:
                return result
            self.last_sample_index = sample_index
            self.recorder.record_representation(
                raw_steps=self.recorder.raw_count,
                decision_steps=self.recorder.decision_count,
                metrics=metrics,
                source="eval_policy",
                episode=self.recorder.episode_index,
                timing="captured_after_existing_predict_before_env_step; sample_index-deduplicated",
            )
            return result

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
            evaluator_model = (
                RepresentationDiagnosticPredictor(
                    model, representation_extractor, eval_recorder
                )
                if behavior_diagnostics
                else model
            )
            detailed = evaluate_model_detailed(
                evaluator_model,
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
    summary = dict(
        episodes=n,
        success_rate=sum(int(r["success"]) for r in records) / n if n else 0.0,
        collision_rate=sum(int(r["collision"]) for r in records) / n if n else 0.0,
        off_route_rate=sum(int(r["off_route"]) for r in records) / n if n else 0.0,
        timeout_rate=sum(int(r["timeout"]) for r in records) / n if n else 0.0,
        mean_return=float(np.mean([r["episode_return"] for r in records])) if n else None,
    )
    return summary, records


# --------------------------------------------------------------------------- #
# 方法调度：全局唯一方法名 -> (家族, 模块, 模块内部方法名)
# --------------------------------------------------------------------------- #
# 家族：yv2 = yield_v2 降密度；legacy = 原始密度；d1 = D1 表征消融（本文件）。
# d1 家族模块记 None（训练/评估都在本文件内），其余指向对应脚本模块。
DISPATCH: dict[str, tuple[str, object, str]] = {
    "hold35k": ("yv2", base, "hold35k"),
    "mst_slt": ("yv2", base, "mst_slt"),
    "hsac_mlp": ("yv2", base, "hsac_mlp"),
    "sac_mlp": ("yv2", base, "sac_mlp"),
    "sac_mlp_v48": ("yv2", base, "sac_mlp_v48"),
    "hsac_mlp_base": ("yv2", base, "hsac_mlp_base"),
    "hold35k_legacy": ("legacy", legacy, "hold35k"),
    "mst_slt_legacy": ("legacy", legacy, "mst_slt"),
}
DISPATCH.update({m: ("d1", None, m) for m in PARENT})
ALL_METHODS = tuple(DISPATCH)

# run_full（launcher）默认训练的方法：MST+SLT 与 sac_mlp_d1_st_rt，两 CUDA worker 并行。
DEFAULT_METHODS: tuple[str, ...] = ("mst_slt", "sac_mlp_d1_st_rt")
TRAIN_WORKERS = 2


def _apply_patch(
    depart_scale: float,
    scenario: str = "intersection_sorted",
    eval_traffic_split: str = "validation",
) -> None:
    """把各脚本的 SCENARIO（及 yield_v2 的 traffic 源 / depart 缩放）指向本实验。

    只在当前进程内覆盖模块常量：所有复用函数运行时经模块全局查找 SCENARIO /
    SOURCE_TRAFFIC / DEPART_SCALE，因此覆盖即刻生效。legacy 家族经 registry 按
    SCENARIO 名加载 traffic（原始密度），无 SOURCE_TRAFFIC / DEPART_SCALE 常量，
    故只 patch 其 SCENARIO。d1 家族的 SCENARIO 是 import 时拷贝的模块级副本，须单独
    同步（_build_model_d1 用它判断 random_augmentation / carla_contract）。
    """
    global DEPART_SCALE, SCENARIO, NEW_SCENARIO, NEW_SOURCE_TRAFFIC
    global EVAL_TRAFFIC_SPLIT
    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(f"unsupported scenario {scenario!r}; choose from {SUPPORTED_SCENARIOS}")
    if eval_traffic_split not in ("validation", "test"):
        raise ValueError("eval traffic split must be 'validation' or 'test'")
    random_profile = is_random_intersection_scenario(scenario)
    if random_profile:
        if float(depart_scale) != 1.0:
            raise ValueError("random intersection flows require depart_scale=1.0")
        ensure_random_intersection_assets(scenario)
        source_traffic = RANDOM_INTERSECTION_ASSET_ROOT / scenario / "traffic"
    else:
        if eval_traffic_split != "validation":
            raise ValueError("--eval-traffic-split test applies only to random traffic profiles")
        source_traffic = (
            PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1"
            / scenario / "traffic"
        )
    DEPART_SCALE = depart_scale
    NEW_SCENARIO = scenario
    NEW_SOURCE_TRAFFIC = source_traffic
    SCENARIO = scenario
    EVAL_TRAFFIC_SPLIT = eval_traffic_split
    for mod in (base, legacy):
        mod.SCENARIO = scenario
    base.SOURCE_TRAFFIC = NEW_SOURCE_TRAFFIC
    base.DEPART_SCALE = depart_scale
    base.EVAL_TRAFFIC_SPLIT = eval_traffic_split

    # The legacy training module owns a separate environment factory. Adapt
    # only newly randomized scenarios so eval gets its requested disjoint seed
    # domain; restore its original factory unchanged for the sorted baseline.
    if random_profile:
        def legacy_factory(adapter: str, overlay_root: Path):
            make_base_environment = _LEGACY_MAKE_ENV_FACTORY(adapter, overlay_root)

            def make_environment(args, *, evaluation=False):
                # The shared high-density factory reserves the contract label
                # "validation" and rejects its own test partition. Random
                # traffic uses PaperSumoSceneEnv's independent SUMO seed domain.
                args.evaluation_split = "validation"
                env = make_base_environment(args, evaluation=evaluation)
                underlying = getattr(env, "unwrapped", env)
                setter = getattr(underlying, "set_traffic_split", None)
                if not callable(setter):
                    raise TypeError(
                        "Random intersection environment must expose set_traffic_split()"
                    )
                setter(eval_traffic_split if evaluation else "train")
                return env

            return make_environment

        legacy.make_env_factory = legacy_factory
    else:
        legacy.make_env_factory = _LEGACY_MAKE_ENV_FACTORY


# --------------------------------------------------------------------------- #
# depart 排序版副本场景生成（幂等，与 train_intersection_sorted.py 同机制）
# --------------------------------------------------------------------------- #
def _sort_traffic_file(src: Path, dst: Path) -> None:
    """按 (depart, departLane) 升序重排一个 traffic 文件的 <vehicle>。

    vType 保持在前（SUMO 要求类型先于车辆定义），vehicle 顺序调整后写入 dst。
    """
    tree = ET.parse(src)
    root = tree.getroot()
    vtypes = [child for child in list(root) if child.tag == "vType"]
    vehicles = [child for child in list(root) if child.tag == "vehicle"]
    vehicles.sort(
        key=lambda v: (
            float(v.attrib.get("depart", "0")),
            int(v.attrib.get("departLane", "0")),
        )
    )
    new_root = ET.Element("routes")
    for vt in vtypes:
        new_root.append(vt)
    for v in vehicles:
        new_root.append(v)
    ET.ElementTree(new_root).write(dst, encoding="UTF-8", xml_declaration=True)


def ensure_sorted_scenario() -> Path:
    """生成（或复用已生成的）``intersection_sorted`` 副本场景，返回其目录。"""
    src = PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1" / "intersection"
    dst = PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1" / "intersection_sorted"
    marker = dst / ".depart_sorted"
    if marker.is_file():
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    for name in ("map.net.xml", "ego.rou.xml"):
        shutil.copy2(src / name, dst / name)

    traffic_src = src / "traffic"
    traffic_dst = dst / "traffic"
    traffic_dst.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(traffic_src.glob("traffic_*.rou.xml")):
        _sort_traffic_file(src_file, traffic_dst / src_file.name)

    # base SumoSceneEnv 构造时 get_scenario_spec 会检查 scenarios/<name>/
    # scenario.sumocfg 存在性；复制 intersection 的占位即可。
    base_src = PROJECT_ROOT / "envs" / "sumo" / "scenarios" / "intersection"
    base_dst = PROJECT_ROOT / "envs" / "sumo" / "scenarios" / "intersection_sorted"
    base_dst.mkdir(parents=True, exist_ok=True)
    for name in ("scenario.sumocfg", "routes.rou.xml"):
        candidate = base_src / name
        if candidate.is_file():
            shutil.copy2(candidate, base_dst / name)

    marker.write_text(
        "depart-sorted traffic (three flows all active)\n", encoding="utf-8"
    )
    return dst


# --------------------------------------------------------------------------- #
# 训练 + 评估编排
# --------------------------------------------------------------------------- #
def _depart_token(scale: float) -> str:
    return str(scale).replace(".", "p")


def _run_dir(global_name: str, scale: float) -> Path:
    """run 目录带 depart 缩放后缀，避免不同密度档 / 不同方法互相覆盖。"""
    return base.RESULT_ROOT / f"{global_name}__{NEW_SCENARIO}_depart{_depart_token(scale)}"


def _train(
    global_name: str,
    run_dir: Path,
    smoke: bool,
    behavior_diagnostics: bool = False,
) -> Path:
    """复用底层训练函数（当前进程内训练，返回 final_model 路径）。"""
    family, mod, method = DISPATCH[global_name]
    if family == "d1":
        return run_training_d1(
            method,
            run_dir,
            smoke=smoke,
            max_steps=MAX_STEPS,
            behavior_diagnostics=behavior_diagnostics,
        )
    if method == "hold35k":
        return mod.run_training_hold35k(run_dir, smoke=smoke, max_steps=MAX_STEPS)
    if method == "mst_slt":
        return mod.run_training_mst_slt(run_dir, smoke=smoke, max_steps=MAX_STEPS)
    # 其余为 yield_v2 的 MLP 方法（hsac_mlp / sac_mlp / sac_mlp_v48 / hsac_mlp_base）
    return mod.run_training_mlp(
        method,
        run_dir,
        smoke=smoke,
        max_steps=MAX_STEPS,
        behavior_diagnostics=behavior_diagnostics,
    )


def _evaluate_d1(
    global_name: str,
    run_dir: Path,
    final_model: Path,
    smoke: bool,
    behavior_diagnostics: bool = False,
) -> dict:
    """d1 家族进程内评估（无 run_eval_worker，用 _evaluate_saved）。"""
    family, mod, method = DISPATCH[global_name]
    n_ep = SMOKE_EVAL_EPISODES if smoke else EVAL_EPISODES_TOTAL
    summary, records = _evaluate_saved(
        method, final_model, run_dir, n_ep,
        behavior_diagnostics=behavior_diagnostics,
    )
    result = dict(
        identity=dict(
            method=global_name,
            source_script="train_intersection_yield_v2_d1",
            source_method=method,
            family=family,
            scenario=NEW_SCENARIO,
            depart_scale=DEPART_SCALE,
            eval_traffic_split=EVAL_TRAFFIC_SPLIT,
            checkpoint=str(final_model),
            checkpoint_sha256=base._sha256(final_model),
            episodes=n_ep,
            workers=1,
            smoke=smoke,
            behavior_diagnostics=behavior_diagnostics,
        ),
        summary=summary,
        episode_records=records,
    )
    base._write_json_atomic(run_dir / "evaluation_results.json", result)
    return result


def evaluate(
    global_name: str,
    run_dir: Path,
    final_model: Path,
    smoke: bool,
    behavior_diagnostics: bool = False,
) -> dict:
    """d1 家族进程内评估；yv2 / legacy 家族 subprocess 并行评估。

    subprocess 隔离让每个 worker 退出时自动回收 SUMO 子进程，避免进程内线程并行时
    SUMO 进程残留锁住 traffic 文件。eval-worker 子进程带 ``--depart-scale``，由其
    main 分支 ``_apply_patch`` 恢复场景与发车间隔后调用底层 ``run_eval_worker``。
    """
    family, mod, method = DISPATCH[global_name]
    if family == "d1":
        return _evaluate_d1(
            global_name,
            run_dir,
            final_model,
            smoke,
            behavior_diagnostics=behavior_diagnostics,
        )

    workers = 1 if behavior_diagnostics else (2 if smoke else mod.EVAL_WORKERS)
    episodes = SMOKE_EVAL_EPISODES if smoke else mod.EVAL_EPISODES_TOTAL

    ranges: list[tuple[int, int]] = []
    for worker_id in range(workers):
        start = worker_id * episodes // workers
        end = (worker_id + 1) * episodes // workers
        if end > start:
            ranges.append((start, end))

    script = str(Path(__file__).resolve())
    procs: list[tuple[int, subprocess.Popen, object]] = []
    for wid, (start, end) in enumerate(ranges):
        cmd = [
            sys.executable, script, "eval-worker",
            "--method", global_name,
            "--scenario", NEW_SCENARIO,
            "--depart-scale", str(DEPART_SCALE),
            "--eval-traffic-split", EVAL_TRAFFIC_SPLIT,
            "--worker-id", str(wid),
            "--start-episode", str(start),
            "--end-episode", str(end),
            "--model-path", str(final_model),
            "--output-dir", str(run_dir),
            "--run-root", str(base.RESULT_ROOT.resolve()),
        ]
        if behavior_diagnostics:
            cmd.append("--behavior-diagnostics")
        handle = (run_dir / f"eval_worker_{wid:02d}.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        procs.append((wid, proc, handle))
    for wid, proc, handle in procs:
        code = proc.wait()
        handle.close()
        if code != 0:
            raise RuntimeError(
                f"eval worker {wid} ({global_name}) failed: exit {code}; "
                f"see {run_dir}/eval_worker_{wid:02d}.log"
            )

    records: list[dict] = []
    for wid in range(len(ranges)):
        path = run_dir / f"eval_worker_{wid:02d}.json"
        records.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    records.sort(key=lambda r: r["episode"])
    summary = mod.summarize_records(records)

    result = dict(
        identity=dict(
            method=global_name,
            source_script=mod.__name__,
            source_method=method,
            family=family,
            scenario=NEW_SCENARIO,
            depart_scale=DEPART_SCALE,
            eval_traffic_split=EVAL_TRAFFIC_SPLIT,
            checkpoint=str(final_model),
            checkpoint_sha256=mod._sha256(final_model),
            episodes=episodes,
            workers=workers,
            episodes_per_worker=[end - start for start, end in ranges],
            smoke=smoke,
            behavior_diagnostics=behavior_diagnostics,
        ),
        summary=summary,
        episode_records=records,
    )
    mod._write_json_atomic(run_dir / "evaluation_results.json", result)
    return result


def _ratio_or_unknown(numerator, denominator):
    if denominator is None or denominator <= 0 or numerator is None:
        return None
    return float(numerator) / float(denominator)


def _diagnostic_behavior_metrics(summary: dict, manifest: dict) -> dict:
    """Pool raw/decision counters with their valid denominators; never infer zero."""
    episodes = summary.get("episode_summaries") or []
    sum_keys = (
            "speed_samples", "speed_sum_mps", "stopped_ticks",
            "risk_evaluable_ticks", "risk_evaluable_low_ttc_ticks",
            "risk_and_observation_covered_ticks", "covered_critical_unobserved_ticks",
    )
    sums = {}
    for key in sum_keys:
        values = [ep.get(key) for ep in episodes]
        sums[key] = (
            sum(float(value) for value in values)
            if episodes and all(value is not None for value in values)
            else None
        )
    action_n = action_near = 0
    action_fields_complete = bool(episodes)
    for episode in episodes:
        if "action_statistics" not in episode or episode["action_statistics"] is None:
            action_fields_complete = False
            continue
        for action in episode["action_statistics"]:
            if "n" not in action or "abs_ge_0p95_count" not in action:
                action_fields_complete = False
                continue
            action_n += int(action.get("n") or 0)
            action_near += int(action.get("abs_ge_0p95_count") or 0)
    if not action_fields_complete:
        action_n = action_near = None

    raw_records = summary.get("raw_records")
    metrics = {
        "mean_actual_speed_mps": _ratio_or_unknown(sums["speed_sum_mps"], sums["speed_samples"]),
        "mean_actual_speed_denominator_speed_samples": sums["speed_samples"],
        "stopped_fraction": _ratio_or_unknown(sums["stopped_ticks"], sums["speed_samples"]),
        "stopped_fraction_denominator_speed_samples": sums["speed_samples"],
        "near_unit_bound_fraction": _ratio_or_unknown(action_near, action_n),
        "near_unit_bound_numerator_abs_action_ge_0p95": action_near,
        "near_unit_bound_denominator_action_scalar_samples": action_n,
        "near_unit_bound_action_space": manifest.get("action_space"),
        "low_ttc_fraction": _ratio_or_unknown(
            sums["risk_evaluable_low_ttc_ticks"], sums["risk_evaluable_ticks"]
        ),
        "low_ttc_numerator_evaluable_ticks_ttc_lt_3s": sums["risk_evaluable_low_ttc_ticks"],
        "low_ttc_denominator_risk_evaluable_ticks": sums["risk_evaluable_ticks"],
        "covered_critical_unobserved_fraction": _ratio_or_unknown(
            sums["covered_critical_unobserved_ticks"],
            sums["risk_and_observation_covered_ticks"],
        ),
        "covered_critical_unobserved_numerator_ticks": sums["covered_critical_unobserved_ticks"],
        "covered_critical_unobserved_denominator_ticks": sums["risk_and_observation_covered_ticks"],
        "coverage_speed_samples_of_raw_records": _ratio_or_unknown(
            sums["speed_samples"], raw_records
        ),
        "coverage_risk_evaluable_ticks_of_raw_records": _ratio_or_unknown(
            sums["risk_evaluable_ticks"], raw_records
        ),
        "coverage_risk_and_observation_ticks_of_raw_records": _ratio_or_unknown(
            sums["risk_and_observation_covered_ticks"], raw_records
        ),
        "raw_records": raw_records,
        "decision_records": summary.get("decision_records"),
        "diagnostic_error_count": summary.get("diagnostic_error_count"),
        "episodes_finished": summary.get("episodes_finished"),
    }
    null_reasons = {}
    for field in (
        "mean_actual_speed_mps", "stopped_fraction", "near_unit_bound_fraction",
        "low_ttc_fraction", "covered_critical_unobserved_fraction",
    ):
        if metrics[field] is None:
            null_reasons[field] = "no valid denominator or source samples were recorded"
    metrics["null_reasons"] = null_reasons
    return metrics


def write_behavior_comparison(run_root: Path, methods: tuple[str, ...]) -> dict:
    """Summarize existing train/eval artifacts; this does not replay an environment."""
    run_root = Path(run_root)
    root_manifest_path = run_root / "experiment_manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    method_results = {}
    markdown_rows = []
    for method in methods:
        run_dir = _run_dir(method, root_manifest.get("depart_scale", DEPART_SCALE))
        # A custom run root is the only supported source for launcher outputs.
        run_dir = run_root / run_dir.name
        result_path = run_dir / "evaluation_results.json"
        train_complete_path = run_dir / "training_complete.json"
        run_manifest_path = run_dir / "experiment_manifest.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        train_complete = json.loads(train_complete_path.read_text(encoding="utf-8")) if train_complete_path.exists() else {}
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.exists() else {}
        family, _, _ = DISPATCH[method]
        eval_phase = "eval_worker_00" if family != "d1" else "eval"
        train_summary_path = run_dir / "diagnostics" / "train" / "summary.json"
        train_diag_manifest_path = run_dir / "diagnostics" / "train" / "manifest.json"
        eval_summary_path = run_dir / "diagnostics" / eval_phase / "summary.json"
        eval_diag_manifest_path = run_dir / "diagnostics" / eval_phase / "manifest.json"
        train_summary = json.loads(train_summary_path.read_text(encoding="utf-8")) if train_summary_path.exists() else {}
        train_diag_manifest = json.loads(train_diag_manifest_path.read_text(encoding="utf-8")) if train_diag_manifest_path.exists() else {}
        eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8")) if eval_summary_path.exists() else {}
        eval_diag_manifest = json.loads(eval_diag_manifest_path.read_text(encoding="utf-8")) if eval_diag_manifest_path.exists() else {}

        records = result.get("episode_records") or []
        complete_n = len(records)
        outcome_fields = ("success", "collision", "timeout", "off_route")
        outcomes_complete = bool(records) and all(
            key in record and record[key] is not None
            for record in records for key in outcome_fields
        )
        outcome_counts = (
            {key: sum(bool(record[key]) for record in records) for key in outcome_fields}
            if outcomes_complete
            else {key: None for key in outcome_fields}
        )
        eval_diagnostics = _diagnostic_behavior_metrics(eval_summary, eval_diag_manifest)
        train_diagnostics = _diagnostic_behavior_metrics(train_summary, train_diag_manifest)
        train_episodes = train_summary.get("episode_summaries") or []
        training_outcomes = train_summary.get("outcome_counts") or {}
        entry = {
            "run_dir": str(run_dir.resolve()),
            "training": {
                "completed_raw_steps": train_complete.get("raw_steps"),
                "requested_raw_step_budget": run_manifest.get("raw_budget", root_manifest.get("raw_step_budget")),
                "updates": train_complete.get("updates"),
                "smoke": run_manifest.get("smoke", root_manifest.get("smoke")),
                "incomplete_episodes": sum(
                    1 for episode in train_episodes
                    if episode.get("outcome") == "incomplete"
                ),
                "episode_outcome_counts_including_incomplete": training_outcomes,
                "behavior": train_diagnostics,
                "optimization_samples": train_summary.get("optimization_samples"),
                "optimization_statistics": train_summary.get("optimization_statistics"),
            },
            "evaluation": {
                "expected_episodes": (result.get("identity") or {}).get("episodes"),
                "complete_episode_denominator": complete_n,
                "episode_count_matches_expected": (
                    complete_n == (result.get("identity") or {}).get("episodes")
                    if (result.get("identity") or {}).get("episodes") is not None
                    else None
                ),
                "outcome_counts": outcome_counts,
                "outcome_rates": {
                    key: _ratio_or_unknown(count, complete_n)
                    for key, count in outcome_counts.items()
                },
                "behavior": eval_diagnostics,
            },
            "artifacts": {
                "evaluation_results": str(result_path.resolve()) if result_path.exists() else None,
                "training_complete": str(train_complete_path.resolve()) if train_complete_path.exists() else None,
                "train_diagnostics": str(train_summary_path.resolve()) if train_summary_path.exists() else None,
                "eval_diagnostics": str(eval_summary_path.resolve()) if eval_summary_path.exists() else None,
            },
        }
        method_results[method] = entry
        rates = entry["evaluation"]["outcome_rates"]
        markdown_rows.append(
            "| {method} | {steps} | {incomplete} | {n} | {success} | {collision} | {timeout} | {off_route} | {speed} | {stopped} | {near} | {low_ttc} | {unobserved} |".format(
                method=method,
                steps=entry["training"]["completed_raw_steps"],
                incomplete=entry["training"]["incomplete_episodes"],
                n=complete_n,
                success=_format_fraction(rates["success"]),
                collision=_format_fraction(rates["collision"]),
                timeout=_format_fraction(rates["timeout"]),
                off_route=_format_fraction(rates["off_route"]),
                speed=_format_metric(eval_diagnostics["mean_actual_speed_mps"], " m/s"),
                stopped=_format_fraction(eval_diagnostics["stopped_fraction"]),
                near=_format_fraction(eval_diagnostics["near_unit_bound_fraction"]),
                low_ttc=_format_fraction(eval_diagnostics["low_ttc_fraction"]),
                unobserved=_format_fraction(eval_diagnostics["covered_critical_unobserved_fraction"]),
            )
        )

    comparison = {
        "schema_version": "behavior-comparison/v1",
        "run_root": str(run_root.resolve()),
        "protocol": {
            "traffic_protocol": root_manifest.get("traffic_protocol"),
            "scenario": root_manifest.get("scenario"),
            "depart_scale": root_manifest.get("depart_scale"),
            "training_seed": root_manifest.get("training_seed"),
            "raw_step_budget": root_manifest.get("raw_step_budget"),
            "expected_eval_episodes": root_manifest.get("eval_episodes"),
            "eval_outcome_denominator": "returned evaluation episode_records; a missing/short run remains visible via count vs expected and is never recoded as timeout",
            "training_terminal_note": "training episodes closed by run truncation are counted as incomplete, not environment timeouts",
            "continuous_metric_weighting": "raw-tick or action-scalar weighted by each metric's explicit valid sample denominator",
        },
        "methods": method_results,
    }
    base._write_json_atomic(run_root / "comparison.json", comparison)
    lines = [
        "# Run comparison",
        "",
        f"Run root: `{run_root.resolve()}`",
        "",
        "Outcome rates use completed episode records only; denominator is shown as eval n. Training truncation is reported separately as incomplete episodes. Continuous behavior metrics pool valid raw ticks or action scalar samples rather than averaging episodes equally.",
        "",
        "| Method | Train raw steps | Train incomplete episodes | Eval n | Success | Collision | Timeout | Off-route | Mean speed | Stopped | Near-unit-bound | Low TTC | Covered critical unobserved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *markdown_rows,
        "",
        "`near_unit_bound_fraction` is the share of action scalar samples with |a| >= 0.95; action-space bounds and definition are in comparison.json. Low-TTC uses TTC < 3 s among geometry-valid risk-evaluable ticks. Covered critical-unobserved uses ticks with both valid risk and policy-observation coverage. Null means no valid denominator or missing source artifact, not zero.",
        "",
        "Full denominators, coverage, train/eval summaries, optimization statistics, and artifact paths are in `comparison.json`.",
        "",
    ]
    (run_root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return comparison


def _format_fraction(value):
    return "null" if value is None else f"{100.0 * value:.1f}%"


def _format_metric(value, suffix=""):
    return "null" if value is None else f"{value:.3f}{suffix}"


def train_and_eval(
    global_name: str,
    smoke: bool,
    behavior_diagnostics: bool = False,
) -> Path:
    family, mod, method = DISPATCH[global_name]
    run_dir = _run_dir(global_name, DEPART_SCALE)
    if not smoke and run_dir.exists():
        raise RuntimeError(f"run output already exists; refusing to overwrite: {run_dir}")
    # 不预先 mkdir：mst_slt 经 train_sb3.main 要求 run_dir 尚不存在（exist_ok=False），
    # hold35k / mlp / d1 各自 mkdir(exist_ok=True)。
    final = _train(
        global_name, run_dir, smoke=smoke,
        behavior_diagnostics=behavior_diagnostics,
    )
    result = evaluate(
        global_name, run_dir, final, smoke=smoke,
        behavior_diagnostics=behavior_diagnostics,
    )
    curve = None
    if family != "d1":
        curve = mod.plot_training_curves(method, run_dir)

    random_profile = is_random_intersection_scenario(NEW_SCENARIO)
    traffic_config = (
        get_random_intersection_config(NEW_SCENARIO) if random_profile else None
    )
    manifest = dict(
        method=global_name,
        source_script=(
            "train_intersection_yield_v2_d1" if family == "d1" else mod.__name__
        ),
        source_method=method,
        family=family,
        scenario=NEW_SCENARIO,
        depart_scale=DEPART_SCALE,
        eval_traffic_split=EVAL_TRAFFIC_SPLIT,
        traffic_protocol=(
            traffic_config["protocol"] if traffic_config is not None
            else "sorted_explicit_traffic_departures"
        ),
        traffic_config=traffic_config,
        training_traffic_split="train" if random_profile else None,
        run_dir=str(run_dir),
        final_model=str(final),
        evaluation=result["summary"],
        training_curve=str(curve) if curve else None,
        smoke=smoke,
        behavior_diagnostics=behavior_diagnostics,
        raw_budget=(300 if smoke else MAX_STEPS),
        training_seed=base.SEED,
        checkpoint_frequency_raw_steps=(
            100 if smoke else base.CHECKPOINT_FREQUENCY
        ),
        note=(
            "seeded randomized intersection traffic profile."
            if random_profile
            else (
                "depart 排序版 intersection 副本：三股车流（北→南 / 东→西 / 西→东）"
                f"都参与仿真 + 发车间隔 ×{DEPART_SCALE}（降密度）。"
            )
        ),
    )
    base._write_json_atomic(run_dir / "experiment_manifest.json", manifest)
    print(
        json.dumps(
            dict(method=global_name, summary=result["summary"]),
            ensure_ascii=False,
            indent=2,
        )
    )
    return run_dir


def run_full(
    smoke: bool,
    methods: tuple[str, ...],
    depart_scale: float,
    max_steps: int,
    behavior_diagnostics: bool = False,
    reference_run_root: Path | None = None,
) -> Path:
    if not smoke and not base._cuda_available():
        raise RuntimeError("CUDA unavailable; training has no CPU fallback")

    root = base.RESULT_ROOT  # fast-developer
    root.mkdir(parents=True, exist_ok=True)
    if len(set(methods)) != len(methods):
        raise ValueError(f"duplicate methods requested: {methods}")
    unknown = [name for name in methods if name not in DISPATCH]
    if unknown:
        raise ValueError(f"unknown methods requested: {unknown}")
    if not smoke:
        existing = [n for n in methods if _run_dir(n, depart_scale).exists()]
        if existing:
            raise RuntimeError(f"output dirs already exist (remove first): {existing}")

    random_profile = is_random_intersection_scenario(NEW_SCENARIO)
    if random_profile:
        traffic_config = get_random_intersection_config(NEW_SCENARIO)
        effective_traffic = tuple(sorted(NEW_SOURCE_TRAFFIC.glob("traffic_*.rou.xml")))
    else:
        traffic_config = None
        # Prepare the shared scaled route pool once before two child processes start.
        # Each child then reads the same 30-file pool instead of racing to create it.
        effective_traffic = base._traffic_paths_for_scale(depart_scale)

    script = str(Path(__file__).resolve())
    codes: dict[str, int] = {}
    log_dir = root / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_lock = threading.Lock()
    launcher_state = {
        "status": "launching",
        "run_root": str(root.resolve()),
        "source_script": script,
        "scenario": NEW_SCENARIO,
        "depart_scale": depart_scale,
        "eval_traffic_split": EVAL_TRAFFIC_SPLIT,
        "raw_step_budget": 300 if smoke else max_steps,
        "smoke": smoke,
        "training_seed": base.SEED,
        "checkpoint_frequency_raw_steps": 100 if smoke else base.CHECKPOINT_FREQUENCY,
        "eval_episodes": SMOKE_EVAL_EPISODES if smoke else base.EVAL_EPISODES_TOTAL,
        "behavior_diagnostics": behavior_diagnostics,
        "reference_run_root": (
            str(Path(reference_run_root).resolve())
            if reference_run_root is not None else None
        ),
        "method_configs": {
            method: D1_CONFIG[method]
            for method in methods if method in D1_CONFIG
        },
        "traffic_protocol": (
            traffic_config["protocol"] if traffic_config is not None
            else "same_complete_effective_pool_for_train_and_eval; no holdout"
        ),
        "traffic_config": traffic_config,
        "training_traffic_split": "train" if random_profile else None,
        "traffic_seed_domains": (
            traffic_config["traffic_seed_domains"]
            if traffic_config is not None else None
        ),
        "traffic_files": [str(Path(path).resolve()) for path in effective_traffic],
        "methods": {},
        "started_at": time.time(),
    }

    def persist_launcher_state():
        with state_lock:
            base._write_json_atomic(root / "launcher_status.json", launcher_state)

    def update_method_state(name: str, **fields):
        with state_lock:
            entry = launcher_state["methods"].setdefault(name, {})
            entry.update(fields)
            base._write_json_atomic(root / "launcher_status.json", launcher_state)

    persist_launcher_state()

    def run_one(name: str) -> tuple[str, int]:
        cmd = [
            sys.executable, script, "train-method",
            "--method", name,
            "--scenario", NEW_SCENARIO,
            "--depart-scale", str(depart_scale),
            "--eval-traffic-split", EVAL_TRAFFIC_SPLIT,
            "--max-steps", str(max_steps),
            "--run-root", str(root.resolve()),
            "--checkpoint-frequency", str(base.CHECKPOINT_FREQUENCY),
        ]
        if behavior_diagnostics:
            cmd.append("--behavior-diagnostics")
        if smoke:
            cmd.append("--smoke")
            cmd.extend(["--smoke-eval-episodes", str(SMOKE_EVAL_EPISODES)])
        # 每个方法独立 train log（含 depart 后缀），两 worker 并行不互相覆盖。
        log = log_dir / f"{name}__train.log"
        update_method_state(
            name,
            status="queued",
            command=cmd,
            log=str(log),
        )
        try:
            with log.open("w", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                update_method_state(
                    name,
                    status="running",
                    pid=process.pid,
                    started_at=time.time(),
                )
                returncode = process.wait()
            update_method_state(
                name,
                status=("complete" if returncode == 0 else "failed"),
                exit_code=returncode,
                finished_at=time.time(),
            )
            return name, returncode
        except BaseException as exc:
            update_method_state(
                name,
                status="failed_to_launch",
                error=repr(exc),
                finished_at=time.time(),
            )
            return name, -1

    # 两 CUDA worker 共享本机 GPU 并行（不隔离 CUDA_VISIBLE_DEVICES）。
    with ThreadPoolExecutor(max_workers=TRAIN_WORKERS) as pool:
        futures = {pool.submit(run_one, name): name for name in methods}
        for future in as_completed(futures):
            name, code = future.result()
            codes[name] = code
            print(f"[launcher] {name} exit {code}", flush=True)

    failed = {n: c for n, c in codes.items() if c != 0}
    launcher_state["status"] = "failed" if failed else "complete"
    launcher_state["exit_codes"] = codes
    launcher_state["finished_at"] = time.time()
    persist_launcher_state()
    base._write_json_atomic(
        root / "experiment_manifest.json",
        {
            "source_script": script,
            "run_root": str(root.resolve()),
            "methods": methods,
            "scenario": NEW_SCENARIO,
            "depart_scale": depart_scale,
            "eval_traffic_split": EVAL_TRAFFIC_SPLIT,
            "raw_step_budget": 300 if smoke else max_steps,
            "smoke": smoke,
            "training_seed": base.SEED,
            "checkpoint_frequency_raw_steps": 100 if smoke else base.CHECKPOINT_FREQUENCY,
            "eval_episodes": SMOKE_EVAL_EPISODES if smoke else base.EVAL_EPISODES_TOTAL,
            "behavior_diagnostics": behavior_diagnostics,
            "reference_run_root": (
                str(Path(reference_run_root).resolve())
                if reference_run_root is not None else None
            ),
            "method_configs": {
                method: D1_CONFIG[method]
                for method in methods if method in D1_CONFIG
            },
            "traffic_protocol": (
                traffic_config["protocol"] if traffic_config is not None
                else "same_complete_effective_pool_for_train_and_eval; no holdout"
            ),
            "traffic_config": traffic_config,
            "training_traffic_split": "train" if random_profile else None,
            "traffic_seed_domains": (
                traffic_config["traffic_seed_domains"]
                if traffic_config is not None else None
            ),
            "traffic_files": [str(Path(path).resolve()) for path in effective_traffic],
            "exit_codes": codes,
            "failed_methods": failed,
        },
    )
    if failed:
        raise RuntimeError(
            f"training worker(s) failed: {failed}; inspect "
            f"{root}/*{NEW_SCENARIO}_depart*__train.log"
        )
    if behavior_diagnostics:
        write_behavior_comparison(root, methods)
    if any(method.startswith("sac_mlp_d1_st_rt_topo") for method in methods):
        from paired_stage_comparison import compare_stages

        def result_path(root_path: Path, method: str) -> Path:
            return (
                Path(root_path)
                / f"{method}__{NEW_SCENARIO}_depart{_depart_token(depart_scale)}"
                / "evaluation_results.json"
            )

        if "sac_mlp_d1_st_rt_topo" in methods and reference_run_root is not None:
            compare_stages(
                result_path(reference_run_root, "sac_mlp_d1_st_rt"),
                result_path(root, "sac_mlp_d1_st_rt_topo"),
                root / "comparison_strt_to_topo.json",
                label="ST-RT -> Topo",
                expected_episodes=SMOKE_EVAL_EPISODES if smoke else EVAL_EPISODES_TOTAL,
            )
        if {
            "sac_mlp_d1_st_rt_topo",
            "sac_mlp_d1_st_rt_topo_3slot",
        }.issubset(methods):
            compare_stages(
                result_path(root, "sac_mlp_d1_st_rt_topo"),
                result_path(root, "sac_mlp_d1_st_rt_topo_3slot"),
                root / "comparison_topo_to_3slot.json",
                label="Topo -> Topo-3Slot",
                expected_episodes=SMOKE_EVAL_EPISODES if smoke else EVAL_EPISODES_TOTAL,
            )
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("train-method", "eval-worker"),
        default=None,
        help="留空表示 launcher（两 CUDA worker 并行训练默认方法）",
    )
    parser.add_argument("--smoke", action="store_true", help="缩小规模快速验证全链路")
    parser.add_argument(
        "--smoke-eval-episodes", type=int, default=DEFAULT_SMOKE_EVAL_EPISODES,
        help="仅smoke模式的eval回合数（默认8）",
    )
    parser.add_argument("--method", choices=ALL_METHODS)
    parser.add_argument(
        "--scenario",
        choices=SUPPORTED_SCENARIOS,
        default="intersection_sorted",
        help="Scenario/profile; existing sorted scene remains the default",
    )
    parser.add_argument(
        "--methods",
        help="逗号分隔的方法清单；launcher 模式训练这些方法（默认 DEFAULT_METHODS）",
    )
    parser.add_argument(
        "--depart-scale",
        type=float,
        default=None,
        help="Sorted explicit-vehicle scale (default 4.0); randomized flow profiles require 1.0",
    )
    parser.add_argument(
        "--eval-traffic-split",
        choices=("validation", "test"),
        default="validation",
        help="Evaluation RNG domain for randomized profiles (test is held out)",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_STEPS,
                        help="d1 家族训练 raw 步数（默认 50k 满预算）")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--run-root",
        type=Path,
        help="独立结果根目录；用于从新目录运行，避免覆盖历史结果",
    )
    parser.add_argument(
        "--reference-run-root",
        type=Path,
        help="已完成对照结果根目录，用于无环境复跑的paired stage comparison",
    )
    parser.add_argument(
        "--behavior-diagnostics",
        action="store_true",
        help="记录训练优化轨迹和最终评估行为诊断",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=None,
        help="以原始仿真步为单位的检查点间隔",
    )
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--start-episode", type=int)
    parser.add_argument("--end-episode", type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--make-scenario-only",
        action="store_true",
        help="确保所选 sorted/random 场景资产存在后退出，不启动训练",
    )
    args = parser.parse_args(argv)

    random_profile = is_random_intersection_scenario(args.scenario)
    depart_scale = (
        float(args.depart_scale)
        if args.depart_scale is not None
        else (1.0 if random_profile else DEFAULT_DEPART_SCALE)
    )
    if not random_profile and args.eval_traffic_split != "validation":
        parser.error("--eval-traffic-split test applies only to random traffic profiles")

    global MAX_STEPS, SMOKE_EVAL_EPISODES
    MAX_STEPS = args.max_steps
    if args.smoke_eval_episodes <= 0:
        parser.error("--smoke-eval-episodes must be positive")
    SMOKE_EVAL_EPISODES = args.smoke_eval_episodes
    if args.checkpoint_frequency is not None:
        if args.checkpoint_frequency <= 0:
            parser.error("--checkpoint-frequency must be positive")
        base.CHECKPOINT_FREQUENCY = args.checkpoint_frequency
    if args.run_root is not None:
        base.RESULT_ROOT = args.run_root.expanduser().resolve()
        base.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        _apply_patch(depart_scale, args.scenario, args.eval_traffic_split)
    except ValueError as exc:
        parser.error(str(exc))

    if args.make_scenario_only:
        if random_profile:
            ensure_random_intersection_assets(NEW_SCENARIO)
        else:
            ensure_sorted_scenario()
        print(f"scenario ready: {NEW_SOURCE_TRAFFIC.parent}")
        return 0

    if not random_profile:
        ensure_sorted_scenario()

    if args.command == "train-method":
        if args.method is None:
            parser.error("train-method requires --method")
        train_and_eval(
            args.method,
            smoke=args.smoke,
            behavior_diagnostics=args.behavior_diagnostics,
        )
        return 0

    if args.command == "eval-worker":
        if (
            args.method is None
            or args.worker_id is None
            or args.start_episode is None
            or args.end_episode is None
            or args.model_path is None
            or args.output_dir is None
        ):
            parser.error(
                "eval-worker requires --method --worker-id --start-episode "
                "--end-episode --model-path --output-dir"
            )
        family, mod, m = DISPATCH[args.method]
        if family == "d1":
            parser.error("d1 家族走进程内评估，不经 eval-worker 子进程")
        args.method = m  # 映射回模块内部方法名
        args.behavior_diagnostics = args.behavior_diagnostics
        mod.run_eval_worker(args)
        return 0

    if args.eval_only:
        if args.method is None or args.model_path is None or args.output_dir is None:
            parser.error("--eval-only requires --method, --model-path, --output-dir")
        run_dir = Path(args.output_dir)
        base._guard_random_evaluation_output(
            run_dir, scenario=NEW_SCENARIO, traffic_split=EVAL_TRAFFIC_SPLIT
        )
        result = evaluate(
            args.method,
            run_dir,
            Path(args.model_path),
            smoke=args.smoke,
            behavior_diagnostics=args.behavior_diagnostics,
        )
        family, mod, method = DISPATCH[args.method]
        if family != "d1":
            mod.plot_training_curves(method, run_dir)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0

    if args.method is not None:
        # 便捷入口：``--method X`` 直接训练单个方法（无需 train-method 子命令）。
        train_and_eval(
            args.method,
            smoke=args.smoke,
            behavior_diagnostics=args.behavior_diagnostics,
        )
        return 0

    methods = None
    if args.methods:
        methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    run_full(
        smoke=args.smoke,
        methods=methods or DEFAULT_METHODS,
        depart_scale=depart_scale,
        max_steps=MAX_STEPS,
        behavior_diagnostics=args.behavior_diagnostics,
        reference_run_root=args.reference_run_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
