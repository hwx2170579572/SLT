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

# 新副本场景（depart 排序版 intersection）与 traffic 源。
NEW_SCENARIO = "intersection_sorted"
NEW_SOURCE_TRAFFIC = (
    PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1"
    / "intersection_sorted" / "traffic"
)

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
    "sac_mlp_d1_full": dict(use_route=True, use_topology=True, use_slots=True),
    "hsac_mlp_base_d1_st": dict(use_route=False, use_topology=False, use_slots=False),
    "hsac_mlp_base_d1_st_rt": dict(use_route=True, use_topology=False, use_slots=False),
    "hsac_mlp_base_d1_st_rt_topo": dict(use_route=True, use_topology=True, use_slots=False),
    "hsac_mlp_base_d1_full": dict(use_route=True, use_topology=True, use_slots=True),
}

DEFAULT_STEPS = base.RAW_TRAINING_STEPS  # 50_000 满预算
EVAL_EPISODES_TOTAL = base.EVAL_EPISODES_TOTAL


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
    开关。D1-4（use_slots=True）额外开 SLT + SBS（算法类换成 SceneRepresentationSACV2）。
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

    # 算法类：D1-4（三槽 + SLT/SBS）换 SceneRepresentationSACV2；其余同父基线。
    use_slots = bool(cfg["use_slots"])
    algorithm_class = SceneRepresentationSACV2 if use_slots else SceneRepresentationSAC

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

    if use_slots:
        # D1-4：三槽输出 + Graph-SLT（M7）+ SoftBalancedSlots（M8）。
        common_kwargs.update(
            structured_representation=True,
            representation_coef=1.0,
            representation_learning_rate=learning_rate,
            representation_heads=2,
            representation_separate_target_projector=False,
            representation_online_target_encoder=(SCENARIO != "cross"),
            slot_balance_coef=0.01,
        )
    else:
        # D1-1/2/3：关 Scene-Rep 表示目标（与父基线 representation_coef=0.0 一致）。
        common_kwargs.update(representation_coef=0.0)

    return algorithm_class(policy_class, env, **common_kwargs)


def run_training_d1(method: str, run_dir: Path, smoke: bool, max_steps: int) -> Path:
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
    env_adapter = _env_adapter_d1(method)

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
        model = _build_model_d1(method, env, learning_starts=learning_starts)

        cfg = D1_CONFIG[method]
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
                ablation="d1_representation_incremental",
                use_route=cfg["use_route"],
                use_topology=cfg["use_topology"],
                use_slots=cfg["use_slots"],
                structured_representation=bool(cfg["use_slots"]),
                representation_coef=(1.0 if cfg["use_slots"] else 0.0),
                slot_balance_coef=(0.01 if cfg["use_slots"] else 0.0),
                note=(
                    f"与父基线 {PARENT[method]} 唯一差异：features_extractor 换 "
                    f"IncrementalTopoEncoder（use_route={cfg['use_route']}, "
                    f"use_topology={cfg['use_topology']}, use_slots={cfg['use_slots']}）。"
                    "动作头 / 熵 / 调参 / n_step / 环境契约完全一致。"
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
    """进程内评估（对齐父基线部署语义：连续头直接 load，混合头 + use_actor_only）。"""
    import torch

    from algos.sb3_torch.evaluation import evaluate_model_detailed
    from algos.sb3_torch.sac import SceneRepresentationSAC
    from algos.sb3_torch.sac_v2 import SceneRepresentationSACV2

    torch.set_num_threads(1)
    parent = PARENT[method]
    use_slots = bool(D1_CONFIG[method]["use_slots"])
    env_adapter = _env_adapter_d1(method)
    # 评估 overlay 目录保持极短（"oe"）：续训时 run_dir = <run>/c100000 多一层，
    # 长方法名（如 sac_mlp_d1_st_rt_late）会把 manifest 路径顶到 264 > MAX_PATH(260)，
    # os.replace 报 WinError 3。原 "overlays/eval"(12 字符) 减到 "oe"(2 字符) 后降到 254。
    overlay_root = run_dir / "oe"
    env = base.make_env_factory(env_adapter, overlay_root)(
        base._environment_namespace(), evaluation=True
    )
    load_cls = SceneRepresentationSACV2 if use_slots else SceneRepresentationSAC
    loaded = load_cls.load(str(model_path), env=env, device="cpu", buffer_size=32)
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


def _apply_patch(depart_scale: float) -> None:
    """把各脚本的 SCENARIO（及 yield_v2 的 traffic 源 / depart 缩放）指向本实验。

    只在当前进程内覆盖模块常量：所有复用函数运行时经模块全局查找 SCENARIO /
    SOURCE_TRAFFIC / DEPART_SCALE，因此覆盖即刻生效。legacy 家族经 registry 按
    SCENARIO 名加载 traffic（原始密度），无 SOURCE_TRAFFIC / DEPART_SCALE 常量，
    故只 patch 其 SCENARIO。d1 家族的 SCENARIO 是 import 时拷贝的模块级副本，须单独
    同步（_build_model_d1 用它判断 random_augmentation / carla_contract）。
    """
    global DEPART_SCALE, SCENARIO
    DEPART_SCALE = depart_scale
    SCENARIO = NEW_SCENARIO
    for mod in (base, legacy):
        mod.SCENARIO = NEW_SCENARIO
    base.SOURCE_TRAFFIC = NEW_SOURCE_TRAFFIC
    base.DEPART_SCALE = depart_scale


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


def _train(global_name: str, run_dir: Path, smoke: bool) -> Path:
    """复用底层训练函数（当前进程内训练，返回 final_model 路径）。"""
    family, mod, method = DISPATCH[global_name]
    if family == "d1":
        return run_training_d1(method, run_dir, smoke=smoke, max_steps=MAX_STEPS)
    if method == "hold35k":
        return mod.run_training_hold35k(run_dir, smoke=smoke)
    if method == "mst_slt":
        return mod.run_training_mst_slt(run_dir, smoke=smoke)
    # 其余为 yield_v2 的 MLP 方法（hsac_mlp / sac_mlp / sac_mlp_v48 / hsac_mlp_base）
    return mod.run_training_mlp(method, run_dir, smoke=smoke)


def _evaluate_d1(global_name: str, run_dir: Path, final_model: Path, smoke: bool) -> dict:
    """d1 家族进程内评估（无 run_eval_worker，用 _evaluate_saved）。"""
    family, mod, method = DISPATCH[global_name]
    n_ep = 8 if smoke else EVAL_EPISODES_TOTAL
    summary, records = _evaluate_saved(method, final_model, run_dir, n_ep)
    result = dict(
        identity=dict(
            method=global_name,
            source_script="train_intersection_yield_v2_d1",
            source_method=method,
            family=family,
            scenario=NEW_SCENARIO,
            depart_scale=DEPART_SCALE,
            checkpoint=str(final_model),
            checkpoint_sha256=base._sha256(final_model),
            episodes=n_ep,
            workers=1,
            smoke=smoke,
        ),
        summary=summary,
        episode_records=records,
    )
    base._write_json_atomic(run_dir / "evaluation_results.json", result)
    return result


def evaluate(
    global_name: str, run_dir: Path, final_model: Path, smoke: bool
) -> dict:
    """d1 家族进程内评估；yv2 / legacy 家族 subprocess 并行评估。

    subprocess 隔离让每个 worker 退出时自动回收 SUMO 子进程，避免进程内线程并行时
    SUMO 进程残留锁住 traffic 文件。eval-worker 子进程带 ``--depart-scale``，由其
    main 分支 ``_apply_patch`` 恢复场景与发车间隔后调用底层 ``run_eval_worker``。
    """
    family, mod, method = DISPATCH[global_name]
    if family == "d1":
        return _evaluate_d1(global_name, run_dir, final_model, smoke)

    workers = 2 if smoke else mod.EVAL_WORKERS
    episodes = 8 if smoke else mod.EVAL_EPISODES_TOTAL

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
            "--depart-scale", str(DEPART_SCALE),
            "--worker-id", str(wid),
            "--start-episode", str(start),
            "--end-episode", str(end),
            "--model-path", str(final_model),
            "--output-dir", str(run_dir),
        ]
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
            checkpoint=str(final_model),
            checkpoint_sha256=mod._sha256(final_model),
            episodes=episodes,
            workers=workers,
            episodes_per_worker=[end - start for start, end in ranges],
            smoke=smoke,
        ),
        summary=summary,
        episode_records=records,
    )
    mod._write_json_atomic(run_dir / "evaluation_results.json", result)
    return result


def train_and_eval(global_name: str, smoke: bool) -> Path:
    family, mod, method = DISPATCH[global_name]
    run_dir = _run_dir(global_name, DEPART_SCALE)
    # 不预先 mkdir：mst_slt 经 train_sb3.main 要求 run_dir 尚不存在（exist_ok=False），
    # hold35k / mlp / d1 各自 mkdir(exist_ok=True)。
    final = _train(global_name, run_dir, smoke=smoke)
    result = evaluate(global_name, run_dir, final, smoke=smoke)
    curve = None
    if family != "d1":
        curve = mod.plot_training_curves(method, run_dir)

    manifest = dict(
        method=global_name,
        source_script=(
            "train_intersection_yield_v2_d1" if family == "d1" else mod.__name__
        ),
        source_method=method,
        family=family,
        scenario=NEW_SCENARIO,
        depart_scale=DEPART_SCALE,
        run_dir=str(run_dir),
        final_model=str(final),
        evaluation=result["summary"],
        training_curve=str(curve) if curve else None,
        smoke=smoke,
        note=(
            "depart 排序版 intersection 副本：三股车流（北→南 / 东→西 / 西→东）"
            f"都参与仿真 + 发车间隔 ×{DEPART_SCALE}（降密度）。"
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
    smoke: bool, methods: tuple[str, ...], depart_scale: float
) -> Path:
    if not smoke and not base._cuda_available():
        raise RuntimeError("CUDA unavailable; training has no CPU fallback")

    root = base.RESULT_ROOT  # fast-developer
    root.mkdir(parents=True, exist_ok=True)
    if not smoke:
        existing = [n for n in methods if _run_dir(n, depart_scale).exists()]
        if existing:
            raise RuntimeError(f"output dirs already exist (remove first): {existing}")

    script = str(Path(__file__).resolve())
    codes: dict[str, int] = {}

    def run_one(name: str) -> tuple[str, int]:
        cmd = [
            sys.executable, script, "train-method",
            "--method", name,
            "--depart-scale", str(depart_scale),
        ]
        if smoke:
            cmd.append("--smoke")
        # 每个方法独立 train log（含 depart 后缀），两 worker 并行不互相覆盖。
        log = root / f"{name}__{NEW_SCENARIO}_depart{_depart_token(depart_scale)}__train.log"
        with log.open("w", encoding="utf-8") as handle:
            returncode = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).returncode
        return name, returncode

    # 两 CUDA worker 共享本机 GPU 并行（不隔离 CUDA_VISIBLE_DEVICES）。
    with ThreadPoolExecutor(max_workers=TRAIN_WORKERS) as pool:
        futures = {pool.submit(run_one, name): name for name in methods}
        for future in as_completed(futures):
            name, code = future.result()
            codes[name] = code
            print(f"[launcher] {name} exit {code}", flush=True)

    failed = {n: c for n, c in codes.items() if c != 0}
    if failed:
        raise RuntimeError(
            f"training worker(s) failed: {failed}; inspect "
            f"{root}/*{NEW_SCENARIO}_depart*__train.log"
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
    parser.add_argument("--method", choices=ALL_METHODS)
    parser.add_argument(
        "--depart-scale",
        type=float,
        default=DEFAULT_DEPART_SCALE,
        help="发车间隔缩放（>1.0 降密度，默认 4.0）",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_STEPS,
                        help="d1 家族训练 raw 步数（默认 50k 满预算）")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--start-episode", type=int)
    parser.add_argument("--end-episode", type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--make-scenario-only",
        action="store_true",
        help="只生成 depart 排序版副本场景后退出",
    )
    args = parser.parse_args(argv)

    global MAX_STEPS
    MAX_STEPS = args.max_steps
    _apply_patch(args.depart_scale)

    if args.make_scenario_only:
        ensure_sorted_scenario()
        print(f"scenario ready: {NEW_SOURCE_TRAFFIC.parent}")
        return 0

    ensure_sorted_scenario()

    if args.command == "train-method":
        if args.method is None:
            parser.error("train-method requires --method")
        train_and_eval(args.method, smoke=args.smoke)
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
        mod.run_eval_worker(args)
        return 0

    if args.eval_only:
        if args.method is None or args.model_path is None or args.output_dir is None:
            parser.error("--eval-only requires --method, --model-path, --output-dir")
        run_dir = Path(args.output_dir)
        result = evaluate(args.method, run_dir, Path(args.model_path), smoke=args.smoke)
        family, mod, method = DISPATCH[args.method]
        if family != "d1":
            mod.plot_training_curves(method, run_dir)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0

    if args.method is not None:
        # 便捷入口：``--method X`` 直接训练单个方法（无需 train-method 子命令）。
        train_and_eval(args.method, smoke=args.smoke)
        return 0

    run_full(smoke=args.smoke, methods=DEFAULT_METHODS, depart_scale=args.depart_scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
