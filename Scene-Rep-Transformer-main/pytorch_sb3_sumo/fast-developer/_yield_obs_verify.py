"""验证改法 1+2：对比原始 vs 修复版环境，ego 在 -E1 接近路口时 map 是否穿过路口。

做法：两个环境各自 reset 后，把 ego 用 TraCI 直接搬到 -E1 左转道（车道 2）接近
路口的位置（65m/70m），再 dump ego 的 ``_actor_paths`` 前 2 条候选路线的坐标。
- 原始版（IndependentV2FiveBySixEnvV1）：ego map 走 ``_route_polylines``，internal 边
  被过滤、且只取车道 0/1，故坐标停在 -E1 直线上、x 始终 > 4，在路口入口截断。
- 修复版（YieldObsIndependentV2EnvV1）：ego map 走无约束车道图 + ego 车道优先，
  故坐标从 x=1.6 沿左转 internal 弧线穿过 x≈0（进入路口）。

用法::

    python fast-developer/_yield_obs_verify.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import train_intersection_hold35k_mst_fixed as base
from envs.sumo.independent_v2_five_methods_six_scenarios_100ep_v1 import (
    IndependentV2FiveBySixEnvV1,
)
from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory
from yield_obs_env import YieldObsIndependentV2EnvV1


def _build_env(env_class, overlay_root: Path):
    factory = _make_environment_factory(
        adapter="base",
        density=base.DENSITY,
        overlay_root=overlay_root,
        baseline_environment_class=env_class,
    )
    return factory(base._environment_namespace(), evaluation=False)


def _ego_map_coords(env) -> np.ndarray:
    ego_id = env.specification.ego_id
    # 把 ego 搬到 -E1 左转道（车道 2）接近路口处（65m/70m），让 10m 前视覆盖到路口。
    env._connection.vehicle.moveTo(ego_id, "-E1_2", 65.0)
    env._connection.vehicle.setSpeed(ego_id, 0.0)
    env._connection.simulationStep()
    road_id = str(env._connection.vehicle.getRoadID(ego_id))
    lane_index = int(env._connection.vehicle.getLaneIndex(ego_id))
    map_state = env._actor_paths(f"vehicle:{ego_id}", is_ego=True)  # (2, 10, 5)
    print(f"  ego road={road_id} lane={lane_index} map_shape={map_state.shape}")
    return map_state[:, :, :2]


def _describe(coords: np.ndarray) -> str:
    rows = []
    for path_index in range(coords.shape[0]):
        pts = coords[path_index]
        nonzero = pts[np.any(pts != 0, axis=1)]
        if not len(nonzero):
            rows.append(f"    path{path_index}: (empty)")
            continue
        xs, ys = nonzero[:, 0], nonzero[:, 1]
        pt_list = ", ".join(f"({x:.1f},{y:.1f})" for x, y in nonzero)
        rows.append(f"    path{path_index}: n={len(nonzero)}")
        rows.append(f"      pts: {pt_list}")
    return "\n".join(rows)


def main() -> int:
    overlay_orig = _THIS_DIR / "_yield_verify_overlay_orig"
    overlay_fixed = _THIS_DIR / "_yield_verify_overlay_fixed"
    shutil.rmtree(overlay_orig, ignore_errors=True)
    shutil.rmtree(overlay_fixed, ignore_errors=True)

    print("== 原始环境（IndependentV2FiveBySixEnvV1）==")
    env_orig = _build_env(IndependentV2FiveBySixEnvV1, overlay_orig)
    env_orig.reset(seed=base.SEED_START["mst_slt"])
    coords_orig = _ego_map_coords(env_orig)
    print(_describe(coords_orig))
    env_orig.close()

    print("== 修复版环境（YieldObsIndependentV2EnvV1）==")
    env_fixed = _build_env(YieldObsIndependentV2EnvV1, overlay_fixed)
    env_fixed.reset(seed=base.SEED_START["mst_slt"])
    coords_fixed = _ego_map_coords(env_fixed)
    print(_describe(coords_fixed))
    env_fixed.close()

    def _n_pts(coords: np.ndarray) -> int:
        return int(np.count_nonzero(np.any(coords != 0, axis=2)))

    def _path0_start_x(coords: np.ndarray) -> float:
        pts = coords[0][np.any(coords[0] != 0, axis=1)]
        return float(pts[0, 0]) if len(pts) else float("nan")

    # 改法 1：修复版 path 应穿过路口入口（非零点数显著多于原始版，原始版截断在入口）。
    # 改法 2：修复版 path0 应从 ego 车道（x≈1.6，车道 2）而非车道 0（x≈8.0）出发。
    n_orig, n_fixed = _n_pts(coords_orig), _n_pts(coords_fixed)
    x_orig, x_fixed = _path0_start_x(coords_orig), _path0_start_x(coords_fixed)
    print(f"\n原始版 path0 起点 x={x_orig:.1f}（车道0≈8.0）  修复版 path0 起点 x={x_fixed:.1f}（ego车道2≈1.6）")
    print(f"非零点数：原始版={n_orig}（截断）  修复版={n_fixed}（穿过路口入口）")
    method1_ok = n_fixed > n_orig
    method2_ok = abs(x_fixed - 1.6) < 1.0 and abs(x_orig - 8.0) < 1.0
    if method1_ok and method2_ok:
        print("结论：改法 1+2 生效 —— ego map 从「车道0/1、路口入口截断」变为「ego车道出发、穿过 internal 弧线」。")
        return 0
    print("结论：未观察到预期差异，需检查 moveTo 位置 / map 前视 / 车道归属。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
