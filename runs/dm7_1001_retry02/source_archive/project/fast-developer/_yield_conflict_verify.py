"""验证改法 3（neighbor 冲突相关性选车）与改法 1+2 在 v4 环境上的接线。

两层验证：

1. **单元验证（假对象）**：直接喂三辆预设邻居车，断言 ``_ConflictAwareNeighborMixin``
   的 TTC 排序把「迎面接近的对向冲突车」提到首位，而父类纯欧氏距离把它排到末尾——
   证明改法 3 确实改了「选哪几辆车」。
2. **真实环境 smoke**：实例化 ``YieldConflictIndependentV2EnvV4V1``（hold35k 用，
   改法 1+2+3）并 reset + 跑一步，断言 (a) MRO 正确（``_nearest_actor_keys`` 来自改法 3
   mixin、``_smarts_waypoint_polylines`` 来自改法 1+2 mixin）、(b) 环境不崩。

用法::

    python fast-developer/_yield_conflict_verify.py
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

from yield_conflict_env import _ConflictAwareNeighborMixin
from yield_obs_env import _YieldObsMixin


# --------------------------------------------------------------------------- #
# 1. 单元验证（假对象）
# --------------------------------------------------------------------------- #
class _FakeConnection:
    """假 TraCI connection.vehicle：按 actor_id 返回预设 speed(m/s)/angle(deg)。

    SUMO angle 约定：0=北(north)、顺时针为正；笛卡尔 (east, north) =
    (speed*sin(angle), speed*cos(angle))，与 yield_conflict_env._cartesian_velocity 一致。
    """

    def __init__(self, speeds, angles):
        self._speeds = speeds
        self._angles = angles
        self.vehicle = self

    def getSpeed(self, actor_id):
        return self._speeds[actor_id]

    def getAngle(self, actor_id):
        return self._angles[actor_id]


class _FakeSpec:
    ego_id = "ego"


class _FakeNeighborEnv(_ConflictAwareNeighborMixin):
    def __init__(self, states, speeds, angles, neighbor_radius, neighbors):
        self._states = states
        self.specification = _FakeSpec()
        self._connection = _FakeConnection(speeds, angles)
        self.neighbor_radius = neighbor_radius
        self.neighbors = neighbors

    def _active_actor_keys(self):
        return list(self._states.keys())

    def _state(self, key):
        return self._states[key]


def test_unit() -> None:
    print("== 单元验证（TTC 冲突相关性排序）==")
    # ego：位置 (0,0)、速度 (5,0)（沿 +east 前进）。
    ego_state = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
    # 位置用 _state[:2]，速度用 TraCI getSpeed/getAngle（SUMO angle：0=北、顺时针正）。
    states = {
        # A：迎面接近的对向冲突车（+east 远端、沿 -east 高速接近，angle=270°=西）。
        "vehicle:A": np.array([75.0, 0.0, 0.0, -10.0, 0.0]),
        # B：已通过路口的邻道出口车（很近、但横向且远离，angle=90°=东）。
        "vehicle:B": np.array([0.0, 9.6, 0.0, 10.0, 0.0]),
        # C：同向前方静止车（中距、speed=0，ego 正在接近）。
        "vehicle:C": np.array([30.0, 0.0, 0.0, 0.0, 0.0]),
    }
    speeds = {"ego": 5.0, "A": 10.0, "B": 10.0, "C": 0.0}
    angles = {"ego": 90.0, "A": 270.0, "B": 90.0, "C": 0.0}
    env = _FakeNeighborEnv(
        states, speeds, angles, neighbor_radius=80.0, neighbors=5
    )

    conflict_order = env._nearest_actor_keys(ego_state)

    # 父类纯欧氏距离排序（用于对照）。
    by_distance = sorted(
        states.items(), key=lambda kv: float(np.linalg.norm(kv[1][:2] - ego_state[:2]))
    )
    euclid_order = [k for k, _ in by_distance]

    print(f"  纯欧氏距离 top-5 : {euclid_order}")
    print(f"  冲突相关性 top-5 : {conflict_order}")
    # 改法 3 生效：冲突车 A 从欧氏排序的末尾提升到首位。
    assert euclid_order == ["vehicle:B", "vehicle:C", "vehicle:A"], euclid_order
    assert conflict_order == ["vehicle:A", "vehicle:C", "vehicle:B"], conflict_order
    print("  断言通过：对向冲突车 A 从『欧氏第 3』提升到『冲突第 1』；无关出口车 B 被降到最后。")
    print("  单元验证通过。\n")


# --------------------------------------------------------------------------- #
# 2. 真实环境 smoke（MRO 接线 + 不崩）
# --------------------------------------------------------------------------- #
def test_real_env_smoke() -> None:
    print("== 真实环境 smoke（v4 冲突版环境 MRO + reset）==")
    from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory
    from train_intersection_yield_v2 import _environment_namespace, DENSITY
    from yield_conflict_env import YieldConflictIndependentV2EnvV4V1

    # MRO 静态断言：两个 override 分别来自对应 mixin。
    assert (
        YieldConflictIndependentV2EnvV4V1._nearest_actor_keys
        is _ConflictAwareNeighborMixin._nearest_actor_keys
    ), "改法 3 的 _nearest_actor_keys 未从 mixin 接管"
    assert (
        YieldConflictIndependentV2EnvV4V1._smarts_waypoint_polylines
        is _YieldObsMixin._smarts_waypoint_polylines
    ), "改法 1+2 的 _smarts_waypoint_polylines 未从 mixin 接管"
    print("  MRO 静态断言通过：改法 3（neighbor）+ 改法 1+2（map）均在 v4 类上生效。")

    overlay = _THIS_DIR / "_conflict_verify_overlay_v4"
    shutil.rmtree(overlay, ignore_errors=True)
    factory = _make_environment_factory(
        adapter="v4_8",
        density=DENSITY,
        overlay_root=overlay,
        v4_environment_class=YieldConflictIndependentV2EnvV4V1,
    )
    env = factory(_environment_namespace(), evaluation=False)
    try:
        obs, _ = env.reset(seed=420_000)
        # 跑一步验证 step 不崩、observation 形状正常。
        action = np.array([0.0, 0.0])
        _, r, terminated, truncated, info = env.step(action)
        print(f"  reset + 1 step 成功；reward={float(r):+.3f}")
        print(f"  obs keys: {sorted(obs.keys())}")
        keys = env._nearest_actor_keys(np.zeros(5, dtype=np.float32))
        print(f"  _nearest_actor_keys 返回 {len(keys)} 个邻居（改法 3 接管）")
        print("  真实环境 smoke 通过。\n")
    finally:
        env.close()


def main() -> int:
    test_unit()
    test_real_env_smoke()
    print("结论：改法 3（neighbor 冲突相关性选车）生效，且与改法 1+2 正确叠加在 v4 类上。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
