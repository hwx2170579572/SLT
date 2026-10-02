"""obs 修复版环境子类（改法 3）：neighbor 从纯欧氏距离改为「冲突相关性」选车。

背景（intersection 让行根因的 obs 盲区，见
``analysis_hold35k_mst_low_success.md`` 与记忆 ``intersection-rl-cant-yield-rootcause``）：
父类 ``SumoSceneEnv._nearest_actor_keys`` 对 ``neighbor_radius`` 内的车按**纯欧氏距离**
排序取 top-``neighbors``。在 ego 南进口左转时，top-5 槽位会被「离 ego 近但已通过路口、
正在远离的邻道出口车（约 9.6m）」占满，而真正「迎面接近冲突点的对向车（约 75m）」
因欧氏距离大被挤出 top-5，导致 MST/SLT 的 neighbor 张量里没有真正的冲突车。

改法 3 把排序键从「欧氏距离」换成「冲突相关性」——用每辆邻居车的**接近时间 TTC**
(time-to-closest-approach) 衡量它与 ego 多快会空间上逼近：

- 正在接近（相对距离缩小，closing_speed > 阈值）的车按 TTC 升序优先；
- 远离（closing_speed <= 阈值）的车兜底按欧氏距离排后。

位置用 ``_state[:2]``（SUMO east/north 中心坐标），速度用 TraCI 的 ``getSpeed`` +
``getAngle`` 读真实笛卡尔速度（而非 smarts 契约下被旋转 -90° 的速度分量），两者
坐标系一致。不依赖路口/路线等 intersection 特有信息，因此天然可迁移到其它场景
（cross/carla/cross_left/merge）。

MRO 说明：``YieldConflictIndependentV2EnvV4V1`` = ``_ConflictAwareNeighborMixin`` +
``YieldObsIndependentV2EnvV4V1``（改法 1+2 + 改法 3）。两个 mixin 都未定义
``__init__``，继承链由 ``_IndependentV2FiveBySixNamespaceMixin.__init__`` 收口；
``_nearest_actor_keys`` 与 ``_smarts_waypoint_polylines`` 各自 override、互不冲突。
"""

from __future__ import annotations

import math

import numpy as np

from yield_obs_env import YieldObsIndependentV2EnvV4V1


class _ConflictAwareNeighborMixin:
    """改法 3：neighbor 按「接近时间 TTC」的冲突相关性选车（而非纯欧氏距离）。

    TTC 依赖真实笛卡尔速度。注意：``_state`` 返回的 ``state[3:5]`` 在 smarts 契约下
    是 ``(speed*cos(heading), speed*sin(heading))``（heading 为 north-zero），即真实
    速度整体旋转 -90° 的非笛卡尔分量（父类 sumo_env.py 注释明确 "not a Cartesian
    direction vector"），与 ``state[:2]`` 的 SUMO (east, north) 位置差坐标系不一致，
    直接点积会把迎面接近的车误判成「远离」。故本 mixin 用 TraCI 的 ``getSpeed`` +
    ``getAngle`` 直接读 SUMO 真实笛卡尔速度 (east, north)，与契约无关。
    """

    # closing_speed（两车相对距离缩小速率）超过该阈值（m/s）视为「正在接近」。
    CLOSING_SPEED_EPS = 0.5

    def _cartesian_velocity(self, actor_key):
        """读 SUMO 真实笛卡尔速度 (east, north)，与观测契约无关；失败返回 None。"""
        domain, actor_id = actor_key.split(":", 1)
        try:
            if domain == "vehicle":
                speed = float(self._connection.vehicle.getSpeed(actor_id))
                angle = math.radians(float(self._connection.vehicle.getAngle(actor_id)))
            else:
                return None
        except Exception:
            return None
        # SUMO angle：0=北(north)、顺时针为正 → 笛卡尔 (east, north) 分量。
        return np.asarray(
            [speed * math.sin(angle), speed * math.cos(angle)], dtype=np.float64
        )

    def _nearest_actor_keys(self, ego_state):
        ego_key = f"vehicle:{self.specification.ego_id}"
        ego_vel = self._cartesian_velocity(ego_key)
        if ego_vel is None:
            ego_vel = np.zeros(2, dtype=np.float64)
        candidates: list[tuple[int, float, float, str]] = []
        for actor_key in self._active_actor_keys():
            state = self._state(actor_key)
            if state is None:
                continue
            rel_pos = state[:2] - ego_state[:2]
            distance = float(np.linalg.norm(rel_pos))
            if distance <= 1e-6:
                distance = 1e-6
            if distance > self.neighbor_radius:
                continue
            vel = self._cartesian_velocity(actor_key)
            if vel is None:
                # 读不到速度（如行人），兜底按欧氏距离排在远离组。
                candidates.append((1, float("inf"), distance, actor_key))
                continue
            rel_vel = vel - ego_vel
            # closing_speed > 0 表示两车相对距离在缩小（接近）。
            closing_speed = float(-np.dot(rel_vel, rel_pos) / distance)
            if closing_speed > self.CLOSING_SPEED_EPS:
                ttc = distance / closing_speed
                # 0 = 接近组优先，按 TTC 升序；distance 作为同 TTC 的次级键。
                candidates.append((0, ttc, distance, actor_key))
            else:
                # 1 = 远离组兜底，按欧氏距离升序。
                candidates.append((1, float("inf"), distance, actor_key))
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return [actor_key for _, _, _, actor_key in candidates[: self.neighbors]]


class YieldConflictIndependentV2EnvV4V1(
    _ConflictAwareNeighborMixin, YieldObsIndependentV2EnvV4V1
):
    """hold35k 用：改法 1+2（ego map）+ 改法 3（neighbor 冲突相关性选车）。"""


__all__ = ["_ConflictAwareNeighborMixin", "YieldConflictIndependentV2EnvV4V1"]
