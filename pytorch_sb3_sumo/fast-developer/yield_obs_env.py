"""obs 修复版环境子类（改法 1+2）：ego 的 map 沿真实车道图穿过 internal 弧线、从 ego 所在车道出发。

背景：论文 Scene-Rep 的 ``map`` 张量语义是「每个 actor 的候选未来路线」，应完整穿过路口。
但论文源码的固定 route 过滤会拒绝 internal edge（源码旁留 ``What about internal lanes?``
TODO，见 experiments/sb3_sumo_paper/SOURCE_PAPER_DIFFERENCES.md 第 9 条），且
``waypoint_paths[:2]`` 按车道索引升序取前 2 条，导致 ego 在 3 车道左转道（departLane=2）
时：左转 internal 弧线在路口入口被截断、左转道不进 map。

本子类只 override ego 的候选路线生成，做「合理复现」（忠于论文原理而非源码 bug），
模型结构与输入 shape 均不变：

- 改法 1：ego 不再走 ``_route_polylines``（allowed_edges=route 过滤 internal 边），
  改走无约束车道图（allowed_edges=None），左转 internal 弧线不再被截断。
- 改法 2：候选路线从 ego 当前车道出发（而非车道索引升序取前 2 条），
  保证 ego 的左转道进 map。

邻居（is_ego=False）完全走父类原逻辑，语义不变。仅用于 intersection 的让行诊断，
作为 ``_make_environment_factory`` 的 ``baseline_environment_class`` /
``v4_environment_class`` 传入。

逻辑抽成 ``_YieldObsMixin``，供 base（连续动作头，mst_slt）与 v4（混合动作头，
hold35k）两类复用；``YieldObsIndependentV2EnvV1`` 为 mst_slt 用（改法 1+2），
``YieldObsIndependentV2EnvV4V1`` 为 hold35k 的「改法 1+2 only」对照。
"""

from __future__ import annotations

from envs.sumo.independent_v2_five_methods_six_scenarios_100ep_v1 import (
    IndependentV2FiveBySixEnvV1,
    IndependentV2FiveBySixEnvV4V1,
)


class _YieldObsMixin:
    """改法 1+2 的 mixin：ego 候选路线穿过 internal 弧线、从 ego 车道出发。"""

    def _smarts_waypoint_polylines(self, *, route, road_id, position, is_ego, limit):
        if not is_ego:
            return super()._smarts_waypoint_polylines(
                route=route, road_id=road_id, position=position,
                is_ego=False, limit=limit,
            )
        if road_id and not road_id.startswith(":"):
            # 改法 1+2：普通 edge 上，ego 走无约束车道图且 ego 车道优先。
            return self._yield_ego_polylines(road_id, position, limit=limit)
        # ego 在路口 internal edge 内：复用父类「非 ego」的路口内分支
        # （current internal edge + 其 outgoing edge），让 ego 路径从 internal
        # 边继续穿到出口，而不是回跳入口。
        return super()._smarts_waypoint_polylines(
            route=route, road_id=road_id, position=position,
            is_ego=False, limit=limit,
        )

    def _yield_ego_polylines(self, start_edge, position, *, limit):
        """无约束车道图前视，且 ego 当前车道排最前。

        复刻父类 ``_lane_graph_polylines`` 的完整逻辑，仅改两点：
        (1) ``allowed_edges=None``（不过滤 internal 边，改法 1）；
        (2) 遍历车道时把 ego 当前车道排最前（改法 2）。
        做法是临时替换 ``self._edge_lane_ids`` 让最外层车道遍历顺序「ego 优先」，
        再调用父类 ``_lane_graph_polylines``，finally 恢复。该重排只影响 ego 的
        这一次 map 生成，不影响任何其它调用方。
        """
        original_edge_lane_ids = self._edge_lane_ids
        try:
            ego_lane_index = int(
                self._connection.vehicle.getLaneIndex(self.specification.ego_id)
            )
        except self._traci.TraCIException:
            ego_lane_index = None
        ego_lane = (
            f"{start_edge}_{ego_lane_index}" if ego_lane_index is not None else None
        )

        def _ego_first_lane_ids(edge_id: str):
            lane_ids = original_edge_lane_ids(edge_id)
            if edge_id == start_edge and ego_lane in lane_ids:
                return [ego_lane, *[lane for lane in lane_ids if lane != ego_lane]]
            return lane_ids

        self._edge_lane_ids = _ego_first_lane_ids
        try:
            return self._lane_graph_polylines(
                start_edge, position, allowed_edges=None, limit=limit
            )
        finally:
            self._edge_lane_ids = original_edge_lane_ids


class YieldObsIndependentV2EnvV1(_YieldObsMixin, IndependentV2FiveBySixEnvV1):
    """base 连续动作头（mst_slt 用）：改法 1+2，邻居语义不变。"""


class YieldObsIndependentV2EnvV4V1(_YieldObsMixin, IndependentV2FiveBySixEnvV4V1):
    """v4 混合动作头（hold35k 用）：改法 1+2 only，邻居语义不变（不加改法 3 的对照）。"""


__all__ = [
    "_YieldObsMixin",
    "YieldObsIndependentV2EnvV1",
    "YieldObsIndependentV2EnvV4V1",
]
