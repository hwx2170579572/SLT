"""reward shaping 包装器（P1 根因修复）：不改动 sumo_env.py，仅包裹环境改写 reward。

根因（见 analysis_mst_slt_fix_failure_rootcause_v2.md）：环境原始 reward 为稀疏三元
``raw_reward = float(success) - float(collision)``（中间步全 0、timeout=0），配合
``setSpeedMode(ego, 0)`` 关闭 SUMO 让行保护后，"停车等到超时"成为价值恒 0 的零风险
吸收策略，严格支配早期价值为负的抢行策略；策略滑入该盆地后 policy entropy→0、
SAC 自动降 alpha、回放全 timeout 无梯度，正反馈锁死。

本包装器在单环境层面改写 step() 返回的 reward，三处 shaping：

  (a) timeout -> 负值（``-timeout_penalty``），打破 ``timeout(0) > collision(-1)`` 的支配；
  (b) 稠密进度奖励：沿 route 的累计行驶距离增量 ``progress_scale * ΔgetDistance``，
      为"接近路口→观察间隙→择机穿行"的时序让行提供可回传梯度；
  (c) 小步长生活成本（``-step_cost``/决策步），抑制原地停留。

不改 observation / termination / truncated / info，故评估（读 info["is_success"]）
不受影响，训练与评估可共用同一包装器。
"""

from __future__ import annotations

import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        timeout_penalty: float = 1.0,
        step_cost: float = 0.005,
        progress_scale: float = 0.01,
    ):
        super().__init__(env)
        self.timeout_penalty = float(timeout_penalty)
        self.step_cost = float(step_cost)
        self.progress_scale = float(progress_scale)
        self._prev_distance: float | None = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_distance = self._ego_distance()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = float(reward)
        # (a) timeout -> 负值（与 collision 同罚量级），消除零风险吸收盆地。
        if bool(info.get("max_time", False)):
            shaped = -self.timeout_penalty
        # (c) 生活成本：抑制原地停留。
        shaped -= self.step_cost
        # (b) 进度奖励：沿 route 前进的累计行驶距离增量。
        cur = self._ego_distance()
        if self._prev_distance is not None and cur is not None:
            shaped += self.progress_scale * float(cur - self._prev_distance)
        if cur is not None:
            self._prev_distance = cur
        return obs, shaped, terminated, truncated, info

    def _ego_distance(self) -> float | None:
        """读 ego 自插入以来沿 route 的累计行驶距离（米）；不可读时返回 None。"""
        try:
            unwrapped = self.env.unwrapped
            conn = getattr(unwrapped, "_connection", None)
            ego_id = getattr(unwrapped.specification, "ego_id", None)
            if conn is None or ego_id is None:
                return None
            if ego_id not in conn.vehicle.getIDList():
                return None
            return float(conn.vehicle.getDistance(ego_id))
        except Exception:
            return None
