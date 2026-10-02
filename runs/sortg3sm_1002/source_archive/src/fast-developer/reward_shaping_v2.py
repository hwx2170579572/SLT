"""可泛化的 reward shaping（v2）：打破 timeout 吸收盆地 + 提供前进梯度。

P1 版（``reward_shaping_wrapper.py``）的三处 shaping 有已知副作用（根因分析
``analysis_mst_slt_fix_failure_rootcause_v2.md``）：``progress_scale`` 取 0.01/米 且
``timeout=-1`` 与 ``collision=-1`` 同量级时，抢行的进度奖励累计会把「抢行」的累积
return 抬到「让行」之上，反鼓励让行。本版（v2）修正量级结构：

- **终局奖励压倒性**（±10 量级），决定 episode 相对序，是唯一的高回报信号；
- **过程项只提供梯度方向**（progress 0.02/米、step_cost 0.01/步，量级远小于终局），
  不改变「成功 > 超时 > 碰撞」的相对序。

数值设计（全部信号在所有场景通用，不依赖 intersection 特有信息，见 docstring 末尾）：

  success    = +10.0   成功到达终点（压倒性正信号，明确目标）。
  collision  = -10.0   碰撞（压倒性负信号，明确避碰）。
  off_route  = -10.0   偏离路线/被 teleport（等同碰撞的灾难）。
  timeout    = -5.0    超时（负，但比 collision 温和，避免「宁可撞也不超时」）。
  progress   = +0.02/米  沿 route 累计行驶距离增量（前进梯度）。
  step_cost  = -0.01/决策步  生活成本（抑制原地停留）。

量级分离的关键：一个成功 episode 约前进 ~200m（progress ≈ +4），终局 +10 仍是主导；
一个「停车等超时」episode 的 progress ≈ 0、timeout -5、生活成本 ≈ -2（600 raw 步 /
action_repeat=3 ≈ 200 决策步 × -0.01），净 return ≈ -7（明确为负，不再是旧版
time=0 的零风险吸收盆地）。

**可泛化性**：终局四信号（success/collision/off_route/timeout）与过程两信号
（progress/step_cost）都来自环境通用 info 与 ego 的 ``getDistance``，不含任何
「路口中心 / 让行 gap / 冲突车」等 intersection 特有量，可直接迁移到
cross / carla / cross_left / merge 等场景。

不改 observation / termination / truncated / info，评估（读 info["is_success"]）
不受影响；训练与评估可共用同一包装器。
"""

from __future__ import annotations

import gymnasium as gym

# reward shaping 的六个分支（episode 累计，写进 info 供 Monitor / callback 记录）。
REWARD_BRANCH_KEYS = (
    "reward_success",
    "reward_collision",
    "reward_off_route",
    "reward_timeout",
    "reward_step_cost",
    "reward_progress",
)


class GeneralizedRewardShapingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        success_reward: float = 10.0,
        collision_reward: float = -10.0,
        off_route_reward: float = -10.0,
        timeout_reward: float = -5.0,
        progress_scale: float = 0.02,
        step_cost: float = 0.01,
    ):
        super().__init__(env)
        self.success_reward = float(success_reward)
        self.collision_reward = float(collision_reward)
        self.off_route_reward = float(off_route_reward)
        self.timeout_reward = float(timeout_reward)
        self.progress_scale = float(progress_scale)
        self.step_cost = float(step_cost)
        self._prev_distance: float | None = None
        self._episode_branches = {key: 0.0 for key in REWARD_BRANCH_KEYS}

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_distance = self._ego_distance()
        for key in REWARD_BRANCH_KEYS:
            self._episode_branches[key] = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = 0.0
        branch_success = 0.0
        branch_collision = 0.0
        branch_off_route = 0.0
        branch_timeout = 0.0
        # 终局：info 字段只在终止步为 True。success/collision/off_route 互斥，但
        # max_time 是纯 raw 步计数器、可与任一终局同时为 True（恰在第 600 raw 步
        # 到达/碰撞时）。故用 if/elif 优先级（success > collision > off_route >
        # timeout），避免 success+max_time 被错加成 +5 而非 +10。
        if bool(info.get("is_success")):
            branch_success = self.success_reward
        elif bool(info.get("collision")):
            branch_collision = self.collision_reward
        elif bool(info.get("off_route")):
            branch_off_route = self.off_route_reward
        elif bool(info.get("max_time")):
            branch_timeout = self.timeout_reward
        shaped = branch_success + branch_collision + branch_off_route + branch_timeout
        # 过程：生活成本 + 前进梯度。
        branch_step_cost = -self.step_cost
        shaped += branch_step_cost
        cur = self._ego_distance()
        branch_progress = 0.0
        if self._prev_distance is not None and cur is not None:
            branch_progress = self.progress_scale * float(cur - self._prev_distance)
            shaped += branch_progress
        if cur is not None:
            self._prev_distance = cur
        # 记录 episode 累计分支（Monitor info_keywords 在 episode 结束时采样该值）。
        self._episode_branches["reward_success"] += branch_success
        self._episode_branches["reward_collision"] += branch_collision
        self._episode_branches["reward_off_route"] += branch_off_route
        self._episode_branches["reward_timeout"] += branch_timeout
        self._episode_branches["reward_step_cost"] += branch_step_cost
        self._episode_branches["reward_progress"] += branch_progress
        info = dict(info)
        for key in REWARD_BRANCH_KEYS:
            info[key] = self._episode_branches[key]
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


__all__ = ["GeneralizedRewardShapingWrapper", "REWARD_BRANCH_KEYS"]
