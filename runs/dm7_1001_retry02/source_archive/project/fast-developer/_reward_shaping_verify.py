"""验证 reward shaping v2（``reward_shaping_v2.GeneralizedRewardShapingWrapper``）。

两层验证：

1. **单元验证（mock env）**：精确断言 wrapper 对四类终局 + 前进梯度 + 生活成本的
   改写数值，证明「成功 > 超时 > 碰撞」的相对序、且超时不再为 0（打破吸收盆地）。
2. **真实环境停车 smoke**：在 intersection 上把 ego 用动作 [-1, 0]（speed=0）原地
   停到 episode 结束，断言累积 return < 0——旧版（raw_reward=success-collision）下
   这个「停车等超时」策略的累积 return 恒为 0（零风险吸收盆地），新版应为负。

用法::

    python fast-developer/_reward_shaping_verify.py
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

import gymnasium as gym

from reward_shaping_v2 import GeneralizedRewardShapingWrapper


# --------------------------------------------------------------------------- #
# 1. 单元验证（mock env）
# --------------------------------------------------------------------------- #
class _DummyEnv(gym.Env):
    """可配置 info 的假环境，用于精确断言 reward 改写数值。"""

    def __init__(self):
        self.observation_space = gym.spaces.Dict()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
        self.next_info: dict = {}

    def reset(self, *, seed=None, options=None):
        return {}, {}

    def step(self, action):
        info = self.next_info
        self.next_info = {}
        return {}, 0.0, False, False, info


def _wrapper(distance=100.0):
    env = _DummyEnv()
    w = GeneralizedRewardShapingWrapper(env)
    w._ego_distance = lambda: distance  # progress 恒 0
    w._prev_distance = distance
    return env, w


def _assert_close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > 1e-9:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def test_unit() -> None:
    print("== 单元验证（mock env）==")
    # success：+10 - 0.01
    env, w = _wrapper()
    env.next_info = {"is_success": True}
    _, r, *_ = w.step(np.zeros(2))
    _assert_close(float(r), 10.0 - 0.01, "success")
    print(f"  success   -> {float(r):+.4f}  (期望 {10.0 - 0.01:+.4f})")

    # collision：-10 - 0.01
    env, w = _wrapper()
    env.next_info = {"collision": True}
    _, r, *_ = w.step(np.zeros(2))
    _assert_close(float(r), -10.0 - 0.01, "collision")
    print(f"  collision -> {float(r):+.4f}  (期望 {-10.0 - 0.01:+.4f})")

    # off_route：-10 - 0.01
    env, w = _wrapper()
    env.next_info = {"off_route": True}
    _, r, *_ = w.step(np.zeros(2))
    _assert_close(float(r), -10.0 - 0.01, "off_route")
    print(f"  off_route -> {float(r):+.4f}  (期望 {-10.0 - 0.01:+.4f})")

    # timeout：-5 - 0.01（关键：不再是 0）
    env, w = _wrapper()
    env.next_info = {"max_time": True}
    _, r, *_ = w.step(np.zeros(2))
    _assert_close(float(r), -5.0 - 0.01, "timeout")
    print(f"  timeout   -> {float(r):+.4f}  (期望 {-5.0 - 0.01:+.4f})")

    # 中间步（无终局）：仅 -0.01
    env, w = _wrapper()
    _, r, *_ = w.step(np.zeros(2))
    _assert_close(float(r), -0.01, "mid-step")
    print(f"  mid-step  -> {float(r):+.4f}  (期望 {-0.01:+.4f})")

    # progress：上一步距离 100 -> 当前 105，0.02*5 - 0.01
    env, w = _wrapper()
    w._ego_distance = lambda: 105.0
    w._prev_distance = 100.0
    _, r, *_ = w.step(np.zeros(2))
    _assert_close(float(r), 0.02 * 5.0 - 0.01, "progress")
    print(f"  progress  -> {float(r):+.4f}  (期望 {0.02 * 5.0 - 0.01:+.4f})")

    # 相对序断言：success > timeout > collision
    assert 10.0 > -5.0 > -10.0, "相对序 success > timeout > collision 被破坏"
    assert -5.0 < 0.0, "timeout 仍为 0，吸收盆地未打破"
    print("  相对序 success(+10) > timeout(-5) > collision(-10) 成立；timeout < 0。")
    print("  单元验证通过。\n")


# --------------------------------------------------------------------------- #
# 2. 真实环境停车 smoke
# --------------------------------------------------------------------------- #
def test_real_env_stop() -> None:
    print("== 真实环境停车 smoke（intersection）==")
    from train_intersection_yield_v2 import (
        _environment_namespace,
        make_env_factory,
    )

    overlay = _THIS_DIR / "_reward_verify_overlay"
    shutil.rmtree(overlay, ignore_errors=True)
    factory = make_env_factory("base", overlay)  # base=mst_slt 环境，包了 wrapper
    env = factory(_environment_namespace(), evaluation=False)
    total = 0.0
    steps = 0
    ended = None
    try:
        env.reset(seed=10_000)
        for _ in range(400):
            # 动作 [-1, 0] -> speed=0、lane=0，让 ego 原地停车（direct 模式）。
            _, r, terminated, truncated, info = env.step(np.array([-1.0, 0.0]))
            total += float(r)
            steps += 1
            if terminated or truncated:
                ended = ("terminated" if terminated else "truncated")
                break
        print(f"  停车 {steps} decision step 后 episode 结束（{ended}）")
        print(f"  累积 return = {total:+.4f}")
        print(
            f"  info: is_success={bool(info.get('is_success'))} "
            f"collision={bool(info.get('collision'))} "
            f"off_route={bool(info.get('off_route'))} "
            f"max_time={bool(info.get('max_time'))}"
        )
        if total >= 0.0:
            raise AssertionError(
                f"停车等超时的累积 return 应为负（打破吸收盆地），实际 {total:+.4f} >= 0"
            )
        print("  停车累积 return < 0，吸收盆地已打破。")
        print("  真实环境 smoke 通过。\n")
    finally:
        env.close()


def main() -> int:
    test_unit()
    test_real_env_stop()
    print("结论：reward shaping v2 生效 —— timeout 为负、成功压倒性正、停车不再无代价。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
