# SUMO 等价场景

本目录是独立的 Gymnasium/TraCI 实现，不依赖 CARLA、SMARTS、TensorFlow 或
TF2RL。

目录结构：

- `sumo_env.py`：观测、动作、奖励、历史缓存、候选路径和终止事件；
- `scenario_registry.py`：6 个场景的张量与步数契约；
- `scenarios/`：各任务的 SUMO route/config；
- `networks/intersection/`：SMARTS/CARLA 左转任务使用的无信号路口；
- `networks/double_merge/`：SMARTS `cross` 对应的连续双汇入道路；
- `networks/roundabout/`：三种交通密度共用的环岛；
- `legacy.py`：需要与原 runner 对照时使用的四返回值适配器；
- `build_networks.py`：从 plain XML 重新生成 `.net.xml`。

注册的 Gymnasium ID 为 `SceneRepSUMO-<scenario>-v0`。也可以直接创建：

```python
from envs.sumo.sumo_env import SumoSceneEnv

env = SumoSceneEnv(scenario="cross", action_repeat=3)
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

SUMO 可执行文件按以下顺序解析：显式参数、PATH、`SUMO_HOME`、本机已验证的
`D:\Program Files (x86)\Eclipse\Sumo`。TraCI 优先从环境导入，缺少时只把同一
SUMO 安装的 `tools` 目录加入 Python 搜索路径。

重新生成网络：

```powershell
python envs/sumo/build_networks.py
```

这只需要 SUMO 自带的 `netconvert`，不需要安装任何被替换的仿真器。
