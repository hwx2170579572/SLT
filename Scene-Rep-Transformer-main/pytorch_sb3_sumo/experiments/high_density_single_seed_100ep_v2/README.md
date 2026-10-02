# 高车流单种子 100 回合六方法对比 v2

本目录是独立于 `high_density_same_scene_v1` 的新实验。它复用已经验证的地图、ego
任务、密度倍率和六种方法，但将正式训练固定为 seed 0，并把每个方法×场景的最终
确定性评估改为 100 回合。

## 独立命名

- 协议目录：`experiments/high_density_single_seed_100ep_v2/`
- 环境入口：`envs/sumo/high_density_single_seed_100ep_v2.py`
- 训练入口：`tools/train_high_density_single_seed_100ep_v2.py`
- 编排入口：`tools/run_high_density_single_seed_100ep_v2.py`
- run 前缀：`hd_ss100_v2__`
- overlay 目录：`overlays_hd_ss100_v2/`
- overlay 后缀：`_ss100_v2.rou.xml`
- 结果目录：`results_hd_ss100_v2/`

以上短名仍携带唯一的 `ss100_v2` 标识，同时避免 Windows 260 字符路径上限。

因此不会覆盖或混入 v1 三种子实验的协议、场景 overlay、run 或汇总。

## 冻结矩阵

方法仍为 MST+SLT、TemporalGraph、Initial Full、Full+BalancedSlots、v4.1 和 v4.8；
场景为同地图 CARLA、Cross 与 Roundabout-C。机动车倍率分别为 1.35×、1.50×、
1.25×；CARLA 行人同步提高到 1.35×。Cross 复制车延后 8--10 秒，保留 ego 第 5 秒
出发的可行性。

正式 profile 为 6 方法 × 3 场景 × 1 seed = 18 个 50k raw-step 新训练任务；每个
任务评估 100 回合，总计 1,800 个评估回合。单种子不能估计跨训练 seed 方差，因此
汇总中的 population std 必须为 `null`，不得报告显著性或稳定性结论。

## 命令

```powershell
conda run -n llm_pipeline python tools/run_high_density_single_seed_100ep_v2.py plan `
  --profile comparison

conda run -n llm_pipeline python tools/run_high_density_single_seed_100ep_v2.py preflight `
  --device cpu --workers 1

conda run -n llm_pipeline python tools/run_high_density_single_seed_100ep_v2.py run `
  --profile comparison --device cuda --workers 1

conda run -n llm_pipeline python tools/run_high_density_single_seed_100ep_v2.py status `
  --profile comparison

conda run -n llm_pipeline python tools/run_high_density_single_seed_100ep_v2.py summarize `
  --profile comparison --require-complete
```

编排器可断点续跑：完整任务会跳过；不完整目录不会被静默覆盖。任何尚未完成的正式
结果都保持 `TBD/null`，不得用 v1 或普通密度历史结果补格。
