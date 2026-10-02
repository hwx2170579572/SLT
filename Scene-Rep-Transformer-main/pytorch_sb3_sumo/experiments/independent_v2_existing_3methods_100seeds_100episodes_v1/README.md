# 独立 v2 已有模型：3×3×100×100 评估

本实验只复用独立 v2 已完成的 9 个方法–场景部署模型，不重新训练，也不修改或覆盖
父实验和已取消的六方法实验计划。

正式矩阵为：

- 3 方法：MST+SLT、TemporalGraph 时序车辆图基线、v4.8 Tie-only Replicated Calibration；
- 3 场景：CARLA、Cross、Roundabout-C；
- 每个方法–场景单元 100 个逻辑测试种子 block；
- 每个 block 100 个确定性评估回合；
- 共 9 个部署、900 个 block、90,000 个评估回合。

每个逻辑测试种子 block 使用 100 个显式、连续且互不重叠的 episode seed。同一场景下
三种方法共享相同 seed block 和物理 held-out 20% 流量分区。路由轮换起点由逻辑 block
索引确定，因此断点续跑与不中断运行一致。

## 独立命名

- 协议目录：`experiments/independent_v2_existing_3methods_100seeds_100episodes_v1/`
- 环境入口：`envs/sumo/independent_v2_existing_3methods_100seeds_100episodes_v1.py`
- 单任务入口：`tools/evaluate_independent_v2_existing_3methods_100s100e_v1.py`
- 编排入口：`tools/run_independent_v2_existing_3methods_100s100e_v1.py`
- run 前缀：`iv2e3m100x100_v1__`
- overlay 后缀：`_iv2e3m100x100_v1.rou.xml`
- 结果目录：`results_iv2_eval_3m_100s100e_v1/`

v4.8 继续使用父 v2 `paper_evaluation_detailed.json` 封存的 `selected_model.zip` 和部署
decoder；MST+SLT 与 TemporalGraph 使用各自封存的 `final_model.zip`。

## 命令

```powershell
conda run -n llm_pipeline python tools/run_independent_v2_existing_3methods_100s100e_v1.py plan

conda run -n llm_pipeline python tools/run_independent_v2_existing_3methods_100s100e_v1.py source-audit

conda run -n llm_pipeline python tools/run_independent_v2_existing_3methods_100s100e_v1.py preflight `
  --device cpu --workers 1

conda run -n llm_pipeline python tools/run_independent_v2_existing_3methods_100s100e_v1.py launch `
  --cpu-workers 1 --gpu-workers 2

conda run -n llm_pipeline python tools/run_independent_v2_existing_3methods_100s100e_v1.py status

conda run -n llm_pipeline python tools/run_independent_v2_existing_3methods_100s100e_v1.py summarize `
  --require-complete
```

正式任务按 100 回合 block 原子落盘并可续跑。缺失或无效 block 保持显式，不会从父 v2
或旧六方法冒烟结果填补。900 个 block 全部完成后自动生成 execution receipt、JSON 汇总
以及 block、方法–场景、方法宏平均三份 CSV。

混合设备模式维护独立的持久槽位：1 个任务始终在 CPU 上执行，最多 2 个任务共享本机
CUDA 设备执行。正式启动前应在独立结果目录运行相同拓扑的 `preflight`，确认没有 CUDA
OOM、SUMO 并发或系统内存错误。
