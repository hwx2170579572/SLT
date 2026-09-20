# 独立 v2 已有模型：6×3×100×100 评估

本实验只复用 `high_density_single_seed_100ep_v2` 已完成的 18 个方法–场景部署模型，
不重新训练，也不修改或覆盖独立 v2 的协议、模型、场景 overlay 和结果。

正式矩阵为：

- 6 方法：MST+SLT、TemporalGraph、Initial Full、Full+BalancedSlots、v4.1、v4.8；
- 3 场景：CARLA、Cross、Roundabout-C；
- 每个方法–场景单元 100 个逻辑测试种子 block；
- 每个 block 100 个确定性评估回合；
- 共 18 个模型部署、1,800 个 block、180,000 个评估回合。

每个逻辑测试种子 block 使用 100 个显式、连续且互不重叠的 episode seed。相同场景下
六种方法共享完全相同的 seed block；每个 block 完成后原子落盘，因此中断后最多重跑
当前 100 回合，不会丢失已完成 block。

## 独立命名

- 协议目录：`experiments/independent_v2_existing_models_100seeds_100episodes_v1/`
- 环境入口：`envs/sumo/independent_v2_existing_models_100seeds_100episodes_v1.py`
- 单任务评估入口：`tools/evaluate_independent_v2_existing_models_100s100e_v1.py`
- 编排入口：`tools/run_independent_v2_existing_models_100s100e_v1.py`
- run 前缀：`iv2e100x100_v1__`
- overlay 后缀：`_iv2e100x100_v1.rou.xml`
- 结果目录：`results_iv2_eval_100s100e_v1/`

v4.8 不强制改用 `final_model.zip`。评估器读取独立 v2 原
`paper_evaluation_detailed.json` 中封存的模型路径，因此保留原 selector 选择的
`selected_model.zip` 和部署 decoder。

## 命令

```powershell
conda run -n llm_pipeline python tools/run_independent_v2_existing_models_100s100e_v1.py plan

conda run -n llm_pipeline python tools/run_independent_v2_existing_models_100s100e_v1.py source-audit

conda run -n llm_pipeline python tools/run_independent_v2_existing_models_100s100e_v1.py preflight `
  --device cpu --workers 1

conda run -n llm_pipeline python tools/run_independent_v2_existing_models_100s100e_v1.py run `
  --device cuda --workers 1

# 如果另一个实验调度器仍在运行，可精确排队到该 PID 退出后再启动：
conda run -n llm_pipeline python tools/run_independent_v2_existing_models_100s100e_v1.py launch `
  --device cpu --workers 6 --wait-for-pid <PID>

conda run -n llm_pipeline python tools/run_independent_v2_existing_models_100s100e_v1.py status

conda run -n llm_pipeline python tools/run_independent_v2_existing_models_100s100e_v1.py summarize `
  --require-complete
```

完整实验耗时很长；`run` 可安全重复执行并从缺失 block 继续。`launch` 可在 Windows
后台隐藏运行；若配置 `--wait-for-pid`，收据会先标记为 `queued`，等待的是 PID 与创建时间
共同标识的精确进程，不会因 PID 复用而误判。若发现已有 block 的协议、模型 hash、seed、
路由轮换起点或指标无法重算一致，评估器会拒绝覆盖并显式报错。全部 1,800 个 block
完成后会自动写出 execution receipt 和四份正式汇总文件。
