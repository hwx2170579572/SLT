# 独立 v2 扩展：5 方法 × 6 场景 × 100 回合

本实验保留独立 v2 的单训练种子、50k raw-step、确定性评估和同一物理 80/20
流量划分，比较以下五种方法：

- MST+SLT
- TemporalGraph
- Full+BalancedSlots
- v4.8
- v4.13

`Initial Full` 与 `v4.1` 不在本矩阵中。六个场景是发布/重建的固定场景：
Left-turn、Cross、Roundabout-A、Roundabout-B、Roundabout-C 和 CARLA。

## 车流设置

- Cross：沿用独立 v2 的 1.50× 高车流；
- Roundabout-C：沿用独立 v2 的 1.25× 高车流；
- CARLA：沿用独立 v2 的机动车/行人 1.35× 高车流；
- Left-turn：1.30×，克隆出发抖动 4–6 秒；
- Roundabout-A：1.25×，克隆出发抖动 2–4 秒；
- Roundabout-B：1.20×，克隆出发抖动 1.5–3 秒。

新增三场景不是统一套倍率：其原生 route 在前 150 秒平均约有 40、44、76 辆车，递减
倍率后预计分别约为 52、55、91.2 辆，即新增约 12、11、15 辆。这样三者都形成明确的
高车流压力，同时控制 Roundabout-B 因原生负载最高而出现插入阻塞的风险。六场景全部
使用只增不改的独立 overlay；原始 route 文件保持只读且不修改。

## 不重复运行合同

独立 v2 中 MST+SLT、TemporalGraph、Full+BalancedSlots、v4.8 在 Cross、
Roundabout-C、CARLA 上的 12 个 100 回合结果只通过路径与 SHA-256 收据引用，不复制、
不重新训练、不重新评估。新运行仅有：

- 四个旧方法 × Left-turn/Roundabout-A/Roundabout-B：12 个任务；
- v4.13 × 六场景：6 个任务。

共 18 个新训练/评估任务；完整汇总仍是 30 个方法–场景单元、3,000 个评估回合。
v4.13 的正式 test 分区保持锁定，本实验只使用与独立 v2 对齐的物理 holdout，并标记为
独立扩展证据。

## 独立命名

- 协议：`experiments/independent_v2_five_methods_six_scenarios_100ep_v1/`
- 环境：`envs/sumo/independent_v2_five_methods_six_scenarios_100ep_v1.py`
- 单任务训练：`tools/train_independent_v2_5m6s100e_v1.py`
- 编排：`tools/run_independent_v2_5m6s100e_v1.py`
- run 前缀：`i5m6s100_v1__`
- overlay 后缀：`_i5m6s100_v1.rou.xml`
- 结果：`results_iv2_5m6s100e_v1/`

## 命令

```powershell
D:\Programs\Anaconda\envs\llm_pipeline\python.exe tools/run_independent_v2_5m6s100e_v1.py plan

D:\Programs\Anaconda\envs\llm_pipeline\python.exe tools/run_independent_v2_5m6s100e_v1.py adopt

D:\Programs\Anaconda\envs\llm_pipeline\python.exe tools/run_independent_v2_5m6s100e_v1.py preflight `
  --device cpu --workers 1

D:\Programs\Anaconda\envs\llm_pipeline\python.exe tools/run_independent_v2_5m6s100e_v1.py run `
  --device cuda --workers 1

D:\Programs\Anaconda\envs\llm_pipeline\python.exe tools/run_independent_v2_5m6s100e_v1.py launch `
  --device cuda --workers 1

D:\Programs\Anaconda\envs\llm_pipeline\python.exe tools/run_independent_v2_5m6s100e_v1.py status

D:\Programs\Anaconda\envs\llm_pipeline\python.exe tools/run_independent_v2_5m6s100e_v1.py summarize `
  --require-complete
```

运行器以单个方法–场景任务为恢复单位。通过完整收据校验的任务会跳过；存在但不完整或
哈希不匹配的目录不会被覆盖，而会显式拒绝并保留供审计。
