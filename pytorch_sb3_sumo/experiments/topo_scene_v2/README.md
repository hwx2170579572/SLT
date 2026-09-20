# Topology-Temporal v2 实验队列

本目录只管理新版方案。旧版实现、旧矩阵和旧结论不被覆盖。

## 固定研究问题

v2 检验的是一个有边界的设计原则：保留 TemporalGraph 主干，把拓扑限制为可退化到零的路线兼容残差，并用批统计约束替代单样本硬归一化。历史 Full+BalancedSlots 的结果只用于提出假设，不能充当 v2 的正式测试。

## 流量三分区

对按字典序排列的 traffic XML，索引 `mod 5 == 0` 用作开发验证，`mod 5 == 1` 冻结为 v2 正式测试，`mod 5 in {2,3,4}` 用于训练。CARLA 只有一个重建流量文件，无法资产互斥，所有结论必须保留该限制。

## 当前执行策略：fast-v2 成功递进筛选

原始 100k 全开发矩阵保留为冻结设计，但已按用户要求停止铺满。停止时有 4 个完整验收运行、4 个仅有 20k 持久检查点的中断目录和 64 个未启动任务；事实与哈希见 `results_topo_v2/causal_2x2/interruption_snapshot_20260816.json`。这些运行不会与 fast-v2 混算。

新增 `fast_iteration_contract.yaml`（SHA-256 `42b6b32c343125ca4e97cbdb26019fa0bbf63b785edd13612b996a8710090455`）采用成功递进式开发：

1. `factorial_anchor`：Cross/Double Merging 与 Roundabout-B 上做 8 个 20k、单种子 2×2 粗筛，只淘汰明显有害机制。
2. `merge_probe → query_probe → gated_probe`：每版仅 3 个 20k 诊断锚点，逐项输出成对行为/机制归因。
3. `soft_center`：先只测 `lambda=1e-3` 的 2 个任务；仅当安全且 slot 尺度改善时，才触发 4 个外侧系数任务。
4. `promotion`：只对冻结候选与 TemporalGraph 做 3 场景×2 seeds×2 方法的 50k 配对晋级门。
5. `formal_test`：晋级门通过后，才运行不变的 6 场景×2 方法×10 seeds×100k×50 episodes 正式测试。

fast-v2 必跑开发量为 0.98M raw steps，条件任务全部触发时为 1.06M；原完整开发链为 14.7M，最大开发计算量下降约 92.8%。20k/50k 证据只用于筛选与晋级，不支持正式论文结论。

### F1 锚点筛选结果（已完成）

`factorial_anchor` 已完成 8/8 个严格验收任务，调度失败为 0。硬归一化的两锚点主效应为 Success `+6.25 pp`、Collision `-4.17 pp`，预注册决策为 `not_rejected_by_screen`。但 Cross 上 Success 交互达到 `+91.67 pp`：无拓扑的 hard 路径出现 75% timeout，而 topology+hard 路径消除 timeout、同时带来更积极动作与碰撞权衡。因此这里只保留机制，不能将其表述为独立增益。

机制证据显示 hard 路径将 latent std 降低 34.57%--58.73%，但 slot scale ratio 在四组对照中仅一组改善，Roundabout-B 的 topology+hard 反而从 2.084 升至 3.988。这是继续测试后续 SoftBalancedSlots、而不是直接接受硬归一化的依据。完整数字、动作归因和证据哈希见 `results_topo_v2_fast/stages/factorial_anchor/deep_attribution.md`。下一阶段仅启动 3 个 20k `merge_probe`，尚未访问 formal test。

### F2 MERGE 筛选结果（已完成）

`merge_probe` 已完成 3/3 个严格验收任务。Cross/Roundabout-B 相对 Topology-v1 的配对宏观变化为 Success `+37.50 pp`、Collision `+4.17 pp`、Return `+0.333`、Timeout `-41.67 pp`，预注册结论为 `screen_positive`。Cross 的提升主要来自把 10/12 timeout 转为 8/12 success 与 4/12 collision，属于决策活性提升伴随安全权衡；Roundabout-B 则恰好改善一个 success episode 和一个 collision episode。

MERGE mass 在 Cross、Roundabout-B、CARLA 分别为 0.100、0.652、0.096，fallback 均为 0，证明新增关系实际参与计算。CARLA 仍是 12/12 timeout、100% keep，effective lanes 达 8.72，因而下一阶段只加入路线/方向 bias、反向 mask 与 compatible top-k。完整边界与数字见 `results_topo_v2_fast/stages/merge_probe/deep_attribution.md`；formal test 仍未访问。

### F3 Route/Query 筛选结果（已完成）

`query_probe` 已完成 3/3 个严格验收任务，调度失败与 stderr 均为 0。三场景 route-compatible attention mass 都为 1.00、fallback 为 0，最大 effective lanes 为 4.55，满足预注册机制门并得到 `screen_positive`。CARLA 的 effective lanes 从 8.72 降到 4.08，说明表示已成功聚焦；但它仍是 12/12 timeout、100% keep，不能把机制门通过表述为性能提升。

相对 V2 MERGE，三锚点宏观变化为 Success `-13.89 pp`、Collision `+2.78 pp`、Return `-0.167`、Timeout `+11.11 pp`。主要负向贡献来自 Cross：Success 从 8/12 降到 2/12，Collision 从 4/12 升到 6/12，并新增 4/12 timeout；Roundabout-B 则恰好改善一个 success 和一个 collision episode。与此同时 CARLA 的 slot RMS ratio 从 1.447 升到 2.206。结合 topology/goal residual scale 仍固定为 1.0，下一阶段只测试近零 residual gate 与 split pooling，检验是否能在保留定位能力的同时抑制早期特征覆盖。完整归因见 `results_topo_v2_fast/stages/query_probe/deep_attribution.md`；formal test 仍未访问。

### F4 近零 Gate 与 Split Pooling 筛选结果（已完成）

`gated_probe` 已完成 3/3 个严格验收任务，调度失败与 stderr 均为 0。相对 V3 Query，三锚点宏观变化为 Success `+11.11 pp`、Collision `-5.56 pp`、Return `+0.167`、Timeout `-2.78 pp`。Cross 从 2/12 success、6/12 collision、4/12 timeout 恢复为 7/12、3/12、3/12，是“全强度残差覆盖过强”归因的正向筛选证据；Roundabout-B 则从 12/12 success 变为 11/12 success+1/12 collision，说明效果并不均匀；CARLA 仍为 12/12 timeout、100% keep。

初始化为 0.001 的 topology gate 最终学习到 0.106--0.131，goal gate 学习到 0.016--0.029，均远低于 V3 固定的 1.0；route-compatible mass 仍为 1.00、fallback 为 0、最大 effective lanes 为 3.71，说明门控没有让 Query 机制静默失效。CARLA 的 slot RMS ratio 从 2.206 改善到 1.806，但 Cross 从 1.497 恶化到 2.452，Roundabout-B 从 1.547 恶化到 1.968，故 `v4_safe=true` 只解锁 2 个 `lambda=1e-3` 的 `soft_center` 任务，尚不解锁外侧系数。完整归因见 `results_topo_v2_fast/stages/gated_probe/deep_attribution.md`；formal test 仍未访问。

### F5 SoftBalancedSlots 中心点结果（已完成）

`soft_center` 已完成 Roundabout-B/CARLA 上 2/2 个 `lambda=1e-3` 严格验收任务。相对 V4 的两锚点宏观变化为 Success `+4.17 pp`、Collision `-4.17 pp`、Return `+0.083`、Timeout 不变；Roundabout-B 从 11/12 success+1/12 collision 变为 12/12 success，CARLA 仍为 12/12 timeout、100% keep。

预注册 aggregate balance score 从 0.5017 降到 0.1480，改善 0.3537，且安全门通过，因此触发 4 个外侧系数任务。该改善并非两场景 validation 都一致：Roundabout-B 的 eval slot ratio 从 1.968 降到 1.530、balance loss 从 0.0874 降到 0.0413；CARLA 则分别从 1.806 变为 1.829、从 0.0637 变为 0.0708。完整边界见 `results_topo_v2_fast/stages/soft_center/deep_attribution.md`。下一步只运行同两锚点的预注册 bracket，不扩充场景、seed 或系数；formal test 仍未访问。

### F6 SoftBalancedSlots Bracket 与候选冻结（已完成）

`soft_bracket` 已完成 4/4 个严格验收任务。预注册安全→成功→balance→系数排序在 V4/`1e-4`/`1e-3`/`1e-2` 中选择 `v5_soft_1e2`（lambda `0.01`）：四者均通过安全门；两锚点宏 Success 分别为 45.83%、45.83%、50.00%、50.00%；balance score 分别为 0.5017、0.2974、0.1480、0.0748。`1e-2` 与 `1e-3` 的短期结果同为 Roundabout-B 12/12 success、CARLA 12/12 timeout，最终由更优的冻结 balance score 区分。

所选候选在 Roundabout-B 的 eval slot ratio 为 1.436、CARLA 为 1.414，但仍未改善 CARLA 的 100% keep/timeout 失败模式。完整系数表与归因见 `results_topo_v2_fast/stages/soft_bracket/deep_attribution.md`，冻结选择文件为 `results_topo_v2_fast/development/candidate_selection.json`。下一阶段是 3 锚点×2 seed×2 方法的 12 个 50k promotion 任务；formal test 仍锁定且未访问。

### F7 冻结候选 Promotion（已完成，门禁失败）

12 个 50k validation 任务已全部通过严格 artifact contract，调度失败与 stderr 均为 0。Promotion 的 8 项检查按预注册模板计算：有限性、逐场景 Collision 非劣、路线注意力、fallback 和 effective-lanes 机制检查通过；宏观 Success 非劣（观测 `-0.161111`，阈值 `-0.05`）、最差配对 seed（CARLA seed 0 为 `-1.0`，阈值 `-0.15`）和至少一个场景的正效应检查失败。因此 `decision=fail`、`formal_test_unlocked=false`。

候选在 Cross 的平均 Success 为 `+0.083333`、Collision 为 `+0.033333`，但两个 seed 方向相反；Roundabout-B 的平均 Success/Collision 均为 `+0.05`；CARLA Success 为 `-0.616667`，候选 keep 达 `0.985479`、实际换道为 `0`、timeout 为 `1.0`。与此同时 route-compatible mass 最低仍为 `0.999463`、fallback 最高 `0.000537`、effective lanes 最高 `3.541154`。这把失败定位到表示之后的动作策略，而不是拓扑定位或 balance 机制失效。

完整真实结果见 `results_topo_v2_fast/stages/promotion/summary.json`（SHA-256 `db8ce421da622e09919e541c240f8a6468231cf564d2894ec44307f72e2f2c8c`）、`attribution.json`（`dc83f942d6a38fb6f7e2dfd4a89a334c24823079250d1fa4e1f90af235779837`）、`deep_attribution.md`（`95106a6dba078206bb684a232e8d5299370fdda17f0e0b6e9c640e0249f925c5`）和 `promotion_gate.json`（`4527894a32365d372bd5706d0b90a3412f9dae2349397495be60a6d558c6da8c`）。完成回执 SHA-256 为 `2ad5202515c1890304c93198d502d681ec560f80d6e30b99b65fdabc823d76b2`。

### F8 v3 动作校准止损诊断（已完成，阶段 A 即停止）

为避免重新铺开训练矩阵，新增独立 v3 协议 `experiments/topo_scene_v3/action_calibration_diagnostic_contract.yaml`（SHA-256 `d9b630d45d6f96b5273dd538b952e454e88d8694b7658d42451e9cf6edf35218`）：不训练，只把冻结候选的确定性横向输出放大 5 倍；仅运行一个 CARLA checkpoint 的 10 个新 validation seeds，只有通过活动、成功、安全门才允许一个 Cross 护栏。倍率由已有动作日志离线选择，在线不扫倍率。

CARLA 阶段把 keep 从原 seed-0 promotion 的 `1.0` 降到 `0.824752`，但 177 次非保持指令没有一次被环境应用，最终 Success `0`、Collision `0`、Timeout `1.0`、applied lane change `0`。预注册的 Success、Timeout 和 applied-lane-change 三项失败，因此 Cross 护栏、任何新训练和 formal test 均未运行。深层归因表明非保持指令只出现在前 0--31 个决策、车辆仍运动但尚不可换道的位置，之后需要换道时策略回到保持；问题是状态—动作时机错配，而非简单幅值不足。报告见 `results_topo_v3_dev/deep_attribution.md`（SHA-256 `a92d19914f61330f39fb2495d6b8e0afc595000b8c07f007946f6809cf225246`），完成回执 SHA-256 为 `01dfc05dfb182a18030258ce98dcbf2f95634d97c63a17809b3a40d05afb5054`。

全部执行结果、资源统计、未执行单元和最终边界结论已统一索引到 `results_topo_v2_fast/bounded_final_summary.json` 与 `results_topo_v2_fast/BOUNDED_FINAL_REPORT.md`；两者由 `bounded_final_receipt.json` 绑定，回执 SHA-256 为 `bf54f96b84a35440484249781dae60f6fd211b92fff6e4bb603eb728d0ee1e29`。最终完整回归为 `206 passed, 5 warnings, 99.97 seconds`。

Promotion 的 8 项闸门、6 个配对单元和正式报告格式均在结果读取前冻结。正式报告预注册为 `artifacts/topo_v2/formal_reporting_preregistration.md`（SHA-256 `098e1746e0f269767ac344d38e8c4b8e5caaf418bc024c5931f2e57543b21b74`），但由于开发门禁失败，不生成正式数据。

正式阶段硬锁另经真实负向启动验证：缺少 promotion gate 时，执行器在写 stage plan 或创建 run 目录前以 `FastProtocolError` 拒绝，且不影响正在运行的 promotion。证据为 `artifacts/topo_v2/formal_lock_negative_test.json`（SHA-256 `84eea31b43395cae76c0653c22fad84beb28ade49d6b8b9a2938a36c87ccb3df`）。

Promotion 失败后又使用当前真实 `decision=fail` gate 做了一次启动负测；执行器同样在写 plan 前以 `Formal test locked: gate did not pass` 拒绝，formal 目录仍不存在。回执为 `artifacts/topo_v2/formal_failed_gate_negative_test.json`（SHA-256 `d0283ebafb86d86192c858684f8f88598299579369e607e64a7905bac81ed725`）。

不写 formal plan 的声明预检还验证了 120 个 job name 全部唯一，且严格展开为 2 方法×6 场景×10 seeds、每项 100k raw steps/50 test episodes，候选系数为 0.01。该预检未创建 formal 目录、未打开 test 资产，证据为 `artifacts/topo_v2/formal_declaration_preflight.json`（SHA-256 `764b3a6faa1d1b6b22a574b52a0b48734f88e5a09a6eca9a6ac8d590748fc6ae`）。

Promotion 运行期间还对冻结资产做了只读重算：使用 `PAPER_SCENARIOS` 的权威路径在内存中重建完整 manifest，26 个源码、12 个网络/自车资产和 263 个交通资产（合计 301 项）与冻结 payload 完全一致。此前一次临时诊断漏拼 `traffic/` 子目录造成的“文件不存在”仅为路径误判，未改动任何文件或结果。复核收据为 `artifacts/topo_v2/freeze_manifest_revalidation_20260817.json`（SHA-256 `78fb75ceff95948bf4498a9f1975c41e455517189758a7869edfa3da7037c3da`）。

## 原冻结执行顺序（保留供审计）

1. `engineering`：所有方法×场景接口、保存/加载、SUMO 和产物契约。
2. `causal_2x2`：完整拓扑×硬 BalancedSlots 正交拆分。
3. `v2_merge`：只加入 MERGE。
4. `v3_query`：只加入路线/方向 top-k 查询。
5. `v4_gated`：只加入近零残差和路线/拓扑分池。
6. `v5_lambda`：在验证集选择 `0/1e-4/1e-3/1e-2`，允许结论为放弃 SoftBalancedSlots。
7. `candidate_validation`：冻结候选并计算开发门槛。
8. `formal_test`：只有门槛通过后，才访问 test 分区并执行 6 场景×2 方法×10 seeds×50 episodes。

原冻结方案所有训练均保持 100k raw steps。fast-v2 只在 validation 上把探索/晋级预算改为 20k/50k；正式测试仍保持 100k。SAC、奖励、动作空间、Graph-SLT 目标、actor/critic、交通资产内容和 10k 检查点间隔始终不变。

`v2_merge` 在 Cross、Roundabout-B、Roundabout-C、CARLA 上均运行，使 `v3_query - v2_merge` 在四个重点场景中保持同场景、同种子、单变量对照。每个运行同时绑定实验合同、源码/资产清单、六场景拓扑审计和 `llm_pipeline` 环境清单的 SHA-256。运行目录使用短代码规避 Windows 路径上限；完整方法名、场景名和设置仍保存在 `jobs.json` 与每个 `arguments.json` 中。

## 主要入口

```powershell
conda run -n llm_pipeline python tools\run_topo_v2_fast_iterations.py validate
conda run -n llm_pipeline python tools\run_topo_v2_fast_iterations.py plan --stage factorial_anchor --device cuda
conda run -n llm_pipeline python tools\run_topo_v2_fast_iterations.py run-development --device cuda --workers 4
conda run -n llm_pipeline python tools\run_topo_v2_fast_iterations.py status
```

`run-development` 按预注册顺序逐阶段训练、严格验收、汇总、深层归因和应用停止规则；接受的运行会跳过，已有但未验收的目录绝不覆盖。也可用 `run/summarize/decide --stage ...` 逐阶段执行。4 路并发由独立的非科学证据 600-step 基准确认，汇总仍逐运行校验完整 artifact、checkpoint CRC、动作 JSONL 哈希和真实 rollout 对齐。

正式阶段的 `run --stage formal_test` 必须读取 `results_topo_v2_fast/development/promotion_gate.json`，且其中 `decision` 必须为 `pass` 并与 fast 协议哈希一致。当前真实门禁为 `fail`，因此执行器必须拒绝 formal 启动；正式单元明确标记为“未执行”，不得填值或混入开发结果。
