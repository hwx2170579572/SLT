# v4.11.1 Promotion 中断恢复方案

## 分类

v4.11.1 是工程执行恢复层，不是新的科学方法。科学版本仍为 v4.11
Proper-Calibrated Ranked Risk（PRCR）。本恢复层不改变模型、损失、动作候选、
推理解码、训练预算、随机种子、数据分区、校准协议、评价回合数或 promotion
门槛，也不添加运动学投影、TTC/headway 阈值、车道 veto、安全盾、规则回退或
动作重写。

## 中断事实

逻辑作业 `P__tg__cross__s21__pc71e98e7` 在一次 Codex 长会话中断后只留下
半成品：最后一条聚合训练记录为 32,783/50,000 raw steps，缺少
`training_diagnostics.json`、最终 checkpoint、选择器收据和 30 回合 fresh
validation。冻结 v4.11 runner 因此正确地将其判为
`required artifact missing: training_diagnostics.json`，不会把半成品纳入门控。

中断证据保持原位且不覆盖：

- launcher log SHA-256：`bbd2e7008a3238eb3f97d1fbf4809046f88cc93c1a102b443093c048a3443049`
- arguments SHA-256：`4c8c5951c3fa00ab8a707966077b7a8785985edd27fb813abd3d3411d196bab4`
- early best-training checkpoint SHA-256：`7284df6061ce33d0c7b331446d6f65f297462e8ba5db9c9580558a7299424bca`

不能从该 checkpoint 续训并把结果冒充冻结的连续 50k 训练，因为它不能证明
完整 replay buffer、环境进度和随机状态与原进程连续。因此恢复方案使用相同
预注册命令从头训练，并保留原半成品作为独立中断证据。

## 恢复机制

新增 `tools/run_all_v4_11_1_pipeline.py`：

1. 对 12 个冻结逻辑作业逐一检查 v4.11 的完整验收器；
2. 已完整的 canonical run 直接复用；
3. canonical 路径不存在时，仍在 canonical 路径运行；
4. canonical 路径存在但不完整时，不删除、不移动、不覆盖，改在
   `engineering_recovery_v4_11_1/runs/<logical-job>__recovery_rN` 从头运行；
5. 任一子作业失败或抛异常后继续尝试其余作业；
6. 全部尝试结束后，以逻辑作业到不可变接受源的 source map 汇总 12 项，再调用
   原冻结 v4.11 promotion gate；
7. 永不启动 formal 测试。

全局 `tools/run_all_pending_promotions_v4_11.py` 按既有版本发现约定自动发现
`v4.11.1`，因此当前和未来 promotion 都可继续运行且不因单项失败短路。

## 工程验证

- Python 编译：通过；
- v4.11/v4.11.1 自动化单元测试：11 passed；
- dry-run：12 个逻辑作业中 1 个完整复用、1 个使用 `recovery_r1`、10 个使用
  canonical 路径，formal launch 为 false；
- recovery script SHA-256：`a8a57ebafc4d236caf1a5691ef63628ad01bf35800736e3ed33cadfceda0144f`
- recovery test SHA-256：`586acf5ab08b231a047cca8c230f5a3bc6cdbda1e4385ef3b7add5bb786459cd`
- dry-run receipt SHA-256：`0f8ef0a13fee212aa8845b877a8943c90d92d301f2baf18f6300620d05182596`

## 科学解释边界

中断、补跑和 source mapping 只影响执行容错，不能作为模型效果证据。最终结论
只能来自 12 个逻辑作业全部通过完整性验收后的成对统计。若 gate 失败，必须对
完整六个场景×种子配对进行模型归因并创建新的科学版本；不得用规则机制补救。
