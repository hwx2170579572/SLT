# v4.11.2 Windows 短路径不可变恢复

## 结论与边界

v4.11.2 是 v4.11 PRCR promotion 的工程执行恢复版本，不是新的科学方法。
它只缩短新恢复作业的运行目录和日志目录；模型、损失、动作候选、推理解码、
训练预算、随机种子、数据分区、评价回合、校准协议、实验契约哈希和 promotion
门禁均保持冻结的 v4.11 值。它不添加运动学安全投影、TTC/headway 阈值、
车道 veto、安全盾、规则回退、tie override 或动作重写。

## v4.11.1 失败归因

v4.11.1 已正确执行“单项失败不短路”：Cross TemporalGraph seed21 的首次恢复
失败后，runner 继续进入 RAM seed20。该失败发生在 TensorBoard 创建 event 文件
时，报错为 `FileNotFoundError`，失败 event 路径长度为 278 个字符。它发生在
模型训练开始前，因此是 Windows 旧路径预算问题，不是 PRCR 或 TemporalGraph
的效果证据。

失败目录与日志保持原位，不删除、不覆盖：

- 失败恢复目录：
  `results_topo_v4_11_promotion/engineering_recovery_v4_11_1/runs/P__tg__cross__s21__pc71e98e7__recovery_r1`
- 失败日志 SHA-256：
  `d6258987980bc6d2ea2c9e5e499aaa705129fbb9f38a522f699fc15700efb812`

## 短路径恢复机制

新增 `tools/run_all_v4_11_2_pipeline.py`，复用已测试的 v4.11.1 不可变恢复引擎，
但将新恢复目录放在 `r4112/r/rN_<logical-job-hash>`，日志放在 `r4112/l`。
代表性 TensorBoard event 路径为 195 个字符，比失败路径短 83 个字符，并低于
测试采用的 240 字符保守预算。完整逻辑作业身份及命令仍保存在版本化报告中。

适配器在一次调用期间临时配置 v4.11.1 引擎，退出后恢复所有共享函数与路径，
避免全局 promotion 注册器按版本顺序运行时产生进程内污染。执行策略仍为：

1. 复用已通过冻结验收器的 canonical 作业；
2. 保留 canonical 和旧 recovery 的所有不完整证据；
3. 对每个未解决逻辑作业从头执行，不续接无法证明随机状态连续的 checkpoint；
4. 单项失败或 launcher 异常不取消其余作业；
5. 所有尝试结束后才 source-map、汇总并调用一次冻结 v4.11 promotion gate；
6. promotion gate 未通过前不访问也不启动 formal。

## 工程验证

- Python 编译：通过；
- v4.11/v4.11.1/v4.11.2 联合自动化测试：17 passed；
- dry-run：12 个逻辑作业，1 个已验收复用、11 个计划执行；快照中 2 个计划为
  short recovery，其中一个是已知长路径失败的 Cross seed21，另一个是 dry-run
  时仍在运行、尚无最终产物的 RAM seed20；实跑会在前序批次结束后重新验收，
  已完成者不会重复训练；
- recovery script SHA-256：
  `afb00ff09a45b298b69719817a98465cb54909cacb2c4ac4c76db4b6dd695f5c`；
- scheduler script SHA-256：
  `8223375bf228452d574a8a453294bd157ba926412f049023d176db6d6606ef44`；
- recovery test SHA-256：
  `114ec63b25d9f2656fd5e8faf2bb8255312e25b4a73ebfadcfd774952d1a0192`；
- dry-run receipt SHA-256：
  `5ccc7708be1d53bb6acbfe4f2c8a8171b6b0eb17aa8a44e13cd8a6bb10114a20`。

## 脱离 Codex 会话的顺序调度

v4.11.1 attempt-all runner 已通过隐藏 `Start-Process` 作为 PID 21320 独立运行。
新的隐藏等待进程 PID 14536 在启动工具调用结束后仍存活，等待 PID 21320 完整
退出，然后执行 v4.11.2。其 stdout/stderr 分别写入：

- `results_promotion_automation/v4_11_2_waiter_r2.stdout.log`
- `results_promotion_automation/v4_11_2_waiter_r2.stderr.log`

因此 Codex 的“中断会话”只停止当前交互，不会向这两个独立进程发送终止信号。
第一次等待器启动 PID 32948 因带空格的 `-File` 路径未加引号而立即失败，其日志
保留；r2 使用显式引号后已稳定等待，不覆盖 r1 启动证据。

## 科学决策规则

只有 12 个 promotion 逻辑作业全部完成并通过冻结完整性验收后，才可解释成对
效果。若门禁失败，必须基于完整六对场景×种子结果做深层模型归因并创建新的
科学版本；不得以运动学规则或安全投影掩盖模型缺陷。
