# v4.7.1：Selected-Deployment Metadata Binding Patch

状态：`preregistered_before_patch_implementation_and_retry`  
科学父版本：`v4.7_horizon_correct_16_step_credit`  
补丁类型：纯工程证据封装修复；无科学方法、协议或门禁变化

## 1. 触发原因

v4.7 G1 的训练、四组合 train 校准、选择器、selected-model 重封装和 12 个
validation episodes 均由子进程执行完成，子进程返回码为 0；但外层验收器以
`detailed selected decoder mismatch` 拒绝该目录。

根因定位仅检查了字段存在性和选择收据，不读取或解释 validation outcome：

- `selector/receipt.json` 已封存 `selected_deployment_decoder=target_critic`；
- `paper_evaluation_detailed.json` 缺少
  `selected_deployment_decoder`、`selected_source_checkpoint_sha256`、
  `selected_model_policy_class` 和
  `selected_model_parameter_state_sha256`；
- v4.7 适配器替换父 JSON writer 时保留了 schema 版本化和 return diagnostics，
  但没有复现冻结 v4.6 writer 对上述四个绑定字段的追加。

因此该尝试是协议无效的工程产物，不进入任何科学门禁。其 validation 数值在
补丁预注册时保持未检查、未报告、未使用。

## 2. 唯一工程变化

新增 v4.7.1 适配文件，在写入最终
`paper_evaluation_detailed.json` 前，从已经先行封存的 selector receipt 状态
复制并绑定以下四个字段：

1. `selected_deployment_decoder`；
2. `selected_source_checkpoint_sha256`；
3. `selected_model_policy_class`；
4. `selected_model_parameter_state_sha256`。

字段值不得重新推断或重新选择，只能来自已封存的部署收据。原 v4.7 训练器、
replay buffer、模型、校准、选择器、动作诊断和 evaluation 实现保持冻结不变。

## 3. 严格保持不变

- 科学 contract SHA256：
  `e97ef91a85ddeb60bee50fb6fdccdff4afb4e49f5a9d6e52ffee24431890ea7a`；
- v4.7 implementation freeze SHA256：
  `eb4526f1c9b197c2dc8259af8f3b7e2eea9c6b0b22de05054f54957fbf0082b0`；
- horizon-correct 16-step estimator、reward、网络、优化器、entropy；
- checkpoint/decoder 四组合、选择顺序和收据先于 validation 的时序；
- G1–G5 场景、训练 seed、raw-step、校准 seed、validation seed 和所有门禁；
- promotion 与 formal/test 锁。

## 4. 无效尝试隔离与重试规则

原目录
`results_topo_v4_7_dev/development/runs/G1__cand__cross__s6__pe97ef91a`
保持只读并标记为 `engineering_rejected_not_gate_eligible`。关键证据哈希：

- detailed：`6ef859c5737aa91d47523759095f5bb0bc37838f6983efc3a3555cbc2ad3364c`；
- selector receipt：`ffdfa9540f5a35d20f9e5f43819e5336b2a52a903a40b53db5ea7fe2e50379c8`。

补丁通过定向测试、全量回归、真实 SUMO smoke 和新 implementation freeze 后，
允许使用相同 G1 科学设置从头训练一次，输出到独立的
`results_topo_v4_7_1_dev/`。不得复用原 checkpoint、replay、校准或 evaluation
结果。若相同元数据绑定失败再次出现，则停止并进行版本化工程归因。

## 5. 验收

补丁 smoke 和所有新鲜运行必须同时满足：

- detailed 的四个字段均存在，并与 selector receipt 逐字段相等；
- detailed 的 selector receipt SHA256 可重算；
- selected checkpoint、selected model 和 parameter-state 哈希可重算；
- 选择收据时间早于 validation 环境构造时间；
- 原 v4.7 科学冻结文件逐哈希未变；
- formal/test 未访问。

