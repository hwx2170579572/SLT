# v4.7.2：Parent-Preregistration Context Isolation Patch

状态：`preregistered_before_runner_patch_and_retry`  
科学协议：冻结 v4.7，完全不变  
训练与 evidence-writer：冻结 v4.7.1，完全不变

## 1. 触发原因

v4.7.1 已通过 25 项定向测试、327 项全量回归和真实 SUMO metadata-binding
smoke，并以 implementation freeze
`979080cc6a76936f572cc79d730ce23202803be976f201ab61810967b1b93359`
封存。随后首次 development dry-run 在执行任何训练前失败：

`freeze preregistration hash drifted`

根因是 v4.7.1 runner 的兼容上下文把冻结 v4.7 模块的
`DEFAULT_PREREGISTRATION` 临时替换为 v4.7.1 补丁收据。父 v4.7 freeze
复验函数按设计读取该全局路径，因而错误地用补丁收据哈希比较父收据哈希。

## 2. 单一工程变化

新增 v4.7.2 runner wrapper，复用 v4.7.1 的训练器、命令生成、产物验收和
补丁 freeze 逻辑，仅在兼容上下文中不再覆盖父模块的
`DEFAULT_PREREGISTRATION`。补丁收据哈希仍由 v4.7.2 protocol hash 显式记录，
不依赖父模块全局变量。

## 3. 保持不变

- v4.7 科学 contract、16-step return estimator、模型和所有 seeds/gates；
- v4.7.1 detailed 四字段绑定 writer；
- G1 从头重试要求，不复用已拒绝尝试的任何 checkpoint 或结果；
- selector receipt 先于 validation；
- promotion 与 formal/test 锁。

## 4. 验收与输出隔离

- 单元测试必须证明修正上下文内父 `DEFAULT_PREREGISTRATION` 身份保持不变；
- `status` 与 development `dry-run` 必须同时成功；
- dry-run 唯一合法 next job 必须为 G1，训练器仍为
  `train_paper_sb3_sumo_v4_7_1.py`；
- 父 v4.7 与 v4.7.1 freeze 必须逐哈希复验；
- 新鲜结果写入 `results_topo_v4_7_2_dev/`；
- 未通过 v4.7.2 freeze 前不得启动重试。

