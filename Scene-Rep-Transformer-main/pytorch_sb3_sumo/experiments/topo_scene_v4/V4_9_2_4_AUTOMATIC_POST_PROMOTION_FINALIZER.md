# v4.9.2.4 自动 promotion 后处理接力

状态：工程实现与组合回归通过，后台接力已启动  
科学模型变化：无  
训练/评估协议变化：无  
Formal test：不启动

## 目的

v4.9.2.2 会尝试当前剩余作业，v4.9.2.3 watcher 会在它退出后使用短路径自动重试
所有仍未解决的逻辑作业。v4.9.2.4 再等待 v4.9.2.3 watcher 完整退出，然后自动：

1. 在独立输出路径重建运动学安全/模型梯度审计；
2. 不带 `--allow-partial` 运行完整六配对 attribution；
3. 汇总 12-job matrix、6-pair attribution、promotion gate 与 formal 未访问状态；
4. 根据完整证据只标记下一阶段，不启动 formal、不生成新模型。

## 失败语义

两个分析都必须被尝试。第一个分析失败不会取消第二个分析；最终报告同时记录每个
return code、stdout/stderr 尾部和证据完整性。只有 12/12 accepted、6/6 pairs、
formal untouched 时，报告才标记 `ready_for_scientific_decision=true`。
若证据 JSON 缺失或损坏，也必须把读取错误写入最终报告，而不是在汇总阶段异常退出。

## 科学边界

该层只处理证据，不修改 run tree、checkpoint、实验 contract、训练参数或 decoder。
若 promotion gate 失败，下一 owner 为 `ccf-idea-optimizer`，依据完整归因创建纯学习
式 v4.10；若 gate 通过，也只提示审核 unlock，仍不由该脚本直接启动 formal。

## 已启动接力

- watcher PID：`26396`
- watcher 创建时间：`1787841768.5861342`
- 等待对象：v4.9.2.3 watcher PID `44752`
- 等待对象创建时间：`1787834143.884075`
- 轮询周期：30 秒；最长等待：120 小时
- stdout/stderr：`results_topo_v4_9_2_promotion/engineering_finalizer_v4_9_2_4/`
- 验证：5 个定向测试及 28 个接力链组合回归全部通过
