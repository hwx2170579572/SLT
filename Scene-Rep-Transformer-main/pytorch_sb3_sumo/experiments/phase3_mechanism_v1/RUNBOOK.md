# 第三阶段运行与交付入口

所有命令在 `pytorch_sb3_sumo` 项目目录、已配置的 `llm_pipeline` Python 环境执行。

## 当前运行

- `r3m1/status.json`：实时队列状态、活动 worker、任务完成数；任务条目包括选择与缓存引用，不等于新增实验数。
- `r3m1/controller.lock`：控制器 PID 与创建时间。只检查 PID 不足以区分进程号复用。
- `r3m1/processes/`：各 worker 的命令、PID、创建时间、尝试次数。
- `r3m1/train/<变体__场景>/progress.json`：真实原始步和更新数；进入优化后每 100 次环境决策才更新，更新间隔长不单独视为停滞。
- `r3m1/logs/`、`r3m1/failures/`、`r3m1/retries/`：标准输出、错误和恢复记录。
- `r3m1/launch_record.json`：首轮启动和自动跟进记录。

控制器保持最多 3 个 GPU worker、0 个 CPU worker。训练和评估共用这三个槽位。常规分析和绘图不启动额外训练/仿真 worker。

## 命令

```powershell
python -m tools.phase3_mechanism_v1.cli run
python -m tools.phase3_mechanism_v1.cli analyze
python -m tools.report_phase3_mechanism_v1 --include-running
```

`run` 仅在检查当前控制器确已停止后执行；它校验封存代码，接管仍活跃的原 worker，并复用完整结果。完整评估回合按证据校验跳过。部分训练不允许静默覆盖，若发生中断须根据日志与可用检查点单独制定恢复版本并记录原因。

运行中图只能用 `--include-running` 生成，并保留明确标记。全部训练完成后执行：

```powershell
python -m tools.report_phase3_mechanism_v1
```

图表入口 `r3m1/figures_v1/README.md`，内容寻址的各次快照保留；最终分析入口 `r3m1/analysis/REPORT.md`。

## 自动跟进

本任务心跳 `automation-2` 每 30 分钟检查一次。先读状态、迭代记录和错误，再继续尚未完成的工作。进度无实质变化时不重复通知；出现失败、阶段完成、明确诊断或需要用户决定时更新。

控制器在整批结束后自动汇总结果；心跳继续核验去重、实验完整性、图表和科学解释。450 个任务条目全完成不能替代报告与方法判断。完成本批交付后暂停此心跳。

## 交付判断

1. 12 个新增训练完成 50k 原始步并通过最终模型保存/加载验证；已有完整方法不重训。
2. 30 个配置均有五检查点验证和封存选择记录；新评分的选择模型、最后模型和可用历史模型来源清楚。
3. 机制、解码、训练/评估交通、风险/梯度以及密度验证的预注册任务完整，重复模型对应的等价评估已去重。
4. 按场景、按参数报告曲线与闭环结果；完整六场景才计算等权均值。
5. 给出保持机制、缩小主张或新版本迭代的证据理由。多训练种子确认仍暂缓，当前数据不作为未来未触及的最终测试。
