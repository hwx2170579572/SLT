# 手动启动与结果位置

先切换到项目 `pytorch_sb3_sumo` 目录，在 PowerShell 执行：

```powershell
.\run_v48_stability_v1.ps1 -CheckOnly
.\run_v48_stability_v1.ps1
```

第二行才启动正式实验。默认同时两个CUDA工作进程，使用当前GPU；两个进程并不要求两张GPU卡。没有CPU工作进程。脚本依赖已经验证的Python环境 `D:\Programs\Anaconda\envs\llm_pipeline\python.exe`，启动时验证冻结源码、原模型/日志和协议哈希。

训练任务固定：Cross/CARLA×late_decay/batch64/target_slow。lr_half和lr_quarter复用已有模型，不训练。评估统一直接使用最终模型actor，原部署选择器/解码器全部绕过。训练完成后自动评估、汇总和出图，无需另下分析命令。

产物：

- `r48s1/analysis/REPORT.md`：最终系统比较、筛选理由、未达标项。
- `r48s1/analysis/comparison.csv`：原始回合均值/方差、曲线回撤/振荡/收敛代理、部署表现。
- `r48s1/analysis/driving_metrics.csv`：行为指标均值/std/方差、有效回合数及相对lr_half差值。
- `r48s1/analysis/cross_all.png`、`carla_all.png`：逐场景总览；同目录另有每候选与基准的单独图，均提供PNG/PDF/SVG。
- `r48s1/train/<cell>`：新训练模型、优化轨迹、监视日志、检查点（仅追溯，评估不用选择器）。
- `r48s1/eval/<cell>/eNNN.json`和`.json.gz`：逐回合结果及原始遥测。
- `r48s1/logs`与`controller_progress.json`：进度和失败日志。

可在另一个终端查看状态：

```powershell
& 'D:\Programs\Anaconda\envs\llm_pipeline\python.exe' -m tools.v48_stability_v1.cli status
```

更新当前已有数据的分析（不启动训练）：` .\run_v48_stability_v1.ps1 -ReportOnly `。数据不完整时报告明确标记，不给最终推荐。

重复启动会跳过已完成训练和已完成评估回合；部分训练保留原文件并拒绝静默重训，因为未保存完整回放/RNG恢复状态。遇到部分训练、OOM或数值问题，需要先诊断再决定恢复方案，不能覆盖旧结果。Ctrl+C终止本控制器的子进程并保留已有产物。不要同时启动两份正式控制器。

`r48s1/smoke`只含准备阶段的工程检查，使用独立930xxx种子及96步短训练；不参与科学曲线、评估或排名。准备命令 `prepare/test/parity/smoke/seal` 面向维护流程，用户正常运行仅需上面的手动启动命令。
