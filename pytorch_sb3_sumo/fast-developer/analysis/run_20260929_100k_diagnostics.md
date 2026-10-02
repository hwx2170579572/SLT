# 2026-09-29：SAC+MLP / ST-RT 正常流程内置诊断实验

状态：已启动，两方法分别从零连续训练；本记录不是完成报告。启动后的首轮检查显示两者均已到 1,495 raw steps，仍处于预热阶段，updates=0 属于预期。后续状态以运行目录中的 launcher 状态、各方法 progress 和最终评估文件为准。

结果根目录：`D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\d0929_100k_diag`

| 项目 | 已核实配置 |
| --- | --- |
| 方法 | `sac_mlp`、`sac_mlp_d1_st_rt` |
| 启动时训练 PID | 53048、76980（进程退出后 PID 可能被系统复用） |
| 设备/并行 | 两个独立 CUDA 训练 worker，单张 RTX 5060 Ti |
| 初始化 | 全新模型；未加载旧 checkpoint/replay buffer |
| 预算 | 每方法 100,000 raw simulator steps |
| 预热 | 5,000 raw steps |
| 随机种子 | 0 |
| action repeat | 3 |
| 学习率 / batch size | 1e-4 / 32 |
| 交通配置 | intersection_sorted，depart_scale=4.0 |
| 检查点 | 每 10,000 raw steps |
| 最终正常评估 | 每方法 100 回合 |
| 诊断 | `behavior_diagnostics=true`；`smoke=false` |

两份实际 `arguments.json` 已逐项核实预算、预热、设备、种子、重复步数、学习率、batch size 与诊断开关。脚本默认方法含 MST+SLT 且默认预算 50k，因此本次启动显式覆盖了 methods 与 max-steps。

## 随正常流程保存的证据

每方法 `diagnostics` 下按 train 和 eval 相位保存逐 raw step 的车辆/风险信息、逐 decision 的动作与奖励、逐 episode 的结束原因和汇总、低频优化指标、运行元数据。正常评估结束后自动读取这些现成日志生成根目录 `comparison.json` 与 `comparison.md`；这一步不再运行环境。

信息含实际动作与控制、实际速度/加速度、候选冲突车辆及策略观测覆盖、恒速固定朝向 OBB TTC/CPA、碰撞事件和位置来源、基础与策略奖励、actor/critic/表征损失及采样的 Q/TD。缺失值与有效分母显式保留；预算末尾截断回合列为 incomplete，不算环境 timeout。指标定义和边界见同目录 `embedded_diagnostics_protocol.md`。

能在正常训练/评估中取得的信息直接采集；需要额外回放、干预或对照实验的问题，待本轮训练与评估完成后根据证据安排。

## 启动前验证

- 10 项诊断单测通过，相关修改语法检查通过。
- 实际 Paper/SUMO 环境同种子同动作的启用/关闭记录对照共 132 raw steps、44 decisions：观测、奖励和终止/事件信息一致；采集错误为 0。物理速度方向与相邻位置变化一致。
- 短 CUDA 管线测试每方法 300 raw steps、101 decisions、241 updates；新增五个 Q/TD 字段均实际写出。每方法自动完成 8 个评估回合，4,800 raw records、1,600 decisions，诊断错误为 0，速度/风险有效分母非零。
- 短测仅验证实现和管线，不作为模型性能结论。首次短测遇到 Windows 超长路径，已通过将输出根移至 workspace/runs 解决，未改变交通或训练语义；失败输出保留。

单个训练种子和当前共享 30 个交通文件池仅支持探索性比较，不能据此宣称多种子稳定提升或未见交通模板的泛化能力。
