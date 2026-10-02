# Random-intersection 中档车流证据（2026-09-30）

只读汇总现有 final-evaluation episode 日志和源XML；没有新增仿真或训练。每种新交通100个validation episode，medium与p05逻辑seed相同（10000–10099，SUMO seed 1000010000–1000010099）；旧sorted为SUMO seed 10000–10099。

## 建议与预热数据

建议保留 intersection_random_medium_v1 作为工作中档候选：三入口background lane 0分别400/300/480 veh/h（总1180；0.1s每步概率1/90、1/120、1/75）。p05每入口每步p=0.5、总请求54000 veh/h；它形成了明显的路外插入积压，应作为压力变体而非中档。该选择不是学习难度标定。

表中均为100回合均值（括号P10–P90）。net为在网背景车；pending为路网外待插入请求，不是道路排队；halt为在网背景速度<0.1m/s。全部300个checkpoint完整，due=inserted+pending。30/40/50s的ego_present均为false，reset初始快照在50.1s。

| 协议 | 秒 | net | due | inserted | pending | halt | 每回合mean delay跨集均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| medium | 30 | 5.30 (3–8) | 9.83 (7–13) | 9.82 (7–13) | 0.01 (0–0) | 0 | .039s |
| medium | 40 | 5.57 (3–8) | 13.23 (9–18) | 13.22 (9–18) | .01 (0–0) | 0 | .038s |
| medium | 50 | 5.65 (3–9) | 16.73 (12–21) | 16.70 (12–21) | .03 (0–0) | 0 | .040s |
| p05 | 30 | 29.10 (27–31) | 449.75 (429–470) | 50.74 (50–52) | 399.01 (377–419) | 0 | 11.742s |
| p05 | 40 | 29.31 (27–31) | 600.12 (575–624) | 65.77 (64–67) | 534.35 (508–557) | 0 | 16.164s |
| p05 | 50 | 29.19 (27–31) | 750.16 (722–774) | 80.75 (79–83) | 669.41 (642–694) | 0 | 20.595s |

最终100集成功/碰撞/超时：medium 75/25/0，p05 12/88/0，旧sorted 22/38/40。控制期terminal info中medium departed均值5.14、peak在网背景9.08、pending .01；p05为31.07、33.16、949.51。按所有回合的control departed总数/总暴露时长，插入率约1148/5407 veh/h；平均控制暴露16.12/20.69秒，因episode会提前结束，累计入场数不是固定60秒通量。另据末段完整训练episode复核，medium最后20/50集成功75%/76%，与final 75%相近；仍只有训练seed0。

## 旧协议和公平性

旧 train_intersection_yield_v2.py:119–145 对每辆背景车的XML绝对depart时刻乘4，**不是流率乘4**。intersection_sorted有30模板、每模板590辆，路由计数均为-E3→-E0 200、E0→E3 150、E2→E1 240；背景车均lane0/speed=max。原计划全局首末为0–598.49s，scale4后0–2393.96s；控制窗50–110s映回源时刻12.5–27.5s，每模板13–17个请求、均值14.47。每局只用一个模板，固定seed-42 roll后循环选择；旧train/eval共用完整30模板、无模板holdout。旧环境循环重插背景车：100集source_endless_traffic全为true，reset/terminal reinsertions均值15.38/49.79。旧active_vehicles原始字段均值reset/terminal为14.93/22.83；不将其误当成纯background数量。

三场景map/ego SHA相同，ego仍为-E1→-E0、计划50s从lane2/pos0/speed0发车。随机协议dt=.1s、控制上限600raw=60s、action repeat3、ego最多有约10s发车grace、背景请求时段[0,130)s、vehicle/pedestrian scale=1、无clone jitter、完成route即离网不循环。新场景把30模板聚合成120个车型实例和3个route-conditional driver distributions；旧每局选单一模板，且新旧seed划分/到达过程不同。因此成功率差不能归因于单个交通因素或模型。

两随机实验共同配置为SAC+MLP、seed0 fresh 100k raw、learning-starts 5000、batch32、LR1e-4、buffer20k、gamma .99、CUDA、base观测；reward shaping为成功+10、碰撞/off-route−10、timeout−5、progress .02/m、step cost .01。validation使用logical seed10000–10099，test域隔离。候选中档不等于已标定学习难度。

## 来源

完整30/40/50分位数、种子、terminal指标及模板统计见 [JSON明细](scenario_medium_calibration_20260930.json)。

旧/medium/p05评估JSONL依次位于：
D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\d0929_100k_diag\sac_mlp__intersection_sorted_depart4p0\diagnostics\eval_worker_00\episodes.jsonl；
D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\smlp_p05_a0930\sac_mlp__intersection_random_medium_v1_depart1p0\diagnostics\eval_worker_00\episodes.jsonl；
D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\smlp_p05_b0930\sac_mlp__intersection_random_medium_p05_v1_depart1p0\diagnostics\eval_worker_00\episodes.jsonl。

代码定位：train_intersection_yield_v2.py:119–145, 257–290, 392–430；paper_env.py:152–216, 319–354, 363–439, 441–536, 588–681；random_intersection.py:58–136, 233–265。协议细节见 analysis/random_intersection_v1_protocol.md。旧与新流量生成、司机车型抽样和reinsert机制不同，旧sorted的22%成功与新medium的75%不能视作可控单变量比较。
