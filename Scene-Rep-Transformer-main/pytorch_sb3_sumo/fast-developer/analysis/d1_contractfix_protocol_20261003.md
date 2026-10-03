# D1 contract repairs v1 — 2026-10-03

Status: implementation, independent review, checkpoint roundtrip, launch guard and worker-traffic isolation verified; 31 pure/integration test cases passed. The ST-RT/Longres predecessor pair is complete; the two fresh contractfix workers are running under runs/d1_contractfix_20261003. No repaired-method results exist yet.

## Authorization and comparison

The user requested a new D1 method version repairing C8 (last-valid history selection) and C9 (the coordinate contract of explicit geometric edges), followed by fresh ST and ST-RT training. Preserve all legacy method names, source behavior and historical results.

Planned new methods:

- `sac_mlp_d1_st_contractfix_v1`
- `sac_mlp_d1_st_rt_contractfix_v1`

Primary scene: `intersection_sorted_depart4p0` (`intersection_sorted`, depart scale 4.0). Each arm: seed 0, fresh initialization, 100000 raw steps, CUDA; final checkpoint validation on seeds 10000–10099, 100 real episodes. Keep existing reward, action repeat, replay, optimizer and bootstrap protocol. In particular, this experiment does not also repair the audited single-gamma n-step target or change timeout semantics.

Do not launch until the valid old ST-RT arm in `runs/sortlr_1003_retry01` and the valid longres arm in `runs/sortlr_1003_retry01_longres` have completed training and their final evaluations. The failed candidate inside the first root and the manually interrupted `runs/sortlr_1003` are not valid arms and must not be merged into results. Check method-specific artifacts, not the first launcher's aggregate failed status.

## C8: actual last valid index

For a Boolean history mask m over H slots, select max{i | m[i]} rather than count(m)-1. Return a separate has-valid flag. An empty history has no meaningful selected time: safe gathering must be followed by explicit zero/invalid handling, including temporal attention with no valid keys. Both the route-query selector and the temporal pooling-query selector must use the repaired operation.

Test left padding, right padding, internal gaps, all-valid, all-invalid, different history lengths and supported observation layouts; exercise the actual encoder paths, not just a standalone helper. Preserve the legacy presence-mask contract for this scoped repair. Current x!=0 masks are a proxy; x=0 with other nonzero fields is a known separate ambiguity and must be counted/reported rather than silently changing the observation contract.

Normal trajectory diagnostics must use actual rollout observations and explicit denominators. Encoder-forward counts include replay reuse, actor/critic duplication and shadow variants and cannot be reported as the frequency in training trajectories. Separate short nonempty history, empty history, old/new index mismatch, old selection on padding, and histories with internal gaps. Distinguish episode-start prefixes from left-padded suffixes. Retain phase, episode/decision identity and raw-step context. These measurements add no simulator steps.

## C9: Cartesian velocity for geometric edges only

Local implementation evidence: `sumo_env._state` retains SUMO east/north x/y, uses SMARTS h=-SUMO-angle radians, and stores pseudo velocity (s cos h, s sin h). Actual Cartesian east/north velocity is (-s sin h, s cos h), i.e. (-pseudo_y, pseudo_x). This agrees with the [official SMARTS heading convention](https://smarts.readthedocs.io/en/latest/api/smarts.core.coordinates.html) and [SUMO navigation-angle convention](https://eclipse.dev/sumo/docs/TraCI/Change_Vehicle_State.html).

Recover this velocity independently for explicit geometric edges. Keep original state tokens/learned channels unchanged. A common proper 2D rotation does not correct a mismatch between the original position and velocity bases; conversion can commute with that rotation, but must not be applied indiscriminately to Cartesian or reflected observation contracts.

For relative position r and relative Cartesian velocity v, t*=-r·v/(v·v) is the unconstrained time of closest approach under constant relative velocity. A finite future horizon requires clipping time to [0,T] and computing the corresponding miss distance ||r+v t||. Zero relative velocity has no unique closest-approach time. These point-trajectory proxies alone are not true TTC: vehicle extent, trajectory validity and the future horizon matter. Keep the existing edge width/parameterization for this repair; do not introduce an unrequested risk module or extra input dimensions.

Analytic checks: following, head-on, perpendicular crossing, parallel offset and stationary relative motion; cardinal headings, approach/recede signs, common-rotation invariance, validity masks and finite outputs. Diagnostics use precise proxy names, expose invalid/undefined cases, and distinguish effective edge use from simply executing an edge builder.

## Evidence and interpretation

These are correctness repairs, not demonstrated performance improvements. Old results remain valid observations of their archived implementation. Jointly repairing C8 and C9 does not identify each repair's individual performance contribution; normal exposure/sensitivity measurements can establish whether each path is exercised, not a success-rate causal effect. Do not claim improvement until the authorized new runs finish.

The completed frozen-window baseline audit found zero count-index/true-index mismatch in its 1057 decisions (12 selected episodes); that does not establish the frequency across training or all validation episodes. It also prevents attributing those selected baseline collisions directly to C8 without additional evidence.

Record final checkpoint SHA, runtime provenance/source archives, seed/no-resume, raw/decision/update counts, parameter counts, actual CUDA device, reward-contract identity, shaped vs raw return and six-component reconciliation. Preserve ordinary diagnostics and unique-state shadow budgets (training entire phase at most 20 unique states, evaluation at most four per episode). Do not count probe variant rows as episodes or add simulator evaluation episodes merely to collect these statistics.

## Implementation and verification ledger

- New isolated encoder: `algos/sb3_torch/contractfix_encoder.py`. The additional ego-rotation anchor selector is also covered; repairing only the route and temporal helpers would leave another count-based selector in the inheritance chain.
- Both ST and ST-RT construct geometry/motion edges even with topology relations disabled. C9 is therefore active in both new methods.
- Passive trajectory collector: `fast-developer/contractfix_trajectory_audit.py`. It caches each reset/next observation and records it when the next real action is stepped. It never steps the environment for an audit. The post-terminal observation is not counted as another policy decision. Existing cached actor IDs/history lengths/first-seen times provide a separate identity-based history audit.
- Per-phase outputs: `diagnostics/{phase}/trajectory_history_audit.jsonl` and `trajectory_history_audit.summary.json`. The summary stores sums and denominators, errors, raw-step accounting, started episodes and completed episodes separately. A final automatic reset without an action can increase started episodes without increasing evaluation episodes.
- Geometry diagnostics use unordered valid actor pairs at matching history slots; repeated historical samples across decisions are explicitly repeated observations, not independent traffic events. No vehicle-extent collision threshold is introduced.
- Pure tests passed together in the pinned Python environment: `tests_sb3_sumo.test_contractfix_encoder` (14 tests) and `tests_sb3_sumo.test_contractfix_trajectory_audit` (7 tests), 21/21 on 2026-10-03. These are CPU/pure-environment tests, not formal SUMO results. Legal actor/history-size variants and additional Dict keys are covered; state-LSTM-only, flattened trajectory and whole-Box observations are explicitly rejected instead of guessed.
- Independent trajectory-audit review confirmed that `env.unwrapped` exposes the aligned `PaperSumoSceneEnv` history cache, the wrapper sits before vectorization/normalization, and `raw_steps_executed` counts actual control steps without reset warmup. The wrapper forwards the same observation/info objects and original action and invokes one environment step per decision. Unique run-local paths are required: JSONL is append-only with session IDs, while its summary describes the latest wrapper session.
- Review caught and corrected the new audit's age formula before any formal use: `_history_timestep` is the next append index, so tracked age is `tick - first_seen`, without an additional one. A regression assertion checks first-observation age equals one.
- Independent encoder review found no blocking defect for ST/ST-RT. A further inherited count-based `_last_query_mask` on the currently disabled topology branch was also overridden in the new class and tested; legacy parent code remains unchanged.
- Same-seed old/new encoder checks for each ST/ST-RT configuration: 1,057,348 encoder parameters, identical state-dict keys and initial tensors, and identical post-construction RNG state. This is an encoder count, not the total actor-plus-critic parameter count. The only intended forward changes are the requested index and edge-coordinate repairs.
- The guarded launcher gives the two workers distinct short roots under the global pair root: `st` and `st_rt`. Each command's `--run-root` is its own subroot; `_traffic_paths_for_scale(4.0)` therefore writes its scaled route pool under that worker root, and the D1 environment derives train/eval overlays under that same root's `_hd/<method>/ns_tr` and `_hd/<method>/ns_eval`. The immutable sorted source traffic pool is shared as input. This prevents either worker from replacing the other's `traffic_*.rou.xml` or temporary overlay target; a pure launcher test asserts command, scaled-pool and overlay-root separation.

Pinned test command (from `pytorch_sb3_sumo`):

```powershell
& 'D:\Programs\Anaconda\envs\pytorch\python.exe' -m unittest tests_sb3_sumo.test_contractfix_encoder tests_sb3_sumo.test_contractfix_trajectory_audit
```

## Entry, completion guard and follow-up

New process-local entry: `fast-developer/train_intersection_yield_v2_d1_contractfix.py`. The legacy entry and encoder files are unchanged. Registry/encoder/wrapper registration applies only inside the new process. Runtime provenance/source snapshots include the new entry, encoder, trajectory audit and ancestor `features.py`, `topo_temporal_features.py`, and `topo_temporal_features_v2.py`.

New launcher: `fast-developer/launch_sorted_contractfix_20261003.py`. Intended root: `runs/d1_contractfix_20261003`. It checks the two specific valid predecessor arms, their fresh/no-resume suite manifests, seed 0, raw 100000, training completion and 100-episode final evaluation with exact checkpoint path/SHA and seeds 10000–10099. A missing artifact is not treated as completion. Existing output roots and the protocol-level pair launch receipt prevent accidental duplicate launches. The new entry also checks its authorized run root and refuses resume/checkpoint initialization.

Pure file-fixture gate tests passed for both the ready case and missing/mismatched/resumed/duplicate cases. A CPU SB3 build-save-load test covers both new method aliases and verifies encoder type/SMARTS contract and equal 128-dimensional outputs, with zero environment steps. All four suites passed together: 30/30 (14 encoder + 7 rollout audit + 8 gate + 1 roundtrip). The subsequent worker-isolation change added one guard test, and the affected gate suite passed 9/9; the current verified set therefore contains **31 cases**.

The two CUDA workers use independent short roots, `runs/d1_contractfix_20261003/st` and `runs/d1_contractfix_20261003/st_rt`. Each worker's authorization environment marker matches its own command-line root. Shared original scenario traffic is read-only; scaled traffic pools and train/eval overlays are written under their respective worker root and do not intersect. The global pair receipt still locks the parent protocol. The final real `--check-only` confirmed both distinct worker commands/traffic paths, no existing pair receipt, and only the expected predecessor-not-complete blocker; it created no new run and launched no worker.

Read-only readiness command:

```powershell
& 'D:\Programs\Anaconda\envs\pytorch\python.exe' 'D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\Scene-Rep-Transformer-main\pytorch_sb3_sumo\fast-developer\launch_sorted_contractfix_20261003.py' --check-only --run-root 'D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\d1_contractfix_20261003'
```

Use the same command with `--launch` instead of `--check-only` only after readiness passes and predecessor final-result verification is complete. At the time of the pre-launch readiness check, both valid predecessors were still training and no repaired-method worker had started; the later launch and run status are appended below.

Existing hourly heartbeat `automation-3` has been updated and set ACTIVE for this sequence; it must remain quiet while unchanged, launch this pair only once after completion, update the original research records when results arrive, and pause after the requested follow-up is finished. It may not start other complete training experiments.

## 2026-10-03 09:20(+08)：本轮pair实际启动及首个只读运行快照

前序ST-RT和Longres均通过fresh seed0/100000 raw/final-100身份门后，本协议pair由supervisor PID 72064启动，唯一总根为runs/d1_contractfix_20261003。ST方法sac_mlp_d1_st_contractfix_v1在短子根st运行、PID 48788；ST-RT方法sac_mlp_d1_st_rt_contractfix_v1在st_rt运行、PID 45944。两者status为training，arguments和suite均记录seed0、fresh、no-resume、device=cuda、raw budget=100000、intersection_sorted/depart_scale=4.0；suite runtime probe为RTX 5060 Ti/torch 2.12.0+cu132。PID与runtime provenance的train记录匹配。supervisor stdout/stderr及worker stdout在此次读取时为空，diagnostic_error_count=0，trajectory audit summary errors=0，summary写入无pending/permission denial/last_error。

每个worker的runtime provenance记录11个源文件，均在20文件source archive中且SHA逐项匹配；ZIP SHA256为b86378d2230f339757c463d4bbae98259fcf075c9919189203ac931714ec9f50。两个scaled traffic roots分别位于st与st_rt；各30个traffic XML内容哈希30/30相同。train/eval overlay目录也在各自method root下，未共享临时文件夹；该快照中overlay XML尚未生成。轨迹审计summary实际写出：进度文件最近保存为5082 raw/0 updates；更晚异步summary显示两臂已采集超过5000 raw、分别1425条真实决策审计记录、15个完成episode、errors=0且unknown raw-step decisions=0。progress与summary异步落盘，计数属于不同时间点，不应强行对齐。训练阶段policy shadow/normal diagnostic也无错误，审计记录只对应真实pre-action，不包括replay/shadow forward。新pair正在进行；不得因该初期快照推断性能。

## 2026-10-03 09:32(+08)：越过warmup后的只读核验

两进程仍活跃并为training：ST PID 48788的progress为8976 raw/3976 updates；ST-RT PID 45944为8664/3664。optimization JSONL存在实际正更新记录（ST尾记录4011 updates；ST-RT尾记录3010 updates，落后于progress文件），因此更新证据成立；日志/manifest并非原子同步快照。Suite继续记录seed0/fresh/no-resume/CUDA/100000 raw。NVIDIA compute-apps一次查询列出两个worker PID，显存字段因权限显示N/A；不把N/A解释成未使用GPU。两臂behavior诊断错误均0，summary无pending/permission denial/last error；trajectory audit分别有3003与2900真实决策行，errors=0、unknown raw-step decisions=0。

独立wrapper复核已确认两臂前5个共同完成episode均逐例匹配train_monitor：decision计数为104/104/170/107/85，raw计数为310/311/508/321/255；history cache age=tick-first_seen符合实现，geometry_active在ST与ST-RT均为true。新运行已正常跨过warmup并实际更新；这不代表C8/C9性能收益，也不允许据早期统计外推final成功率。

## 2026-10-03：ST final100身份与奖励核验；ST-RT final100待完成

ST方法`st/sac_mlp_d1_st_contractfix_v1__intersection_sorted_depart4p0`已完成fresh seed0/no-resume/CUDA训练，`training_complete.json`为100000 raw steps/95001 updates、smoke=false。Final validation identity记录`final_model.zip`、100 episodes及validation seeds 10000–10099；逐episode检查确认100个seed唯一且范围完整。Checkpoint SHA256为`AF0D6674ADF321E148DB54D3D48B8349D558E40101C229FDDA9D13EFFE74B162`，training-complete与evaluation identity一致。真实episode outcome字段计数success=45、collision=50、off_route=0、timeout=5（S/C/T/O=45/50/5/0）；timeout的5次与`reward_timeout`均值−0.25相符。Shaped `environment_step_reward_v2` return mean/std=0.7935620412/10.5476265202，raw未折扣return mean/std=−0.05/0.9733961167，两个协议分列。六项shaped分量(success/collision/off-route/timeout/step cost/progress)均值4.5/−5.0/0/−0.25/−0.8644/2.4079620412，覆盖100回合，最大reconciliation error为3.55e−15。

ST eval behavior summary记录100 episodes、8644 decisions、25832 raw、diagnostic_error_count=0；trajectory audit记录8644真实pre-action决策、100 completed episodes、25832 raw、errors=0、unknown raw-step decisions=0。Shadow eval汇总为357 unique samples、episodes_with_probe_rows=100/100、active-invalid=0、error rows=0，每回合unique上限4。Train/evaluation runtime provenance均记录11个源码SHA，22/22匹配20项source archive manifest；manifest记录ZIP SHA256 `b86378d2230f339757c463d4bbae98259fcf075c9919189203ac931714ec9f50`。本轮结论限定于ST的一个训练seed，不能据此主张跨seed稳定性或分开归因联合C8/C9修复。

ST-RT训练终结记录已到100000 raw/95001 updates、status=trained；本次只读快照尚未生成final evaluation identity/results，故状态标为训练完成、final100待核验。Pair对比和ST-RT性能数值待其最终checkpoint身份、seed列表及回报审计完成后补录；此前partial eval或旧训练stdout不得替代final100结果。

## 2026-10-03：contractfix pair final100身份及奖励核验完成

后续suite快照为`complete`，两个worker exit_code均为0，更新上节final待核快照。ST-RT训练终结为fresh seed0/no-resume/CUDA、100000 raw/95001 updates、smoke=false。其evaluation identity记录final checkpoint `st_rt/sac_mlp_d1_st_rt_contractfix_v1__intersection_sorted_depart4p0/final_model.zip`、100回合及validation seeds 10000–10099；100个seed唯一且正好覆盖此区间。最终SHA256 `706548B4E84450ADCE1C224C122D88F0C0647B1F212A695A9558C4FE00A55D0F`已实算，并与training-complete/evaluation identity一致。Outcome字段计数success=52、collision=48、off_route=0、timeout=0（S/C/T/O=52/48/0/0）。Shaped `environment_step_reward_v2` return mean/std=2.0286829937/10.7445324139；raw未折扣return mean/std=0.04/0.9991996797。六项分量(success/collision/off-route/timeout/step cost/progress)均值为5.2/−4.8/0/0/−0.8773/2.5059829937，coverage=100，最大分量重构误差3.55e−15。

ST-RT final eval behavior summary为100 episodes、26218 raw、8773 decisions、errors=0；trajectory audit为100 completed episodes、8773真实pre-action decisions、26218 raw、errors=0、unknown=0；shadow汇总298 unique samples、episodes_with_probe_rows=100/100、active-invalid=0、errors=0、每回合最多4 unique。Train与evaluation runtime provenance各11项源码SHA，22/22与20项source archive manifest相符；manifest内source ZIP SHA256为`b86378d2230f339757c463d4bbae98259fcf075c9919189203ac931714ec9f50`。Suite及method final artifacts是配对状态的权威记录；progress/status较早mtime不得覆盖终结记录。

配对ST结果为S/C/T/O=45/50/5/0、shaped mean=0.7935620412、raw mean=−0.05；ST-RT为52/48/0/0、shaped mean=2.0286829937、raw mean=0.04。两路均为seed0单训练轨迹与同一100-seed验证池，结果只支持这一配对观察；两项修复C8/C9联合启用，无法从该实验分拆各自因果贡献或推断多seed稳定性。

## 2026-10-03：归因报告及独立证据审阅已完成

本协议pair的综合结果与边界见[最终归因报告](d1_contractfix_results_attribution_20261003.md)；真实决策trajectory/诊断、旧索引反事实与shadow/梯度审计见[诊断审计](d1_contractfix_diagnostic_audit_20261003.md)；same-seed final100类别转换、旧新公平性字段和配对限制见[对照证据](d1_contractfix_comparison_evidence_20261003.md)。核心身份、SHA、reward v2/raw分列、六分量和转换矩阵已交叉核对。ST→ST-RT对比同时打开route路径、改变有效计算与critic梯度路径；名义registered parameter count相同不证明有效容量相等或query单独因果。n-step单γ相对标准γ^k为确认的实现协议差异，不定性为已证实的upstream bug。若未来采用新RL协议，SAC+MLP、MST+SLT等主比较也须按同一协议对齐；旧历史结果不视作已对齐。本轮没有追加训练或仿真。automation-3已暂停本轮自动跟进；未追加训练或仿真。
