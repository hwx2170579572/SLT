# 分阶段实现路线：只读核查

日期：2026-10-03。核查对象为 `implementation_roadmap.md`，并对照现有 SAC 实现地图、环境证据、技术路线及研究上下文。未改训练源码、未运行测试、训练、评估或 SUMO。以下记录路线中的接口/状态问题，不表示审阅通过或方法已经验证。

## 需要澄清的问题

1. **`z_t` 维数冲突。** 初始草稿第11行总接口写 `z_t ∈ R^256`；M0规定固定128维，M1/M2与M0同维，M3也要求沿用相同 z 维度。当前实现地图记录的常用 D1 encoder 输出为128维（[`full_mst_slt_implementation.md`](../full_mst_slt_implementation.md#L73-L80)）。问题已反馈root；当前路线第11行已统一为128维，并说明256仅是后续容量选项。状态：**设计文本已修正**，实际policy/critic features_dim、target copy和参数/公平性尚未实现核验。

2. **M2 的 TD/aux 梯度边界还应指名张量。** 初始版本只笼统说停止 critic 对预测概率/事件分布的直接梯度。问题已反馈root；当前第145–159行已列明：`Ξ/ρ/π`在SAC支路stop-gradient；`EventTokenMLP/EventGraph/Readout`重编码这些detached数值并接收TD；历史/地图共享stem接TD与aux的一次合并更新；预测头只接aux；不得让预测hidden/shared query绕过边界直接进入z；对应target参数只同步、无反传。状态：**设计文本已明确**。仍需在实现中核对实际autograd图、optimizer唯一所有权、target同步和梯度流实测。

## 已核对一致的实施边界

- M0→M1→M2→M3是逐级依赖路线；SAC从M0接入。M0先建立同新输入下的通用双图闭环对照；M1引入确定性CV/事件结构；M2才引入progress transition与删失事实监督；M3再检验共享scene latent。M3使用共享scene queries，指数级 joint beam 明确非首版前置条件。若M2数据不足或M3无联合信息价值，应停在简单版本。
- SAC接口仍为连续2维动作、`Q_1(z,a),Q_2(z,a)`；actor/online critic共享 extractor，但当前 `DetachedSceneActor` 将特征detach，actor optimizer不更新encoder；critic TD更新online encoder，target critic有独立encoder并Polyak同步。新路线已给aux event loss与SAC TD的逐参数张量合同；需要实现与梯度流核验，不能当成已有代码行为。
- 时间单位吻合：SUMO raw tick为0.1s，`action_repeat=3`约0.3s/决策；路线的0.3s future grid可按决策序列实现。SAC n-step=4是四个policy decisions，不是4 raw ticks。阶段0建议统一处理 `gamma^k` 与60秒 timeout语义；这是公共SAC协议变化，所有M0及后续控制须同协议，不能把与历史结果差异归因于encoder。
- 数据防泄漏计划写明：新结构化当前/历史 observation 给在线encoder；未来观察只进入延迟生成的aux label；ID只用于episode内track/标签join；不用future social route、未来发车表、跨episode reset、训练中缓存的z。按episode/traffic划分以免相邻帧泄漏。实际collector仍需验证写入与replay配对身份、未来6s标签是否成熟、censor与out-of-set处理。
- 阶段1/2可静态开发并行；进入M0前共同schema及静态zone合同须完成。现有数据没有完整新输入/replay与事件时间序列，旧汇总不能训练新aux头。roadmap各阶段均为计划；有限smoke标为“未执行”，当前没有新增实现/训练/仿真/评估。

## 记录状态

两份研究记录已追加路线索引，保留旧结果并说明方法尚未实施。此核查不是implementation review通过、性能/新颖性结论，也未运行 roadmap 中任何 gate。
