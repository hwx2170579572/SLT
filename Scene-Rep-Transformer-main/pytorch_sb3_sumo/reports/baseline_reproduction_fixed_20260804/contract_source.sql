-- DuckDB-compatible reproduction of the hard acceptance ledger shown in the report.

SELECT *
FROM (VALUES
  (1, '最终环境工厂', '周期评估与最终评估均由注入的 env_factory(evaluation=true) 构造', '统一工厂调用 + 回归测试'),
  (2, '环境类', 'SAC=PaperSumoSceneEnv；SMARTS PPO=PaperPpoRgbEnv', 'expected_environment_class 精确匹配'),
  (3, 'observation/action space', 'SB3 模型空间与环境空间对象及签名完全相同', '推理前 fail-fast + 签名持久化'),
  (4, '交通协议', 'frozen=evaluation 分区；source-all=all 分区', 'provenance.traffic_partition 精确匹配'),
  (5, '发布资产', 'uses_released_assets=true，且每回合 traffic_variant 非空', '逐回合记录与集合重算'),
  (6, '确定性 seed', '评估 seed=训练 seed+10000，并连续覆盖 50 回合', '逐回合 seed 序列精确核对'),
  (7, '成功率', 'success_rate=成功回合数/50，且由逐回合记录重算一致', '绝对误差≤1e-12'),
  (8, '不重训重评', '模型/检查点哈希通过训练完整性审计，收据声明未恢复训练', 'SHA-256 + checkpoint clock/CRC 审计')
) AS t("order", "check", acceptance, enforcement)
ORDER BY "order";
