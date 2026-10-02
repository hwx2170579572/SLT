-- DuckDB-compatible audit of the two seed0 cross training monitors.
-- The first line in each monitor is SB3 metadata and is skipped.

CREATE OR REPLACE TEMP VIEW ppo_cross_monitor AS
SELECT 'PPO' AS algorithm, *
FROM read_csv_auto(
  'results_sb3_sumo_paper/paper_baselines_sac_ppo_v14_frozen/paper__ppo__cross__seed0/train_monitor.csv',
  header = true,
  skip = 1
);

CREATE OR REPLACE TEMP VIEW sac_cross_monitor AS
SELECT 'RLEncoder(GRU)-SAC' AS algorithm, *
FROM read_csv_auto(
  'results_sb3_sumo_paper/paper_baselines_sac_ppo_v14_frozen/paper__sac__cross__seed0/train_monitor.csv',
  header = true,
  skip = 1
);

SELECT algorithm,
       count(*) AS completed_episodes,
       sum(CASE WHEN is_success THEN 1 ELSE 0 END) AS successful_episodes,
       sum(CASE WHEN collision THEN 1 ELSE 0 END) AS collision_events,
       sum(CASE WHEN max_time THEN 1 ELSE 0 END) AS timeout_events
FROM (
  SELECT * FROM ppo_cross_monitor
  UNION ALL
  SELECT * FROM sac_cross_monitor
)
GROUP BY algorithm
ORDER BY algorithm;
