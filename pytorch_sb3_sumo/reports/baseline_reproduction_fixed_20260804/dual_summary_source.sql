-- DuckDB-compatible source queries for the portable reproduction report.
-- Missing seed runs remain absent from metrics and visible in quality.csv.

CREATE OR REPLACE TEMP VIEW validated_seed_results AS
SELECT *
FROM read_csv_auto(
  'results_sb3_sumo_paper/paper_baselines_sac_ppo_v14_frozen/summary_dual_protocol/runs_by_seed.csv',
  header = true
);

CREATE OR REPLACE TEMP VIEW aggregate_seed_results AS
SELECT *
FROM read_csv_auto(
  'results_sb3_sumo_paper/paper_baselines_sac_ppo_v14_frozen/summary_dual_protocol/aggregate_mean_std.csv',
  header = true
);

CREATE OR REPLACE TEMP VIEW evaluation_quality AS
SELECT *
FROM read_csv_auto(
  'results_sb3_sumo_paper/paper_baselines_sac_ppo_v14_frozen/summary_dual_protocol/quality.csv',
  header = true
);

SELECT *
FROM validated_seed_results
ORDER BY traffic_protocol, method, scenario, seed;

SELECT *
FROM aggregate_seed_results
ORDER BY traffic_protocol, method, scenario;

SELECT status, count(*) AS protocol_slots
FROM evaluation_quality
GROUP BY status
ORDER BY status;
