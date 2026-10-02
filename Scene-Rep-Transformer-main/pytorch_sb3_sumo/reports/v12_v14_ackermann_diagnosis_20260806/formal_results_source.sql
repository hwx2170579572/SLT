-- DuckDB-compatible compact source for the report's audited formal aggregates.
-- Percentages and successful-episode means were recomputed from the run-level
-- paper_evaluation_detailed.json files listed in artifact.json.
SELECT *
FROM (VALUES
  ('left_turn', 'v12 Proposed', 0.980, 0.016, 0.004, 11.866),
  ('left_turn', 'v14 SAC',      0.920, 0.068, 0.012, 19.443),
  ('left_turn', 'v14 PPO',      0.528, 0.468, 0.004, 24.460),
  ('cross',     'v12 Proposed', 0.984, 0.016, 0.000, 45.757),
  ('cross',     'v14 SAC',      0.152, 0.108, 0.740, 29.808),
  ('cross',     'v14 PPO',      0.000, 0.176, 0.824, NULL)
) AS t(scenario, result, success_rate, collision_rate, timeout_rate,
       mean_success_completion_time_seconds);
