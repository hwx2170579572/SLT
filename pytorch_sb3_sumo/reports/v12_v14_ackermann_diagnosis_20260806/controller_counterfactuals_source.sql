-- Same-checkpoint deterministic evaluation; only ego_control_profile changes.
SELECT *
FROM (VALUES
  ('left_turn', 'v12 Proposed seed0', 10, 1.00, 1.00, 11.250, 14.130),
  ('left_turn', 'v14 SAC seed0',      10, 0.90, 1.00, 14.744, 17.970),
  ('left_turn', 'v14 SAC seed1',      10, 1.00, 1.00, 32.940, 35.200),
  ('left_turn', 'v14 SAC seed3',      10, 1.00, 0.80, 11.940, 13.675),
  ('cross',     'v12 Proposed seed0',  3, 1.00, 1.00, 40.633, 45.167),
  ('cross',     'v14 SAC seed2',      10, 0.30, 0.90, 27.733, 29.878)
) AS t(scenario, checkpoint, episodes, direct_success, proxy_success,
       direct_success_time_seconds, proxy_success_time_seconds);
