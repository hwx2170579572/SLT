-- Deterministic Double Merging policy traces under smarts_ackermann_proxy.
SELECT *
FROM (VALUES
  ('v14 PPO cross seed0', 3, 0.046, 0.046, 0.046, 'timeout'),
  ('v14 SAC cross seed0', 3, 0.121, 0.121, 0.121, 'timeout'),
  ('v14 SAC cross seed2', 3, 9.706, 9.010, 9.000, 'success')
) AS t(checkpoint, episodes, requested_speed_mps, effective_speed_mps,
       actual_speed_mps, terminal_group);
