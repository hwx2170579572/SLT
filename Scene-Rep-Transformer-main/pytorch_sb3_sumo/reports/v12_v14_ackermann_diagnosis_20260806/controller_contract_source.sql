-- Protocol facts audited from protocol.json, protocol_baselines_v14.json and
-- envs/sumo/sumo_env.py.
SELECT *
FROM (VALUES
  ('v12', 'direct',                  'source_all',   'source', 600),
  ('v14', 'smarts_ackermann_proxy',  'frozen_80_20','paper',  400)
) AS t(version, ego_control_profile, traffic_protocol,
       episode_limit_profile, cross_max_raw_steps);
