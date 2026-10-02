-- DuckDB-compatible source queries for the cross/Roundabout-A/C diagnosis.
-- The CSV snapshots are generated from the JSON evaluation records by
-- tools/build_cross_roundabout_diagnosis_report.py.

CREATE OR REPLACE TEMP VIEW cross_diagnosis AS
SELECT *
FROM read_csv_auto(
  'pytorch_sb3_sumo/reports/cross_roundabout_anomaly_diagnosis_20260805/cross_diagnosis.csv',
  header = true
);

CREATE OR REPLACE TEMP VIEW roundabout_comparison AS
SELECT *
FROM read_csv_auto(
  'pytorch_sb3_sumo/reports/cross_roundabout_anomaly_diagnosis_20260805/ppo_roundabout_comparison.csv',
  header = true
);

CREATE OR REPLACE TEMP VIEW cross_speed_trace AS
SELECT *
FROM read_csv_auto(
  'pytorch_sb3_sumo/reports/cross_roundabout_anomaly_diagnosis_20260805/cross_speed_trace.csv',
  header = true
);

SELECT * FROM cross_diagnosis ORDER BY algorithm;
SELECT * FROM roundabout_comparison ORDER BY scenario, protocol;
SELECT * FROM cross_speed_trace ORDER BY algorithm, raw_steps;
