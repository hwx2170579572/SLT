-- Scenario facts parsed from released ego routes, traffic XMLs and source
-- scenario generators. Values are regenerated into scenario_contract.csv by
-- tools/build_cross_roundabout_diagnosis_report.py.

SELECT *
FROM read_csv_auto(
  'pytorch_sb3_sumo/reports/cross_roundabout_anomaly_diagnosis_20260805/scenario_contract.csv',
  header = true
)
ORDER BY scenario;
