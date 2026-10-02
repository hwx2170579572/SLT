# Training-curves figure contract v2 — mean line with in-log variability band

## Artifact and claim

- Artifact: a reference-style small-multiple figure with one panel per scenario (CARLA, Cross, Roundabout), plus one-panel exports for each scenario.
- Each panel compares the three saved parent-v2 models: MST+SLT, TemporalGraph 时序车辆图基线, and v4.8 Tie-only Replicated Calibration.
- Supported claim: the figures describe how the recorded training success rate evolves over raw simulation steps for the nine saved models.
- The shaded region is a within-training-log rolling standard-error band, not a cross-seed confidence interval; there is one training seed (0) per method×scenario.

## Source and evidence

- Protocol: `experiments/independent_v2_existing_3methods_100seeds_100episodes_v1/protocol.json`.
- Source root: `results_hd_ss100_v2/comparison/runs/`.
- Source metric: per-episode `is_success` from each run's `train_monitor.csv`.
- X-axis: cumulative sum of each episode's logged `raw_simulation_steps`; no interpolation across runs.
- Source model paths, hashes, row counts, and log hashes are retained in `training_curve_manifest.json`.

## Statistics / uncertainty

- Mean line: 20-episode trailing mean of the binary success indicator.
- Band: mean ± one 20-episode trailing population standard error (SD/√n), clipped to [0, 1] because success is a proportion. This is a descriptive smoothing band, not a formal confidence interval.
- The first 19 episodes use the available prefix window. No missing values are imputed.
- The band must be labelled as “descriptive rolling SE (SD/√n) within seed 0” wherever it is shown or captioned.

## Figure map and style

- Composite: one row × three columns, ordered CARLA, Cross, Roundabout; panel labels (a), (b), (c) use a small yellow highlight similar to the supplied visual reference.
- Individual exports: one full-size panel per scenario for paper placement or inspection.
- Y-axis: Avg Success rate, 0–1; X-axis: raw simulation steps (×10³), with shared visual grammar across panels.
- Curves use stable, color-blind-safe Okabe–Ito colors with distinct line styles. The proposed v4.8 method keeps one stable blue accent; baselines use orange and green.
- Use a compact legend inside the upper-left of each panel, short method labels, and a footer note carrying the full metric/band definition and provenance.

## QA and limitations

- Confirm all nine monitor logs and saved models exist and hash correctly before plotting.
- Render and inspect the composite and every individual panel for clipping, overlap, font coverage, color/grayscale legibility, and visible band boundaries.
- Four of nine parent runs do not retain TensorBoard event files; this v2 style therefore uses the monitor logs consistently for all panels.
- These curves are training-dynamics evidence only and must not be read as the 100 logical test-seed evaluation uncertainty reported elsewhere.
