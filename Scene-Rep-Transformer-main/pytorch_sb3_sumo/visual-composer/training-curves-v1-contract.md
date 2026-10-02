# Training-curves figure contract v1

## Artifact and claim

- Artifact: three scenario-specific training-curve figures (CARLA, Cross, Roundabout), each comparing the three methods MST+SLT, TemporalGraph 时序车辆图基线, and v4.8 Tie-only Replicated Calibration. A 3×3 overview is also produced for quick inspection.
- Claim supported: the figures faithfully show the logged training dynamics of the nine saved parent-v2 models. They are descriptive training evidence, not a new evaluation or a claim of statistical superiority.
- Reviewer question: within each scenario, how do the methods' episodic return and success rate evolve over the recorded training budget, and where do trajectories stabilize or remain noisy?

## Scope and provenance

- Source protocol: `experiments/independent_v2_existing_3methods_100seeds_100episodes_v1/protocol.json`.
- Source result root: `results_hd_ss100_v2/comparison/runs/`.
- One parent-v2 run per method×scenario, all with training seed 0; the current independent-v2 evaluation did not retrain these models.
- Primary log: each run's `train_monitor.csv`, which records episode return (`r`), episode length (`l`), elapsed time (`t`), raw simulation steps, success, collision, off-route, and max-time flags. Cumulative raw simulation steps are used on the x-axis.
- The model path and SHA-256, log path and SHA-256, row counts, and protocol hash are recorded in `training_curve_manifest.json`.

## Metrics and transformations

- Panel A: episodic return (`r`). A thin raw trace is overlaid with a 20-episode trailing mean to expose the trend without hiding variability.
- Panel B: success rate. A 20-episode trailing mean of `is_success` is plotted as a proportion (0–1); no values are imputed.
- The first 19 episodes use the available prefix for the trailing mean. X values are cumulative sums of the logged per-episode `raw_simulation_steps`; no cross-run interpolation is used.
- Collision, off-route, and max-time flags remain available in the exported long-format data for audit, but are not silently conflated with success.

## Visual design and accessibility

- Use a color-blind-safe Okabe–Ito palette plus distinct line styles/markers. v4.8 keeps a stable accent color across scenarios; the two baselines use blue/gray tones.
- Keep method ordering and labels identical in all panels, share the scenario-specific x-axis, use direct legends, and retain units in axis labels.
- Export high-resolution PNG and vector SVG; set SVG text to remain editable. Include a short provenance note in each figure.

## QA and limitations

- Verify all nine source runs resolve to the protocol's recorded deployment model (`final_model.zip` for MST+SLT/TemporalGraph, `selected_model.zip` for v4.8), and that the files and hashes match.
- Verify every source has a non-empty monitor log and no malformed numeric/boolean rows; record counts and ranges in the manifest.
- Render and visually inspect all scenario figures and the overview for clipping, unreadable legends, inconsistent scales, or accidental smoothing artifacts.
- Training curves are single-seed parent-v2 logs (seed 0), so they should not be interpreted as uncertainty bands or a replacement for the 100-seed×100-episode evaluation matrix.
