# Training-curves figure contract v3 — time-step axis and EMA smoothing

## Artifact and claim

- Artifact: reference-style three-panel success-rate curves for CARLA, Cross, and Roundabout, with one individual export per scenario.
- The figure compares the three saved parent-v2 models (MST+SLT, TemporalGraph 时序车辆图基线, and v4.8 Tie-only Replicated Calibration) without retraining.
- Supported claim: the logged training success dynamics over simulation time steps, after a transparent, fixed smoothing transformation.

## Time-step definition and source

- Source remains each run's real `train_monitor.csv` and the protocol-bound saved model.
- One episode's `is_success` outcome is held over its logged `raw_simulation_steps`; concatenating those episode-length segments yields a descriptive per-time-step success signal. This is a proxy because the source log stores success at episode resolution, not at every internal simulator step.
- The x-axis is cumulative `raw_simulation_steps`, displayed as actual time-step counts (0–about 50,000), not episode index or wall-clock seconds.

## Window and smoothing

- Rolling window span: 2,000 raw simulation time steps, chosen to cover roughly 10–20 episodes across these runs while remaining responsive to training changes.
- A rolling success mean and a descriptive rolling band are computed on the per-time-step proxy. EMA smoothing is then applied per raw time step with α=0.999; curves are sampled every 100 time steps only for rendering.
- The band is the EMA-smoothed mean ± one descriptive rolling standard error (population SD divided by the square root of the number of overlapping episodes), clipped to [0,1]. It is not a cross-seed confidence interval.

## Figure map and accessibility

- Composite layout: one row × three scenario panels labelled (a), (b), (c), following the supplied reference style; individual scenario panels are also exported.
- Y-axis is `Avg Success rate` on [0,1]. X-axis is `Time steps` with shared 10,000-step ticks.
- Color-blind-safe Okabe–Ito colors and distinct line styles are retained. Full protocol method labels remain in the manifest; compact legend labels keep the plot readable.

## QA and limitations

- Verify all nine source model/log hashes and non-empty monitor logs before plotting.
- Render and inspect composite plus all individual panels for clipping, band visibility, and label readability.
- The per-time-step expansion and EMA are descriptive visualization transformations; they do not create new evaluation episodes or alter the saved models.
