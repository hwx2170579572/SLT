# Training-curves figure contract v4 — focal-method comparison

## Artifact and claim

- Artifact: a two-row, three-scenario dashboard based on the existing v3 time-step curves.
- Top row: all three method curves, with v4.8 visually emphasized and both baselines retained as full-length controls.
- Bottom row: the direct margin `v4.8 − max(MST+SLT, TemporalGraph)` with a zero reference line. Positive regions mean v4.8 is above both baselines at that time step; negative regions remain visible.
- The figure is a diagnostic comparison, not a guarantee that v4.8 dominates every scene or training phase.

## Source data and transformations

- Source: `outputs/iv2e3m100x100_training_curves_v3/training_success_curve_time_steps_ema999_data.csv`, which is derived from the unchanged parent-v2 `train_monitor.csv` logs.
- The same cumulative raw simulation time-step axis, 2,000-step rolling aggregation, and per-time-step EMA α=0.999 are reused; no original row or metric is overwritten.
- The margin is computed pointwise on the aligned EMA curves. It is not a selected interval or a rescaled subset.

## Statistics and uncertainty

- Shaded bands in the top row are the existing descriptive EMA-smoothed rolling SE bands within training seed 0.
- The bottom row shows no fabricated uncertainty; a light horizontal zero line separates positive and negative margins.
- Any summary statistics are descriptive (AUC/terminal margin) and are written to the manifest for audit.

## Visual grammar

- Three columns: CARLA, Cross, Roundabout; two rows: absolute success curves and v4.8-vs-best-baseline margin.
- v4.8 uses a thick blue line and light blue band; baselines use neutral gray/orange thin lines so their trajectories remain visible without competing with the focal method.
- Positive margin is lightly shaded blue; negative margin is lightly shaded gray. Line styles carry method identity in grayscale.
- Panel labels, shared time-step ticks, and zero reference are explicit. No x-range is cropped to hide unfavorable regions.

## QA and limitations

- Verify that all three methods are present at every scenario and that pointwise alignment uses identical time-step samples.
- Inspect composite and individual scenario exports for clipping, legend visibility, zero-line visibility, and grayscale readability.
- The source contains one training seed per method×scenario; this figure cannot establish cross-seed statistical significance or universal dominance.
