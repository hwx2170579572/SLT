# Training-curves figure contract v5 — episode-window / time-step EMA

## Artifact and claim

- Artifact: a two-row, three-scenario dashboard for the three saved parent-v2 models.
- Top row: the three method success trajectories, with v4.8 visually emphasized while MST+SLT and TemporalGraph remain full-length controls.
- Bottom row: the signed pointwise margin `v4.8 − max(MST+SLT, TemporalGraph)`. Positive and negative regions are both retained, so the figure shows where the requested ordering is and is not supported.
- The figure is a descriptive training-dynamics comparison; it does not establish universal dominance or cross-seed significance.

## Source data and transformations

- Source: the unchanged parent-v2 `train_monitor.csv` logs referenced by `experiments/independent_v2_existing_3methods_100seeds_100episodes_v1/protocol.json`.
- For each method×scenario, the success labels are ordered by logged episode. At episode `i`, the statistic is the mean of exactly 20 binary slots containing the most recent outcomes; unavailable earlier slots are explicitly filled with zero. Thus the first success is `1/20`, not `1/1`.
- The episode-level value is held over that episode's logged `raw_simulation_steps` as a descriptive retrospective time-step proxy (the episode outcome is only known at episode end). Consequently the horizontal axis remains cumulative raw simulation time steps, while CARLA, Cross, and Roundabout retain their own observed episode-length distributions and therefore their own 20-episode time spans.
- EMA smoothing is applied once per raw simulation time step with `alpha=0.999`, initialized at `(time_step=0, value=0)`. Curves are sampled every 100 steps only for compact rendering; the EMA is not subsampled.
- The source logs and model files are read-only inputs. No original row or metric is overwritten.

## Statistics and uncertainty

- Shaded top-row bands are descriptive within-seed bands: the population standard deviation of the same 20 binary slots (including zero padding), divided by `sqrt(20)`, then EMA-smoothed per time step. They are not cross-seed confidence intervals.
- The bottom row has no invented uncertainty; its zero line and signed fill expose both favorable and unfavorable intervals.
- AUC and margin summaries in the manifest are descriptive summaries over the common time-step support only.

## Visual grammar

- Three columns: CARLA, Cross, Roundabout. Two rows: absolute success and v4.8 margin versus the best baseline.
- v4.8 uses a thick Okabe–Ito blue line and light blue band. MST+SLT is neutral gray dashed; TemporalGraph is orange dotted. Line style preserves identity in grayscale.
- Positive margin is lightly blue-filled and negative margin lightly gray-filled. The x-axis is labeled `Time steps` and uses a common 0–50,000 frame.
- Panel labels, legends, and the EMA/window note are explicit. No segment is cropped to manufacture a gain.

## QA and limitations

- Validate all nine source logs and model hashes through the existing v1 source resolver; validate monotone raw-step time and finite transformed values.
- Inspect composite and per-scenario PNG/SVG exports for clipping, overlap, band/zero-line visibility, and grayscale readability.
- Only one training seed is available for each method×scenario. The plot cannot support a cross-seed statistical claim.
