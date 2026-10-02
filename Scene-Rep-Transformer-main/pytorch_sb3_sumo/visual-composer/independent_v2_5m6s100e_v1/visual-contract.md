# Independent-v2 5-method × 6-scenario training-curve contract

## Artifact

Two publication-ready small-multiple figures generated from the complete
independent-v2 extension: one figure for average reward and one for success
rate. Each figure has six scene panels and five method curves.

## Scientific question

How do the five methods learn under each high-traffic scene when the horizontal
axis is the scene-specific cumulative raw simulation-step count?

## Source evidence

The 30 completed `train_monitor.csv` logs referenced by
`results_iv2_5m6s100e_v1/comparison/summary/summary.json`. The 12 parent-v2
cells are read from their original directories and the 18 fresh cells are read
from the independent-v2 extension directories. No model is retrained and no
evaluation result is regenerated.

## Transform

- Episode order is the order in each monitor log.
- Each episode contributes its observed `raw_simulation_steps` to the
  scene/method-specific cumulative x-axis.
- Reward signal: trailing mean of the most recent 20 episode returns; before
  20 episodes are available, divide by the number of observed episodes.
- Success signal: mean of exactly 20 binary `is_success` slots; unavailable
  prefix slots are explicit zeros, so episode 1 contributes either `0` or
  `1/20`.
- Each episode signal is held over that episode's observed raw steps as a
  descriptive retrospective proxy, then smoothed at every raw step by
  `EMA_t = 0.999*EMA_(t-1) + 0.001*signal_t`, initialized at zero.
- Curves are sampled every 100 raw steps only for rendering/data size; the EMA
  recurrence itself is evaluated at every raw step.

## Visual contract

- Six panels: Left-turn, Cross, Roundabout-A, Roundabout-B, Roundabout-C, and
  CARLA. Each panel uses its own x-limit based on the largest observed total
  raw-step count among methods in that scene; no cross-scene resampling or
  macro-average curve is used.
- Reward y-axis is the logged return scale; success y-axis is [0, 1].
- Methods use an Okabe-Ito-compatible categorical palette plus distinct line
  styles so the figure remains interpretable in grayscale.
- All five methods are retained with equal visual status; no focal method is
  highlighted and no interval is cropped to imply an ordering.

## Uncertainty and limitations

The plotted lines are deterministic descriptive training signals from one
training seed per method×scene. No confidence interval or cross-seed
significance claim is made. A missing success completion time is unrelated to
these training curves and is not imputed here.

## Traceability

The manifest records the protocol and summary hashes, every monitor-log hash,
the exact transform parameters, row counts, source type (fresh or adopted),
and QA checks for monotone time steps, finite values, and the 20-slot prefix
zero padding.
