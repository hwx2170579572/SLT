# Iteration log

## v1

- Input: complete independent-v2 5-method × 6-scenario matrix.
- Change: added scene-specific cumulative raw-step curves for reward and
  success, with a 20-episode fixed-denominator success window, prefix zero
  padding, and EMA alpha 0.999.
- Outputs: composite and per-scene PNG/SVG figures, episode-level and sampled
  curve CSVs, and a hash-bound manifest.
