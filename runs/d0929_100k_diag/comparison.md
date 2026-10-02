# Run comparison

Run root: `D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\d0929_100k_diag`

Outcome rates use completed episode records only; denominator is shown as eval n. Training truncation is reported separately as incomplete episodes. Continuous behavior metrics pool valid raw ticks or action scalar samples rather than averaging episodes equally.

| Method | Train raw steps | Train incomplete episodes | Eval n | Success | Collision | Timeout | Off-route | Mean speed | Stopped | Near-unit-bound | Low TTC | Covered critical unobserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sac_mlp | 100000 | 1 | 100 | 22.0% | 38.0% | 40.0% | 0.0% | 2.648 m/s | 49.5% | 18.1% | 5.1% | 2.1% |
| sac_mlp_d1_st_rt | 100000 | 1 | 100 | 63.0% | 37.0% | 0.0% | 0.0% | 5.048 m/s | 10.4% | 39.5% | 9.3% | 5.0% |

`near_unit_bound_fraction` is the share of action scalar samples with |a| >= 0.95; action-space bounds and definition are in comparison.json. Low-TTC uses TTC < 3 s among geometry-valid risk-evaluable ticks. Covered critical-unobserved uses ticks with both valid risk and policy-observation coverage. Null means no valid denominator or missing source artifact, not zero.

Full denominators, coverage, train/eval summaries, optimization statistics, and artifact paths are in `comparison.json`.
