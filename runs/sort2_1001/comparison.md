# Run comparison

Run root: `D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\sort2_1001`

Outcome rates use completed episode records only; denominator is shown as eval n. Training truncation is reported separately as incomplete episodes. Continuous behavior metrics pool valid raw ticks or action scalar samples rather than averaging episodes equally.

| Method | Train raw steps | Train incomplete episodes | Eval n | Success | Collision | Timeout | Off-route | Mean speed | Stopped | Near-unit-bound | Low TTC | Covered critical unobserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sac_mlp_d1_st_rt_topo_routeaware_v1 | 100000 | 1 | 100 | 43.0% | 32.0% | 25.0% | 0.0% | 3.449 m/s | 39.9% | 19.1% | 6.1% | 3.2% |
| sac_mlp_d1_st_rt_3slot | 100000 | 1 | 100 | 34.0% | 49.0% | 17.0% | 0.0% | 2.990 m/s | 49.0% | 16.1% | 6.0% | 3.1% |

`near_unit_bound_fraction` is the share of action scalar samples with |a| >= 0.95; action-space bounds and definition are in comparison.json. Low-TTC uses TTC < 3 s among geometry-valid risk-evaluable ticks. Covered critical-unobserved uses ticks with both valid risk and policy-observation coverage. Null means no valid denominator or missing source artifact, not zero.

Full denominators, coverage, train/eval summaries, optimization statistics, and artifact paths are in `comparison.json`.
