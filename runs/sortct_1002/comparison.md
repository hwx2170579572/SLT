# Run comparison

Run root: `D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\sortct_1002`

Outcome rates use completed episode records only; denominator is shown as eval n. Training truncation is reported separately as incomplete episodes. Continuous behavior metrics pool valid raw ticks or action scalar samples rather than averaging episodes equally.

| Method | Train raw steps | Train incomplete episodes | Eval n | Success | Collision | Timeout | Off-route | Mean speed | Stopped | Near-unit-bound | Low TTC | Covered critical unobserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sac_mlp_d1_st_rt_routeact_v1 | 100000 | 1 | 100 | 47.0% | 53.0% | 0.0% | 0.0% | 3.734 m/s | 16.0% | 22.3% | 8.2% | 4.3% |
| sac_mlp_d1_st_rt_conflicttime_v1 | 100000 | 1 | 100 | 40.0% | 57.0% | 3.0% | 0.0% | 3.576 m/s | 31.8% | 40.6% | 8.4% | 4.4% |

`near_unit_bound_fraction` is the share of action scalar samples with |a| >= 0.95; action-space bounds and definition are in comparison.json. Low-TTC uses TTC < 3 s among geometry-valid risk-evaluable ticks. Covered critical-unobserved uses ticks with both valid risk and policy-observation coverage. Null means no valid denominator or missing source artifact, not zero.

Full denominators, coverage, train/eval summaries, optimization statistics, and artifact paths are in `comparison.json`.
