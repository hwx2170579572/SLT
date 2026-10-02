# Run comparison

Run root: `D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\t0930_topo3_retry01`

Outcome rates use completed episode records only; denominator is shown as eval n. Training truncation is reported separately as incomplete episodes. Continuous behavior metrics pool valid raw ticks or action scalar samples rather than averaging episodes equally.

| Method | Train raw steps | Train incomplete episodes | Eval n | Success | Collision | Timeout | Off-route | Mean speed | Stopped | Near-unit-bound | Low TTC | Covered critical unobserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sac_mlp_d1_st_rt_topo | 100000 | 1 | 100 | 39.0% | 39.0% | 22.0% | 0.0% | 4.515 m/s | 46.7% | 22.3% | 7.9% | 3.2% |
| sac_mlp_d1_st_rt_topo_3slot | 100000 | 1 | 100 | 41.0% | 56.0% | 3.0% | 0.0% | 4.626 m/s | 20.0% | 37.1% | 11.3% | 4.9% |

`near_unit_bound_fraction` is the share of action scalar samples with |a| >= 0.95; action-space bounds and definition are in comparison.json. Low-TTC uses TTC < 3 s among geometry-valid risk-evaluable ticks. Covered critical-unobserved uses ticks with both valid risk and policy-observation coverage. Null means no valid denominator or missing source artifact, not zero.

Full denominators, coverage, train/eval summaries, optimization statistics, and artifact paths are in `comparison.json`.
