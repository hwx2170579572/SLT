# Run comparison

Run root: `D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\dm7_1001_retry02`

Outcome rates use completed episode records only; denominator is shown as eval n. Training truncation is reported separately as incomplete episodes. Continuous behavior metrics pool valid raw ticks or action scalar samples rather than averaging episodes equally.

| Method | Train raw steps | Train incomplete episodes | Eval n | Success | Collision | Timeout | Off-route | Mean speed | Stopped | Near-unit-bound | Low TTC | Covered critical unobserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mst_slt | None | 2 | 100 | 16.0% | 53.0% | 31.0% | 0.0% | 2.564 m/s | 58.3% | 13.1% | 5.4% | 2.3% |
| sac_mlp_d1_st | 100000 | 1 | 100 | 16.0% | 83.0% | 1.0% | 0.0% | 6.304 m/s | 14.4% | 18.3% | 12.7% | 4.1% |
| sac_mlp_d1_st_rt | 100000 | 1 | 100 | 14.0% | 66.0% | 20.0% | 0.0% | 3.432 m/s | 35.9% | 7.5% | 6.7% | 2.6% |
| sac_mlp_d1_st_rt_topo | 100000 | 1 | 100 | 31.0% | 69.0% | 0.0% | 0.0% | 4.093 m/s | 1.4% | 18.8% | 8.1% | 3.1% |
| sac_mlp_d1_st_rt_topo_routeaware_v1 | 100000 | 1 | 100 | 31.0% | 45.0% | 24.0% | 0.0% | 2.962 m/s | 51.3% | 15.4% | 4.7% | 1.7% |
| sac_mlp_d1_st_rt_3slot | 100000 | 1 | 100 | 30.0% | 70.0% | 0.0% | 0.0% | 6.375 m/s | 3.6% | 29.1% | 14.3% | 7.6% |
| sac_mlp_d1_st_rt_topo_3slot | 100000 | 1 | 100 | 20.0% | 47.0% | 33.0% | 0.0% | 3.157 m/s | 55.2% | 5.8% | 6.2% | 2.6% |

`near_unit_bound_fraction` is the share of action scalar samples with |a| >= 0.95; action-space bounds and definition are in comparison.json. Low-TTC uses TTC < 3 s among geometry-valid risk-evaluable ticks. Covered critical-unobserved uses ticks with both valid risk and policy-observation coverage. Null means no valid denominator or missing source artifact, not zero.

Full denominators, coverage, train/eval summaries, optimization statistics, and artifact paths are in `comparison.json`.
