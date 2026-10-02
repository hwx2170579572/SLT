# Run comparison

Run root: `D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\sortg3sm_1002`

Outcome rates use completed episode records only; denominator is shown as eval n. Training truncation is reported separately as incomplete episodes. Continuous behavior metrics pool valid raw ticks or action scalar samples rather than averaging episodes equally.

| Method | Train raw steps | Train incomplete episodes | Eval n | Success | Collision | Timeout | Off-route | Mean speed | Stopped | Near-unit-bound | Low TTC | Covered critical unobserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sac_mlp_d1_st_rt_topo_goalonly_v1 | 300 | 1 | 1 | 0.0% | 100.0% | 0.0% | 0.0% | 4.015 m/s | 0.0% | 0.0% | 17.1% | 1.6% |
| sac_mlp_d1_st_rt_3slot_nonlinear_v1 | 300 | 1 | 1 | 0.0% | 100.0% | 0.0% | 0.0% | 5.485 m/s | 0.0% | 0.0% | 18.1% | 8.1% |

`near_unit_bound_fraction` is the share of action scalar samples with |a| >= 0.95; action-space bounds and definition are in comparison.json. Low-TTC uses TTC < 3 s among geometry-valid risk-evaluable ticks. Covered critical-unobserved uses ticks with both valid risk and policy-observation coverage. Null means no valid denominator or missing source artifact, not zero.

Full denominators, coverage, train/eval summaries, optimization statistics, and artifact paths are in `comparison.json`.
