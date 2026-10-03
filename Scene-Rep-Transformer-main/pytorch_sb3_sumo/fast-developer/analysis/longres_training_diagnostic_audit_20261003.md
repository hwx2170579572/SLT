# Longres training-diagnostic audit (2026-10-03)

## Scope and status

This is a read-only audit of the training-phase artifacts for the existing
`sac_mlp_d1_st_rt_longres_v1` arm at
`runs/sortlr_1003_retry01_longres/sac_mlp_d1_st_rt_longres_v1__intersection_sorted_depart4p0`.
The training artifacts were inspected at about 08:18 +08; the train summary
and shadow summary were last written at 08:17:48 +08. `training_complete.json`
records 100,000 raw steps and 95,001 learner updates. A later readback at about
09:19 +08 found the final-evaluation diagnostic stream complete for 100
episodes. This note audits diagnostic coverage only; checkpoint identity and
final outcome analysis are handled separately, and no evaluation performance
result is reported here.

## Rollout telemetry and episode accounting

`diagnostics/train/summary.json` records 100,000 raw simulation steps,
33,505 policy decisions, and 532 episode records. The outcome counts are 186
successes, 335 collisions, and 10 timeouts, plus one `incomplete` record. Thus
531 records ended at an environment terminal/truncation; the remaining record
was closed before a terminal event and is not a completed episode. The summary
field named `episodes_finished` includes that incomplete record because the
diagnostics close path appends it to the same record list. There were zero
behavior-diagnostic errors. These are training rollouts and must not be used as
the final validation comparison.

The latest `reward_branches.json` snapshot covers the first 530 completed
episodes. Its six mean reward components are:

| Component | Mean per completed episode |
|---|---:|
| Success | 3.509434 |
| Collision | -6.301887 |
| Off-route | 0.000000 |
| Timeout | -0.094340 |
| Step cost | -0.630547 |
| Progress | 2.258919 |

Their sum is `-1.2584209871893792`. The mean `return_policy` from the same
first 530 completed rows in `summary.json` is `-1.2584209871893783`, a floating
point difference of less than `1e-15`. The mean underlying `return_base` over
those rows is `-0.2770650943396226`; it is a separate raw-environment return
and should not be conflated with the shaped policy return. The reward callback
writes only at each tenth completed episode, which explains why the reward
snapshot says 530 while the telemetry has 531 completed records.

## Residual-branch gradient and update observations

`representation_statistics_by_source` in the train summary contains 96
longitudinal actor optimizer diagnostic events and 96 critic events. They are
sampled optimizer-batch/update diagnostics, not episode counts or a census of
all replay samples. The event indices cover update 1 through update 95,000;
the model completed 95,001 learner updates in total. The optimizer hooks collect
post-clipping gradients immediately before the optimizer step and compare
parameters across that step.

| Branch evidence | Actor speed-residual branch | Critic Q-residual branch |
|---|---:|---:|
| Parameters in the audited branch | 1,889 | 8,962 |
| Diagnostic events | 96 | 96 |
| Events with defined gradients for every branch parameter | 96 / 96 | 96 / 96 |
| Events with non-finite gradient elements | 0 | 0 |
| Parameters owned by the corresponding optimizer per event | 1,889 | 8,962 |
| Per-event update delta L2 range | 9.75e-7 to 0.004591 | 0.000616 to 0.004642 |
| Changed parameter elements per event | 11–1,574 | 59–7,897 |

All sampled actor events report the branch active. Across those update-batch
statistics, the pre-tanh actor residual mean ranged from `-0.178817` to `0`,
its mean absolute value from `0` to `0.178817`, and the configured bound was
`0.2`. The reported saturation fraction ranged from `0` to `1`. The critic's
mean absolute Q residual ranged from `0` to `0.829762`. These ranges describe
the sampled update batches only; they are not per-trajectory activation rates.

## Same-state shadow probes

The train shadow summary reports 300 rows over 20 unique policy-input states:
20 states reached the whole-training-phase cap, with a 5,000-raw-step sampling
interval and a maximum of 20 unique training states. The 300 rows correspond
to 15 probe variants at each sampled state; variants are not additional
episodes or environment steps. The summary reports 0 probe errors, 116
applicable rows, 184 inactive/not-applicable rows, and 0 active-invalid rows.
The skip reasons are 180 `branch_inactive` and 4 `no_valid_conflict_relations`.

For the dedicated `longitudinal_residual_off` probe, 16 of the 20 sampled
states had valid conflict relations and were evaluated; the other 4 are
not-applicable because no valid conflict relation was present. They are not
active-invalid cases. In the 16 valid same-state comparisons, the lateral
pre-tanh mean changed by exactly zero in all 16. The absolute speed pre-tanh
mean difference ranged from `6.43e-5` to `0.20000005`, consistent with turning
off the bounded residual. This is a deterministic local sensitivity check,
not evidence that training learned a performance improvement.

## Sampling and interpretation limits

The behavior summary counts real training observations and policy decisions.
The 96 optimizer events instead describe sampled learner updates over replay
batches, and the 20 shadow samples are a capped set of same-state probes. Keep
these denominators separate. The legacy Longres run predates the new
`trajectory_history_audit` collector, so these artifacts do not estimate
short-history or last-valid-index exposure in its rollout trajectories.

The source artifacts contain training configuration and runtime diagnostics;
they do not establish final validation completion or a repaired-versus-legacy
performance effect. The independent bootstrap audit also remains a separate
protocol finding: these runs retained their recorded one-gamma n-step and
timeout behavior.

## Final-evaluation diagnostic readback

The final phase files under `diagnostics/eval/` report 100 complete episode
records, 34,026 raw records, 11,379 policy decisions, zero behavior-diagnostic
errors, and no incomplete episode row. This section records coverage and probe
integrity only; it omits evaluation outcomes and checkpoint identity.

The evaluation shadow JSONL contains 4,500 rows over 300 unique policy-input
states: 15 probe variants per unique state. All 100 episodes have probe rows;
the per-episode unique-state count is 2–4 (maximum 4), within the configured
cap. The summary reports zero probe errors, zero active-invalid rows, 1,644
applicable rows, and 2,856 inactive/not-applicable rows (2,700 branch-inactive
and 156 no-valid-conflict-relation cases).

For `longitudinal_residual_off`, 144 rows are applicable and valid; 156 rows
are NA because no valid conflict relation exists. In every valid same-state
comparison, the lateral pre-tanh mean delta is exactly zero. The signed speed
pre-tanh mean delta ranges from `-0.192094` to `0.199996`; the corresponding
on-branch residual mean has the opposite sign and stays within the configured
`0.2` bound. The baseline saturation-fraction diagnostic ranges from 0 to 1;
96 of the 144 valid sampled states report a nonzero saturation fraction. These
are capped shadow states, not all 100-episode policy decisions.

Conflict-timing prediction coverage is one row per policy decision: 11,379
prediction rows match the 11,379 decisions. Prediction rows dropped at budget
or I/O, calibration rows dropped, right-censored pending predictions, and
calibration pairs dropped at pending capacity are all zero. Calibration has
17,987 pair-level rows; this is a separate unit from decision rows.

This legacy run has no `diagnostics/eval/trajectory_history_audit.summary.json`.
Its behavior, shadow, and conflict-timing diagnostic error/drop counters are
zero, but there is no identity-based last-valid-history exposure count to
report for this run.

## Source map

- Run artifacts: `runs/sortlr_1003_retry01_longres/sac_mlp_d1_st_rt_longres_v1__intersection_sorted_depart4p0/{status.json,training_complete.json,reward_branches.json}` and `diagnostics/{train,eval}/{summary.json,manifest.json,policy_shadow_probes.jsonl,policy_shadow_probes_summary.json,task_conflict_timing_summary.json}`.
- `fast-developer/behavior_diagnostics.py:591-641,720-740`: marks a close-before-terminal record `incomplete`, appends it to the completed-record collection, and serializes outcome counts and episode rows.
- `algos/sb3_torch/callbacks.py:231-286`: the six episodic reward branches and every-tenth-completed-episode rolling snapshot.
- `algos/sb3_torch/longitudinal_residual_policy.py:214-229,399-445`: actor residual-off context and actor/critic optimizer hooks, diagnostic sampling cadence, gradient collection, and post-step parameter deltas.
- `fast-developer/train_intersection_yield_v2_d1.py:350-449,467-563,1406-1475,1600-1665,1710-1745`: shadow row accounting and limits, same-state residual-off fields, representation sampling metadata, training cap, and saved coverage summary.
- `algos/sb3_torch/policy_shadow_probes.py:293-300,351-430`: probe applicability and same-observation probe execution.
