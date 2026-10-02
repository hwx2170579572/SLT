# Frozen RouteAct / ConflictTiming interventions (2026-10-03)

This is a fixed-checkpoint diagnostic on `intersection_sorted`, `depart_scale=4.0`, validation traffic. It ran two one-episode control checks, then three 40-episode interventions on logical seeds 10000–10039, for exactly 122 episodes and at most two concurrent CPU workers. No policy learning or replay updates occurred. Each arm reused the same 30 traffic templates; all paired seeds had matching template names and zero mismatches.

The coordinator completed at 01:03:41 +08. Both control gates passed exactly on seed 10000: RouteAct reproduced a success on `traffic_10.rou.xml`, and ConflictTiming reproduced a collision on that same template. These are single-seed control checks, not proof of full 100-episode identity.

| Frozen policy / intervention | Success | Collision | Timeout | Off-route | Mean shaped return (SD) | Mean raw return | Raw / decision records | Diagnostic errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RouteAct checkpoint, route veto removed | 6/40 | 26/40 | 8/40 | 0/40 | -5.441 (7.602) | -0.500 | 15,575 / 5,202 | 0 |
| ConflictTiming checkpoint, full conflict residual off | 19/40 | 20/40 | 1/40 | 0/40 | 1.315 (10.652) | -0.025 | 9,690 / 3,242 | 0 |
| ConflictTiming checkpoint, timing channels off | 11/40 | 28/40 | 1/40 | 0/40 | -3.158 (9.505) | -0.425 | 11,256 / 3,768 | 0 |

The six per-episode reward components reconcile to the shaped `environment_step_reward_v2` return with maximum absolute error at most `3.56e-15`. The raw return is reported separately and uses `info.undiscounted_reward_or_step_reward_fallback_v1`; do not treat it as the shaped return or compare it directly with older raw-return summaries.

The eight timeouts after removing RouteAct's route veto were not homogeneous. The existing episode summaries show known route context for all 600 raw ticks in every timeout (unknown ticks: 0). Six ended on `-E1_0`, which is known not to reach the planned next edge `-E0`; their terminal actual speeds were 0 despite positive target speeds (0.080–7.241 m/s). The other two ended on internal junction lanes `:J1_21_0` and `:J1_14_0`, both labeled route-eligible at termination, with target and actual speeds around 0.01 m/s and lane changes not issued because the vehicle was on an internal lane. Thus six have direct terminal evidence of an incompatible lane and unrealized target speed; the two internal-lane waits remain a distinct case. The compact episode summaries do not contain terminal lane-position meters, so exact stop positions within those lanes are unknown. Per-episode values are recorded in `routeact_no_veto_timeout_episodes` in the result JSON.

Paired transition matrices below are **official frozen reference outcome rows → intervention outcome columns**. Each matrix has 40 matched seeds; off-route counts were zero.

**RouteAct reference first 40: S15/C25/T0 → no-veto intervention S6/C26/T8.**

| Reference ↓ / intervention → | Success | Collision | Timeout |
|---|---:|---:|---:|
| Success | 5 | 8 | 2 |
| Collision | 1 | 18 | 6 |
| Timeout | 0 | 0 | 0 |

**ConflictTiming reference first 40: S12/C26/T2 → full-residual-off S19/C20/T1.**

| Reference ↓ / intervention → | Success | Collision | Timeout |
|---|---:|---:|---:|
| Success | 6 | 6 | 0 |
| Collision | 12 | 13 | 1 |
| Timeout | 1 | 1 | 0 |

**ConflictTiming reference first 40: S12/C26/T2 → timing-channels-off S11/C28/T1.**

| Reference ↓ / intervention → | Success | Collision | Timeout |
|---|---:|---:|---:|
| Success | 3 | 9 | 0 |
| Collision | 6 | 19 | 1 |
| Timeout | 2 | 0 | 0 |

These fixed-policy results suggest the complete ConflictTiming residual and the timing subchannels should not be treated as interchangeable: full residual removal and timing-only removal moved the 40-pair outcomes in different directions. The RouteAct no-veto result also shows that its deployment-time route guard matters to this frozen policy on this subset; it does not establish how a policy trained without the guard would learn. The matched episodes span only 30 reused layouts, and each checkpoint comes from one training seed, so these are descriptive intervention results, not training-seed confidence intervals or a general performance guarantee.

The five evaluation files (two one-episode control gates plus three 40-episode arms; 122 episodes total) were each checked individually. In all five, the checkpoint SHA equals the expected source checkpoint; `model_policy_fingerprint` has identical before/after state-dict SHA and `_n_updates=95001`; `learning_or_replay_updates=0`, `resume=false`, and `policy_shadow_probes=false`. Reward protocol is `environment_step_reward_v2`; six component means reconcile with shaped mean return with max absolute error 0 for the B control and at most `3.56e-15` in the other files. Raw return is a separate field. The B intervention call counts (3,242 and 3,768) are actual policy predictions run under the selected feature override, not additional shadow-suite predictions. The B task recorder wrote one row per decision; calibration streams had no dropped or pending samples at close.

|Evaluation file|Episodes|Checkpoint SHA matches|Before/after fingerprint|Updates|Shadow suite|Policy predictions under intervention|Reward reconciliation error|
|---|---:|---|---|---:|---|---:|---:|
|`control_smoke/a_control/evaluation_results.json`|1|yes|equal|95001|off|0|`1.78e-15`|
|`control_smoke/b_control/evaluation_results.json`|1|yes|equal|95001|off|0|0|
|`intervention/a_no_veto/evaluation_results.json`|40|yes|equal|95001|off|0|`3.55e-15`|
|`intervention/b_conflict_off/evaluation_results.json`|40|yes|equal|95001|off|3242|`3.55e-15`|
|`intervention/b_times_off/evaluation_results.json`|40|yes|equal|95001|off|3768|`3.55e-15`|

The full source records and matrices are in [the result JSON](sortct_frozen_intervention_results_20261003.json), [paired outcomes](../../../../runs/sortct_frozen_1003/paired_outcomes.json), and [completion/manifest annotation](../../../../runs/sortct_frozen_1003/manifest_annotation.json). The original [experiment manifest](../../../../runs/sortct_frozen_1003/experiment_manifest.json) is preserved byte-for-byte, but its `mode` field says `read_only_preview` because the coordinator copied the preflight value. The independent annotation records the actual completed execution. The source archive contains 841 files; the executed runner is archived at SHA-256 `CE08FAE56D0F1C875ECE5CFD2A2D44409C6F91DF07D660B995A161B98072DFEC`. A later source-only fix changed the future-runner mode assignment (current SHA-256 `0B816B4E0A36593DDA8D793A748F5ECBD7FD9C6AFE0CB9E404AF2512976CD6E6`) and did not alter these completed outputs. The original run manifest was not rewritten; its actual execution is documented in the separate annotation.
