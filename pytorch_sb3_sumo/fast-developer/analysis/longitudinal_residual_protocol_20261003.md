# ST-RT bounded longitudinal residual: prospective protocol (2026-10-03)

Status: implementation and preflight, not a result. User explicitly selected the bounded longitudinal residual candidate and authorized two fresh seed-0 100k runs, followed by a bootstrap audit and bounded decision-window interventions.

## Question and comparison

Can a conflict-conditioned branch improve safe task completion when its direct policy effect is restricted to longitudinal control, instead of feeding conflict features into shared social K/V?

- Control: `sac_mlp_d1_st_rt`.
- Candidate: `sac_mlp_d1_st_rt_longres_v1` (new method; existing method definitions remain intact).
- Main scene: `intersection_sorted_depart4p0`, the sorted three-background-flow intersection with depart scale 4.0. This is not a DARRL scene.
- Both: seed 0, fresh initialization, 100000 raw simulation steps, CUDA workers, the same existing training/evaluation protocol and final-checkpoint evaluation on 100 episodes (logical seeds 10000–10099).
- No warm start, no reuse of replay, no selection of a best checkpoint for the primary comparison.
- Preserve `environment_step_reward_v2` and report its six components separately from the underlying raw environment return.
- Preserve the current released-source-compatible n-step/timeout behavior in BOTH arms. The independent bootstrap audit must not silently change the active experiment.

The architecture preserves the original ST-RT shared encoder and the actor's lateral path. With fresh training this does not preserve old learned lateral weights: learning can still indirectly change both actor outputs. Only the *direct same-state effect* of the new branch is restricted to speed.

## Candidate implementation contract

1. Use the existing public conflict-timing observation construction, with its validity masks, static network geometry and low-speed NA semantics. Do not read another vehicle's hidden future route.
2. Keep conflict features outside the shared ST-RT extractor and social K/V.
3. Add a private, zero-output-initialized actor branch. Its bounded scalar changes the speed dimension of the pre-tanh Gaussian mean, before the existing sampling and log-probability calculation. Do not edit already sampled actions while keeping the old log probability.
4. Initial pre-tanh residual bound: 0.2, applied to action dimension 0 (speed), not dimension 1 (lane request). The environment maps normalized speed action `a` to `clip(5*(a+1), 0, 10)` m/s. With the same Gaussian noise and unchanged log-standard-deviation, the strict maximum target-speed difference is `10*tanh(0.1) = 0.99668...` m/s. Keep the existing log-standard-deviation path.
5. Give the critic private access to the same extra conflict context through zero-output-initialized Q residuals. This prevents an unexamined actor-only information asymmetry. The private critic paths must not become new shared-encoder inputs. Record their parameter counts and optimizer/target participation.
6. Unsupported observations must have an explicit inactive/NA state and zero residual, rather than treating missing conflict geometry as a valid zero-risk estimate.
7. Preserve baseline initialization RNG where possible and verify initial actor/Q equivalence with controlled inputs. Save actual configuration and source provenance for each run.

This is a minimal *mechanism* comparison against ST-RT, not a parameter-matched comparison. Relative to the previous ConflictTiming method, it also changes gradient paths and critic conditioning; improvement would not, on its own, prove that lateral coupling caused the old method's failure.

## Checks and passive diagnostics

- Pure tests: only speed mean changes; lateral output and log-std stay unchanged under branch on/off at fixed weights and state; boundedness; zero initialization; masked/unsupported cases; finite gradients; actor/critic optimizer ownership; target-network update; checkpoint save/load.
- Use explicitly named pure tests. Do not invoke broad environment test files that may start unbudgeted SUMO rollouts.
- Before full launch, use an explicitly bounded smoke run if needed, store it outside formal output directories, and report its simulation budget separately.
- Normal training/evaluation: proposal action, forwarded action, target/actual speed, actual lane, route eligibility, terminal cause and partner evidence, reward-component reconciliation, raw/decision steps, branch coverage, activation, saturation, gradients, and actual parameter updates.
- Same-state branch-off probes must verify exactly unchanged lateral output; record speed differences and active/NA/errors. Training sampling is triggered at 5000-raw-step intervals and capped at 20 unique states over the entire training phase; evaluation is capped at four unique states per episode. Variant rows are not episodes.
- Report success/collision/timeout/off-route, paired outcome changes, and successful completion time. Distinguish functional sensitivity from performance causality.

## Execution order and scope

After implementation/preflight, start both authorized formal workers. Then finish the independent n-step/timeout audit without changing their protocol. Perform bounded decision-window interventions on an existing frozen ST-RT checkpoint; these do not require waiting for the new 100k runs and must not update model or replay.

The intervention protocol will explicitly record fixed cases, prefix-equality checks, action/continuation definitions and a hard simulation budget. A successful rescue only establishes that a tested alternative improved that particular continuation; failure to rescue does not establish that a collision was unavoidable.

Update the original scene-specific experiment summary and research records when each formal run completes. Preserve single-seed and historical-source limitations.

## 2026-10-03 formal launch and recovery status

The first formal root, runs/sortlr_1003, was manually interrupted by the user. Its last saved raw progress was 9276/4276 updates for ST-RT and 8679/3679 for Longres. There is no exact-resume state because the run has no periodic/final checkpoint, replay buffer, or RNG/optimizer continuation state; the partial raw work remains in the interruption receipt and is not merged into a fresh arm.

The replacement comparison is ST-RT in runs/sortlr_1003_retry01 and Longres in runs/sortlr_1003_retry01_longres. Both manifests specify fresh seed 0, no resume, 100000 raw steps, CUDA, sorted/depart4, and final validation on logical seeds 10000–10099. At the latest saved progress snapshot at 04:23 +08 on 2026-10-03, ST-RT had 18852 raw/13852 updates and Longres had 12571 raw/7571 updates; child status files said training and launch manifests said running. No final validation or performance result exists yet. The shared-root Longres attempt exited with WinError 5 while replacing a traffic overlay; it remains a failed candidate outside this selected pair.

The split roots have the same 30 depart4 traffic route templates by filename and SHA256. Their source archives contain 8736 entries each; 16 checked worker-critical source files match. The only source-manifest differences are the launch-only helper and the standalone decision-window diagnostic script, neither imported by the training worker. Full source/template evidence and root identities are in runs/sortlr_1003_pair_recovery_receipt_20261003.json.

The earlier runs/sortlr_smoke_1003 run is only a 300-raw-step/one-episode pipeline smoke. Its shadow probe had a pre-residual hook defect that was fixed before the formal runs; do not interpret smoke outcomes as method results. The separate strt_window_1003 diagnostic stopped at 13 episodes/3340 raw steps with its physical snapshot missing and is not merged into this comparison.

One saved 10k-boundary policy-input tensor from each selected root was available in the small policy_observations stream. Both trajectory arrays have shape 6x10x5. Applying the production first-coordinate x!=0 proxy gives 10 valid frames in every slot; the count-minus-one index and last proxy-valid index are both 9 in all slots, with no mismatch in these two rows. This is a two-snapshot observation only, not an estimate of mismatch prevalence or true actor presence. The log does not retain actor identity, first-tracked/history-reset time, or a separate validity mask, so it cannot reconstruct late-entrant history age; the original interrupted run saved no policy-observation rows. See trajectory_audit in the pair recovery receipt.

The independent [last-valid history index audit](last_valid_history_index_audit_20261003.md) confirms the D1 left-padding/count-minus-one indexing risk in the as-run encoder. The two saved threshold tensors do not show that condition, and they cannot estimate its traffic frequency or performance effect. The formal workers were not hot-patched or stopped; retain their as-run code and metrics and revisit module attribution. Do not extend this D1 finding to MST+SLT's separate first-frame mask behavior.
