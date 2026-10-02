# v4.13 implementation and experiment ledger

This ledger separates parent evidence, preregistered scientific changes,
engineering recovery, and efficacy evidence.  No formal-test traffic partition
was accessed in any entry below.

## Parent evidence and attribution

- v4.12 development completed all four primary cells.  It retained its Cross
  gains and CARLA result but failed the roundabout-medium guard at `0.80/0.20`.
- Same-checkpoint, exact-seed deployment-prior tests rejected both removal and
  global strengthening of the lane prior as the next route.
- Real-state autograd localized the remaining coupling to `actor.latent_pi`:
  lane and conditional-speed replay losses both updated that trunk, while both
  already had exactly zero support gradient in the shared scene encoder.
- The complete evidence-bound attribution is
  `results_topo_v4_12_dev/development/attribution/d3_roundabout_regression/complete_attribution.json`.

## Scientific method (sealed before implementation)

- Primary: `topo_v4_13_gradient_isolated_tempered_joint_support_prcr_full`.
- The deployed lane logits remain `lane_head(actor_latent)`.  Auxiliary
  replay-lane logits use the same lane-head parameters with
  `actor_latent.detach()`; no second inference head exists.
- The lane categorical NLL scale inside joint replay support is globally
  tempered from `1.0` to `0.25`.  The outer support coefficient, lane prior,
  and component prior remain `0.05`.
- The conditional learned speed-mixture NLL and deterministic learned
  reward-risk-uncertainty score are unchanged from v4.12.
- Required ablations are separate algorithms: unisolated-tempered,
  isolated-untempered, and isolated-tempered without Cross augmentation.
- Forbidden throughout: kinematic projection, TTC/headway thresholds,
  geometry unsafe labels, lane vetoes, confidence gates, shields/fallbacks,
  semantic tie overrides, scenario-conditioned inference, and post-decoder
  action rewrites.

## Engineering attempts and recovery

1. Core construction, gradient ownership, inference equation, serialization,
   and real-SUMO registry tests passed before any development training.
2. Initial Python `DETACHED_PROCESS` smoke attempts `e` through `e10` failed
   before project import with Windows process-initialization status
   `0xC0000142`.  Their logs and both supervisor summaries are preserved; no
   experiment directory or scientific evidence was accepted from them.
3. A hidden `Start-Process` PyTorch probe succeeded, showing that the failure
   was the Windows detached-process creation path rather than the model,
   PyTorch, or SUMO.
4. `r413s/e11/m` was launched by hidden `Start-Process`, independent of the
   interactive terminal.  It completed 96 raw training steps, train-only
   checkpoint calibration and selection, one validation episode, model
   serialization, and 77 exact action-decision trace records.  The smoke
   outcome itself is not efficacy evidence.

## Engineering gate evidence

- New v4.13 targeted tests: 31 passed, 0 failed, 0 errors.
- Inherited v4.11/v4.10 regression tests: 52 passed, 0 failed, 0 errors.
- Static call-graph, parameter-gradient, inherited environment boundary, and
  real runtime trace audit:
  `results_topo_v4_13_dev/engineering/no_kinematic_projection_audit.json`,
  `integrity_passed=true`.
- Lane NLL gradient: deployed lane head nonzero; actor latent trunk, component
  head, speed heads, and scene encoder exactly zero.
- Conditional-speed NLL gradient: actor latent trunk and all learned speed
  heads nonzero.
- Every runtime selected action equals the learned actor proposal and the
  independently recomputed learned score argmax; no action rewrite appears.

## Execution policy

The stage runner uses `r413/{d,a,p,f}` plus immutable recovery suffixes.  Every
job in an entered stage is attempted even if earlier jobs fail; aggregation and
the scientific decision occur only after the batch.  Development must pass
before the three ablations, ablations must complete before fresh promotion,
and the formal partition remains locked until promotion passes.

The background launcher invokes the existing discovery-based latest-family
dispatcher, so completed v4.10-v4.12 families are skipped, v4.13 is current,
and later versioned pipeline files are discovered on subsequent invocations.

## Completed promotion and phase transition (2026-09-05)

- Promotion completed 12/12 accepted runs after all jobs were attempted.  The
  only failed top-level gate is `candidate_development_guards_each_seed`.
- Five of six candidate scenario/seed cells pass.  Cross seed 20 is the sole
  failed cell at Success/Collision `0.70/0.2667`; both outcome thresholds fail,
  while all integrity, feasibility, Off-route, Timeout, and no-rule/no-
  projection checks pass.
- Scenario-average candidate-minus-TemporalGraph Success deltas are Cross
  `+0.0667`, Roundabout-medium `+0.0667`, and CARLA `+0.55`; Collision deltas
  are `-0.0667`, `-0.0667`, and `-0.05`.  Macro Success delta is `+0.2278` and
  worst paired-seed Success delta is `-0.0667`.
- The promotion decision remains `fail`; formal testing is locked and was not
  accessed.
- Complete metrics, seed-level attribution, mechanism-evidence audit, and the
  conditional architecture-freeze decision are recorded in
  `results_topo_v4_13_promotion/V4_13_COMPLETE_METRICS_AND_ARCHITECTURE_FREEZE_AUDIT.md`.
- Decision: stop adding model structure, provisionally freeze v4.13, close the
  direct horizon and route-bias evidence gaps, and move to the versioned v4.14
  limited stability-tuning design in
  `experiments/topo_scene_v4/V4_14_ARCHITECTURE_FROZEN_STABILITY_TUNING.md`.
