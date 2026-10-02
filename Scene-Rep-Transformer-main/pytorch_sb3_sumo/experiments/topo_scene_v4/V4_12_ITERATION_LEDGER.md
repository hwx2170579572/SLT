# v4.12 implementation and experiment ledger

This ledger distinguishes the preregistered scientific method from engineering
repairs made before development evidence was opened.  No formal-test traffic
partition was accessed in any entry.

## Scientific method (sealed before implementation)

- Primary: `topo_v4_12_augmented_joint_support_prcr_full`.
- Model changes: masked replay-lane categorical NLL, inherited conditional
  speed-mixture NLL, continuous learned lane log-probability in the learned
  proposal score, and the existing random-rotation augmentation enabled for
  Cross training/collection.
- Required ablations are separate algorithms:
  `topo_v4_12_joint_support_no_cross_augmentation` and
  `topo_v4_12_cross_augmentation_only`.
- Forbidden throughout: kinematic projection, TTC/headway threshold, geometry
  unsafe label, lane veto, confidence gate, shield/fallback rule, fixed speed
  grid, semantic tie override, and post-decoder action rewrite.

## Engineering attempts (immutable)

1. `r412s/e0/m`: training completed, but the long descriptive TensorBoard path
   used by the first invocation exceeded the Windows legacy path budget before
   the complete selector/evaluation chain could be accepted.  This directory is
   retained and is not scientific evidence.
2. `r412s/e1/m`: the short-path retry reached checkpoint calibration and exposed
   an SB3 restore bug.  `AugmentedJointSupportPRCRSACV412.__init__` asserted the
   policy type while SB3 intentionally held an `_init_setup_model=False` shell.
   The traceback is retained in `r412s/smoke_e1.stderr.log`.
3. Engineering repair: component invariants are now checked after
   `_setup_model()` and skipped only during the uninitialised restore shell.
   A save/load round-trip regression test was added.  This changes no loss,
   score, action, reward, observation, data split, or scientific hyperparameter.
4. `r412s/e2/m`: the repaired short-path smoke completed training, primary and
   secondary train-partition calibration, target-only checkpoint selection,
   policy-state-preserving materialisation, and validation rollout.  Exact
   learned joint-score/action integrity rates are all 1.0.  The smoke outcome
   itself is not used to claim efficacy because it contains only one episode.

## Engineering gate evidence

- New v4.12 tests: 31 passed, 0 failed, 0 errors.
- Inherited v4.11/v4.10 regressions: 52 passed, 0 failed, 0 errors.
- Static, dynamic-gradient, environment-boundary, and real runtime audit:
  `results_topo_v4_12_dev/engineering/no_kinematic_projection_audit.json`,
  `integrity_passed=true`.
- Real smoke: `r412s/e2/m`; the formal partition remained untouched.

## Execution policy

Stage runners use deterministic short paths and immutable recovery suffixes.
An individual job failure or launcher exception is recorded and does not cancel
remaining jobs in the entered stage.  Aggregation occurs after all selected jobs
are attempted.  Scientific gates, rather than fail-fast process behavior,
control entry into ablation, promotion, and the untouched formal test.
