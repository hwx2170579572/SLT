# v4.8.1 runtime-diagnostic acceptance patch

## Trigger

The frozen v4.8 H1 scientific run completed successfully, sealed its selector
before validation, and wrote all required artifacts.  The v4.8 runner rejected
the run only because `return_estimator_diagnostics.json` does not contain the
newly invented field `return_estimator_changed_from_v4_7`.  The runtime
diagnostic instead retains the inherited field `return_estimator_changed=true`,
whose reference point is the older four-step control.  The v4.8 run-level
fidelity record separately and correctly states
`return_estimator_changed_from_v4_7=false`.

## Single engineering change

Accept the frozen runtime diagnostic using fields it actually emits:

- candidate `n_step == 16`;
- `horizon_correct_bootstrap == true`;
- `bootstrap_discount == gamma_power_actual_horizon`;
- finite, nonempty runtime horizon statistics;
- inherited `return_estimator_changed == true` relative to the old control;
- run-level `implementation_fidelity.return_estimator_changed_from_v4_7 == false`;
- run-level `implementation_fidelity.training_changed_from_v4_7 == false`.

The patch must not write, reinterpret, repair, or regenerate H1 artifacts.  It
must first reproduce the frozen runner's rejection, then demonstrate that the
patched acceptance admits the exact same hashes.

## Frozen invariants

- v4.8 scientific source files and implementation freeze are immutable.
- Training, return calculation, checkpoint weights, selectors, decoders,
  traffic blocks, gates, development order, promotion, and formal protocol do
  not change.
- H1 is adopted in place; it is not rerun.
- Subsequent scientific jobs continue to bind the v4.8 scientific freeze hash,
  while the v4.8.1 patch freeze is checked separately by orchestration.
- Formal test remains locked.

