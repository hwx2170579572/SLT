# v4.9.2.2 immutable interrupted-run recovery

## Trigger

The original attempt-all process was externally interrupted while
`P__cand__cross__s21__p9e99845d` had recorded 49,062 of the frozen 50,000 raw
steps. The directory has no final training diagnostics, selector receipt,
selected model, or validation evaluation and is therefore invalid scientific
evidence.

## Recovery rule

The incomplete directory is preserved byte-for-byte. It is neither resumed,
renamed, deleted, nor overwritten. The same frozen v4.9.2 command is executed
from initialization with the same:

- algorithm and implementation;
- scenario and seed;
- 50,000-step training budget;
- train calibration and validation evaluation seeds;
- hyperparameters and action repeat;
- experiment, attribution, and implementation-freeze hashes.

Only `--output-dir` and `--model-name` change so the fresh attempt is written
to `engineering_recovery_v4_9_2_2/runs/...__recovery_rN`. A recovery attempt
is accepted by all v4.9.2 scientific/model-integrity validators plus the
v4.9.2.1 formal-access provenance validator.

## Attempt-all behavior

`tools/run_all_v4_9_2_2_recovery.py` visits all 12 logical jobs. It selects the
first complete accepted source, runs every unresolved job sequentially, does
not cancel later jobs after a failure, and aggregates only after all attempts.
Future interrupted attempts receive a new `recovery_rN` directory; all earlier
attempts remain preserved.

`tools/run_all_pending_promotions_v4_9_2_2.py` is the canonical versioned
registry for the current and later promotions. It never starts formal tests.

## Scientific boundary

This patch changes no model or experiment decision. It adds no kinematic
projection, TTC/headway threshold, lane veto, geometry label, confidence gate,
or post-decoder action rewrite. Formal testing remains locked unless the
complete selected-source matrix passes every frozen promotion gate.

