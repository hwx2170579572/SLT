# v4.8.3 promotion reconciliation V3

Status: post-run engineering reconciliation; scientific results were already
observed before this document was written.

## Trigger

The V2 automation completed all twelve promotion jobs and correctly continued
after every reported job failure.  It then reported only 6/12 accepted.  All
six reported failures shared these facts:

- subprocess return code: `0`;
- all required training, selector, validation, and diagnostic artifacts exist;
- reported reason: `v4.8 changed return estimator`;
- affected method: candidate only.

The authoritative command
`tools/run_topo_v4_8_3_experiments.py status` reports development 4/4,
promotion 12/12, and formal 0/120.

## Root cause

`run_remaining_v4_8_3_promotion.py::job_state` directly calls
`accepted_run_reason_v4_8_3` after the child runner exits.  This call occurs
outside `_patched_v4_8_3_interfaces`.  For a candidate job the delegated
frozen validator therefore sees the original v4.8 runtime-diagnostic
interface, not the inherited v4.8.1 adapter.  The emitted diagnostic correctly
contains the legacy `return_estimator_changed=true` field and intentionally
does not invent `return_estimator_changed_from_v4_7`; outside the context the
latter is incorrectly required and produces a false negative.

Inside the complete v4.8.3 context, all six candidate artifacts are accepted.
No training, selector, decoder, evaluation, seed, threshold, or scientific
gate changed.

## V3 correction

`tools/reconcile_v4_8_3_promotion_v3.py` is a new file.  It preserves V1 and
V2 and performs only the following:

1. Require the completed V2 6/12 false-negative pattern and formal count zero.
2. Hash all required artifacts of all twelve immutable run directories.
3. Re-evaluate every run inside the full v4.8.3 patch context.
4. Require the authoritative frozen-runner status to be promotion 12/12 and
   formal 0/120.
5. Ask that runner to summarize promotion and compute the preregistered gate.
6. Re-hash all run artifacts and reject any mutation.
7. Write a separate V3 reconciliation receipt and state file.

The script has no command capable of launching training or formal testing.
Gate computation is allowed only after promotion is authoritatively complete.
Formal remains unexecuted regardless of whether the promotion decision is
`pass` or `fail`.

## Interpretation discipline

V2 remains an audit artifact demonstrating that all jobs were attempted and
that its post-child acceptance check was defective.  V3 does not reinterpret
or replace any metric.  Scientific conclusions must use the frozen promotion
gate and the real per-run values returned by the authoritative runner.  If the
gate fails, formal stays locked and the next action is a new versioned failure
attribution followed by a fresh development stage.
