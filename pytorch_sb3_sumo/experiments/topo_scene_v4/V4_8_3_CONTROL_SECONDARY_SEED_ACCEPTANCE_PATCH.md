# v4.8.3 control secondary-seed acceptance patch

## Trigger

The first frozen promotion job, TemporalGraph / Cross / seed 0, completed its
50,000 raw training steps, train-only checkpoint calibration, and 30 validation
episodes with subprocess return code 0.  The frozen v4.8.2 stack nevertheless
rejected the complete run with exactly
`argument secondary_calibration_seed_start mismatch`.

The control runner correctly has no secondary calibration round.  Its
`requested_raw_steps` therefore omits `secondary_calibration_seed_start` (JSON
lookup value `null`), its fidelity record has a null secondary offset, its
selector is `checkpoint_only_parent_control`, and no `cal_secondary` directory
exists.  The inherited acceptance table incorrectly requires
`calibration_seed_start + 100` for both the v4.8 candidate and TemporalGraph.

## Single engineering change

For TemporalGraph control jobs only, require the emitted control semantics:

- `requested_raw_steps.secondary_calibration_seed_start` is null or omitted;
- `implementation_fidelity.secondary_calibration_seed_offset` is null;
- the control selector has no secondary trigger, seed, candidates, or
  secondary-calibration directory.

After those control-only guards pass, provide a read-only normalized view of
the one argument field to the frozen v4.8 acceptance function so that every
other frozen provenance, selector, model, trace, return-estimator, training,
evaluation, and finite-value check runs unchanged.  The JSON file on disk is
never rewritten.  A non-null control value is rejected before normalization.

Candidate jobs do not enter the adapter.  They continue to require the frozen
`calibration_seed_start + 100` argument and all v4.8.1/v4.8.2 checks unchanged.

This is acceptance-only engineering.  It changes no training, return
estimator, checkpoint, selector, decoder, metric, threshold, seed, traffic,
result root, scientific gate, or scientific implementation hash.

## Verification and continuation

- Reproduce the exact v4.8.2 rejection on the immutable promotion run.
- Accept the same run only through the v4.8.3 adapter while hashing all 13
  required run artifacts before and after.
- Prove that an in-memory non-null control mutation is rejected.
- Prove that an in-memory null candidate mutation is still rejected.
- Re-accept an existing immutable candidate development run.
- Run targeted tests and the full regression suite in `llm_pipeline`.
- Freeze the engineering patch, then adopt the completed promotion run in
  place and continue with `P__tg__cross__s1__p42f54135`.
- Keep the formal test locked until the unchanged promotion gate passes.

## Engineering regression finding after implementation

The first whole-suite run is retained as failed evidence.  Fifty setup errors
all came from pytest trying to create `tmp_path` under an inaccessible C-drive
directory.  The single assertion failure was the frozen v4.8.2 H1-adoption
test's historical expectation that development was still incomplete and H2
was next.  Development has since legitimately completed H1--H4 and its frozen
decision is `pass`, so that time-specific assertion can no longer be true.

The first corrected attempt moved `--basetemp` into the nested D-workspace
result directory.  It cleared all permission errors and reached 354 passes,
but three file-creating tests then exposed Windows path-length error 206; this
second XML is retained too.  The final run uses a short path at the same D-drive
workspace root.  It deselects exactly the one obsolete state snapshot and adds
a v4.8.3 replacement test proving all four development prerequisites are
accepted, development is complete/pass, `next_job` is null, and formal remains
locked.  No scientific or mechanism test is removed, the executed test count
is not reduced, and both failed XML files plus the exact exception decision are
sealed as engineering evidence.
