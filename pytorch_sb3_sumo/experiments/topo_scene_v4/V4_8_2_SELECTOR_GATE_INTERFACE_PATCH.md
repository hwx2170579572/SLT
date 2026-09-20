# v4.8.2 selector-gate interface patch

## Trigger

The v4.8.1 acceptance-only implementation correctly accepts the immutable H1
artifacts, but its engineering preflight found that the inherited development
gate still requires the obsolete selector mode `joint_checkpoint_decoder`.
The frozen v4.8 selector correctly emits
`tie_only_replicated_joint_checkpoint_decoder`.  Every other H1 outcome,
mechanism, selector, return-estimator, provenance, and binding check passed.

## Single engineering change

For v4.8 candidate rows only, require the selector mode frozen in the v4.8
contract: `tie_only_replicated_joint_checkpoint_decoder`.  Recompute the
selector sub-gate and overall candidate gate after replacing this one inherited
interface check.

The v4.8.1 runtime-diagnostic acceptance adapter is reused byte-for-byte and is
not changed.  v4.8 scientific code, artifacts, result roots, hashes, thresholds,
seeds, job order, promotion, and formal rules remain unchanged.

## Adoption and continuation

- Reproduce the original v4.8 acceptance rejection.
- Reproduce v4.8.1 acceptance of the exact same H1 artifact hashes.
- Reproduce the inherited gate failure with only `selector_mode` false.
- Demonstrate that the v4.8.2 gate passes by changing only that expected mode.
- Adopt H1 in place without rewriting or rerunning it.
- Continue with H2 only after the patch engineering freeze passes.
- Keep formal test locked.

