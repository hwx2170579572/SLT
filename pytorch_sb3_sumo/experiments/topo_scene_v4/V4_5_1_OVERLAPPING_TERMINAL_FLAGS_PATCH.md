# v4.5.1: Overlapping terminal-event flags engineering patch

Status: preregistered after the E1 train-only calibration failure and before any
validation or test environment was constructed.

Created locally: 2026-08-19T11:01:02+08:00

## Why this patch exists

The frozen v4.5 E1 run completed its 20,000-raw-step training and both paired
12-episode train-partition calibrations.  Selection then stopped before writing
`selected_model.zip` because the inherited v4.4 selector assumed that success,
collision, off-route, and max-time flags form an exclusive partition.

That assumption is false for the source-compatible environment.  On calibration
seed 68002 the vehicle arrived on the exact raw step that reached the episode
limit, so the immutable episode record correctly contains both `success=true`
and `timeout=true`.  The report contains six success flags, three collision
flags, zero off-route flags, and four timeout flags across twelve episodes.
The final-checkpoint calibration has no overlap.

This was an engineering stop, not a scientific gate result.  No selector
receipt, selected model, validation evaluation, validation trace, or formal-test
artifact existed when this amendment was preregistered.

## Frozen evidence at the stop

- v4.5 contract SHA-256:
  `3c1f7f8bb2f2967f7559907386b58e67760c53dfa8e46bf581fc4824d186bb7c`
- v4.5 implementation freeze SHA-256:
  `9ab0ccc71b33110ac526f5a9f89e8f504d0f3fc0ddafd6407c6ff36dc2414666`
- E1 arguments SHA-256:
  `68058cf9f8ef42da160cdf8610b8b230245025d5ff88d5f7e2d53cdcce97e509`
- training-best checkpoint SHA-256:
  `5f324d6f68a46eea527983b589d89960ba72fe40a9970521403d12665be125d6`
- exact-final checkpoint SHA-256:
  `2d9afa0420fd78e6dec7b60f8b5fb53f5139ccdbb63affe8bd3a29fcc30b7519`
- training-best calibration detail SHA-256:
  `748a504f01db7fe8b07b144f7e0e115825ab57812176b153cf5f51ee40332710`
- exact-final calibration detail SHA-256:
  `e729ff915e5cc8a423959706a0635fa81af76e09982796726da1fabc3058a21c`
- failed launcher log SHA-256:
  `6e097e6ea98c2035879da509e9b4b3f5eb302f1ab7d3ecd8fb0d76b4121e43c6`

## Single engineering change

Replace only the selector's invalid exclusivity assertion with explicit
source-event semantics:

1. Each of the four event rates must still imply an integral count in the
   paired calibration population and remain between zero and the episode count.
2. Counts must exactly match the immutable per-episode event flags.
3. Every episode must contain at least one terminal event flag.
4. Event flags may overlap; the overlap count and affected episode/seed pairs
   are recorded in the selector receipt.
5. The selection key remains exactly the preregistered tuple: maximize success
   count, minimize collision count, minimize off-route count, minimize timeout
   count, maximize mean return, then prefer exact-final on a complete tie.
6. Original summaries and episode records are preserved byte-for-byte.  No
   event precedence, relabeling, metric normalization, or gate change is added.

## Explicitly unchanged

The v4.5 model, training trajectory, reward, encoder, critics, entropy settings,
0.90 actor-fusion threshold, deterministic decoder, action masks, checkpoint
candidates, calibration seeds, validation seeds, scenario order, development
gates, promotion gates, and formal-test protocol remain unchanged.  Existing
frozen v4.5 files are not modified.

## E1 recovery rule

E1 may resume from the two already-completed calibration artifacts only if all
listed input hashes, paired episode signatures, checkpoint CRCs, decoder traces,
and the absence of validation artifacts are revalidated.  The v4.5.1 selector
receipt must be durably written before construction of the validation
environment.  Training and calibration must not be rerun or selectively
altered.  E2-E4, if unlocked, start from scratch and use the same selector patch.

Any hash drift, missing terminal flag, count mismatch, paired-seed mismatch,
decoder mismatch, pre-existing validation artifact, or test-partition access is
a hard stop.  Promotion and formal testing remain locked.
