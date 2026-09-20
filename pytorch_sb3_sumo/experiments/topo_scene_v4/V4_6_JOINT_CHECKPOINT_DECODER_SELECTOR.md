# v4.6 Train-Only Joint Checkpoint–Decoder Selector

Status: preregistered before implementation and before every fresh v4.6
development run.

Created locally: 2026-08-19T12:52:35+08:00.

## Why v4.5 stopped

The frozen v4.5 state machine stopped at E1 (Cross, training seed 4). Its
v4.5.1 checkpoint selector, model-integrity checks, and fusion-decoder checks
all passed, but validation achieved only 3/12 success with 7/12 collisions.
No E2–E4, promotion, or formal-test cell was run.

A decoder-only, same-checkpoint, same-traffic post-hoc intervention changed the
selected best checkpoint from 3/12 to 9/12 success and reduced collision from
7/12 to 3/12 when fusion was replaced by target-critic decoding. The trace
shows that all six regular collisions terminated on `gneE5`; on that edge the
actor repeatedly overrode target-critic keep choices after the route request
had returned to keep. Increasing the fusion confidence threshold is not a
narrow repair because 59.8% of the fixed E1 observations still trigger an
actor override at threshold 0.999.

The opposite decoder preference is present in independent frozen evidence.
For v4.4 Cross seed 2, target-critic train calibration achieved at most 1/12
success, whereas both fusion checkpoint candidates achieved 12/12. Therefore
neither target critic nor fusion can be frozen globally without repeating the
same seed-conditioned failure in the opposite direction.

All values above are post-hoc diagnostic evidence. They select the v4.6
hypothesis but cannot pass a v4.6 gate.

## Single method change

Keep the two already eligible weight checkpoints:

- highest training success (last-20 completed-episode metric, frozen rule);
- exact final checkpoint at the requested raw-step boundary.

Keep the two already implemented deterministic deployment decoders:

- `target_critic`: keep-tie-aware argmax of feasible minimum target-twin-Q;
- `fusion_0_90`: feasible actor non-keep argmax at probability at least 0.90,
  otherwise the same target-critic decoder.

Evaluate the Cartesian product of these two checkpoints and two decoders on
the identical 12 train-partition calibration episodes. Rank the four pairs by:

1. maximize success flag count;
2. minimize collision flag count;
3. minimize off-route flag count;
4. minimize timeout flag count;
5. maximize mean return;
6. prefer exact-final on a complete outcome tie;
7. prefer target critic on a complete checkpoint–decoder tie.

Source terminal flags may overlap. Counts are integral and bounded, every
episode has at least one terminal flag, summary counts must equal immutable
per-episode flags, and no precedence or relabeling is introduced.

The four candidate evaluations must use the same episode seeds and traffic
variants. A hash-bound deployment receipt containing the selected checkpoint
and decoder must be durably written before a validation environment can be
constructed. Validation and test outcomes are forbidden selector inputs.

The tie rules are fixed engineering determinism, not a learned or
scenario-specific branch. Exact-final avoids an arbitrary earlier snapshot on
an outcome tie. Target critic is the lower-intervention decoder because fusion
adds an actor override; this preference is used only after every outcome field,
mean return, and checkpoint preference tie.

## Explicitly unchanged

The v4.5 network, learned parameters, stochastic actor used during training,
critic and target-critic training, replay buffer, n-step return, reward,
encoder, entropy terms, optimizer, action masks, action repeat, checkpoint
definitions, target decoder, fusion rule, 0.90 threshold, environment, traffic
partitioning, outcome gates, promotion thresholds, and formal-test protocol are
unchanged. There is no scenario-name branch.

The only scientific change is that the train-only selector chooses one of four
pre-existing checkpoint–decoder deployments instead of choosing one of two
checkpoints under a globally fixed fusion decoder.

## Falsifiable mechanism predictions

- On fresh Cross cells where fusion is harmful, the train-only matrix should
  select a target-critic pair and pass the non-CARLA outcome guard.
- On fresh Cross cells where target critic is harmful, the matrix should
  select a fusion pair and pass the same guard.
- CARLA must pass its stricter outcome gate regardless of which pair is
  selected; if fusion is selected, its predicate must be exact, and if target
  is selected, target-twin-Q argmax and keep-tie behavior must be exact.
- All four calibration pairs must share a 1.00 seed/traffic signature rate and
  the sealed selected pair must exactly match the later validation deployment.
- Any selector, decoder, evidence, or outcome failure stops the version before
  later development cells.

## Fresh sequential development cells

All cells train from scratch and use training seeds plus calibration/validation
blocks unused by v4.4/v4.5 confirmatory or attribution runs.

1. F1: Cross seed 6, 20k raw steps; train calibration 72000–72011;
   validation 63000–63011. Primary joint-selector confirmation.
2. F2: CARLA seed 5, 20k raw steps; train calibration 73000–73011;
   validation 64000–64011. Preservation and decoder-branch guard.
3. F3: Cross seed 7, 20k raw steps; train calibration 74000–74011;
   validation 65000–65011. Independent Cross stability guard.
4. F4: Roundabout-medium seed 2, 20k raw steps; train calibration
   75000–75011; validation 66000–66011. Third-scenario regression guard.

Historical runs and all v4.5 attribution replays are forbidden as gate inputs.
They are used only to preregister this hypothesis.

## Development, promotion, and formal test

Cross and roundabout-medium require success at least 0.50, collision at most
0.45, off-route exactly 0, and timeout at most 0.30. CARLA requires success at
least 0.50, collision at most 0.10, off-route exactly 0, and timeout at most
0.50. These are unchanged from v4.5 and are noncompensatory.

Only four passing development cells unlock the unchanged 12-job promotion:
candidate and TemporalGraph, Cross/roundabout-medium/CARLA, training seeds 0–1,
50k raw steps, and 30 validation episodes. The candidate applies its joint
train-only deployment selector. TemporalGraph applies its own frozen decoder
and the same two-checkpoint train-only outcome selector. Both methods see the
same 12 unique paired train-calibration traffic episodes; the candidate's four
deployments reuse that same block rather than receiving additional traffic.

Formal test remains inaccessible until a hash-bound passing promotion receipt
exists. Its untouched 120-job matrix contains two methods, six scenarios, ten
training seeds, 100k raw steps, and 50 test episodes per job. Every scenario
must keep success loss within 0.05 and collision increase within 0.05, while at
least one scenario must improve success by at least 0.10 or reduce collision by
at least 0.10.

Infrastructure interruptions remain `TBD`, are archived, and never count as
scientific failures or accepted runs.

