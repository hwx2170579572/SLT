# v4.5 Confident-Actor Fusion Decoder

Status: preregistered before implementation and before every fresh v4.5
development run.

## Why v4.4 stopped

v4.4 passed D1 (CARLA seed 3) and stopped at D2 (Cross seed 2). The train-only
checkpoint selector was internally valid, but its selected best checkpoint
achieved 0/12 success, 9/12 collision, and 3/12 timeout on the independent
validation block.

The decoder-only post-hoc intervention is unusually sharp. On the identical
best checkpoint and validation seeds 56000--56011, replacing target-twin-Q
lateral arbitration with the proposed fusion changed success from 0/12 to
11/12 and collision from 9/12 to 1/12. Eleven paired episodes changed from
failure to success and none regressed. On the exact-final checkpoint, fusion
achieved 12/12 success on both the paired train calibration block and the
same validation block. On D1 CARLA, fusion preserved the target-decoder's
12/12 success and exact lane-command rates.

All of those values are post-hoc diagnostic evidence. They select the v4.5
hypothesis but cannot pass any v4.5 gate. The threshold of 0.90 is explicitly
declared as post-hoc selected and is frozen before fresh evidence is collected.

## Single method change

For each deterministic decision, enumerate the three action-mask-feasible
hybrid actor actions (left, keep, right), including their lane-conditioned
longitudinal actions.

1. Let `actor_index` be the masked actor-probability argmax.
2. Let `target_index` maximize the minimum target-twin-Q value, retaining the
   v4.3 rule that exact Q ties prefer keep.
3. If `actor_index` is non-keep and its masked actor probability is at least
   0.90, deploy that actor-indexed hybrid action.
4. Otherwise deploy the target-indexed hybrid action.

The predicate depends only on the current observation, action mask, actor
probabilities, and frozen threshold. It contains no scenario-name branch and
no validation/test feedback. Stochastic actor calls used for replay collection
and training remain byte-for-byte delegated to the v4.2 actor.

The training objective, encoder, reward, entropy terms, replay buffer, n-step
return, optimizer, environments, action repeat, checkpoint candidates, paired
train-only calibration, lexicographic checkpoint selector, and exact-final tie
preference all remain unchanged from v4.4.

## Falsifiable mechanism predictions

- Fresh Cross policies must exercise a high-confidence actor non-keep override
  in at least one episode, exactly obey the fusion predicate, and meet the
  non-CARLA outcome guard.
- Fresh CARLA policies must exercise the target-critic fallback, exactly obey
  the fusion predicate, and retain the stricter CARLA outcome guard.
- A second fresh Cross training seed must independently pass, preventing a
  one-seed decoder rescue from being mistaken for a stable method.
- Roundabout-medium must pass as a third-scenario regression guard.
- Any failed outcome, decoder-integrity, selector-integrity, or evidence gate
  stops the version without running later development cells.

## Fresh sequential development cells

The cells use training seeds and train/validation seed blocks not used by the
v4.4 D1/D2 runs or their post-hoc replays.

1. E1: Cross seed 4, 20k raw steps; train calibration 68000--68011;
   validation 59000--59011. Primary causal confirmation.
2. E2: CARLA seed 4, 20k raw steps; train calibration 69000--69011;
   validation 60000--60011. Preservation and fallback-branch guard.
3. E3: Cross seed 5, 20k raw steps; train calibration 70000--70011;
   validation 61000--61011. Independent seed-stability guard.
4. E4: Roundabout-medium seed 1, 20k raw steps; train calibration
   71000--71011; validation 62000--62011. Third-scenario regression guard.

Every cell trains from scratch. Both eligible checkpoints are evaluated with
the frozen method's own deterministic deployment decoder on paired training
traffic. The selector receipt, checkpoint hashes, and calibration traces must
be sealed before the validation environment is constructed. Historical or
post-hoc outcomes are forbidden as gate inputs.

## Promotion and formal test

Only four passing development cells permit the unchanged 12-job promotion:
candidate and TemporalGraph, three scenarios, and seeds 0--1, trained for 50k
raw steps and evaluated on 30 validation episodes. Each method uses the same
train-only selector protocol with its own frozen deployment decoder.

Formal test remains inaccessible until a hash-bound passing promotion receipt
exists. The untouched formal matrix contains 120 fresh jobs: two methods, six
scenarios, ten training seeds, 100k raw steps, and 50 test episodes per job.
Acceptance remains noncompensatory: every scenario must keep success loss
within 0.05 and collision increase within 0.05, while at least one scenario
must gain at least 0.10 success or reduce collision by at least 0.10.

Infrastructure interruptions remain `TBD`, are archived, and never count as
scientific failures or accepted runs.
