# Why the current Proposed results appear stronger than the paper

## Scope and conclusion

This audit concerns the **complete Proposed method** (MST + SLT + SAC), not the
plain SAC baseline.  It uses only formally completed v12 runs: five seeds each
for `left_turn`, `cross`, and `roundabout_easy`, plus the single completed seed
for `roundabout_medium`.  Interrupted directories are excluded.

The current numbers do **not** show that the migrated method has surpassed the
paper under the paper protocol.  The apparent gain has three dominant causes:

1. the current source-first episode limit is longer than the paper limit for
   Double Merging (and will also be longer for Roundabout-C);
2. direct SUMO ego control bypasses the SMARTS Ackermann lane-following
   controller and its strong curve-speed limits;
3. training and testing reuse the same released traffic-parameter XML pool,
   whereas the paper says test traffic uses different random settings/seeds.

Fixed ego starts, source/paper hyperparameter conflicts, simulator-version
differences, and different checkpoint/statistical aggregation add secondary
non-equivalence.  Results from this profile must therefore be labelled
**source-release/SUMO transfer results**, not direct paper reproductions.

### Explicitly excluded explanation: the plain-SAC baseline architecture

Whether a separately trained plain-SAC baseline accidentally consumes the
hierarchical Transformer cannot explain why the **complete Proposed method**
looks stronger than the paper.  The Proposed method is supposed to contain the
hierarchical scene-representation Transformer, and its completed v12 models,
rollouts, and metrics are independent of the later baseline reconstruction.

The stopped v13 SAC run did have a baseline-fidelity problem: it used the
Proposed trajectory field semantics instead of the released dormant
`STATE_LSTM` adapter contract.  That issue invalidates v13 as a formal SAC
baseline comparison only.  It neither changes nor inflates any Proposed v12
result.  v13 is marked non-formal and excluded; v14 reconstructs SAC with the
released ego/social `STATE_LSTM` fields, GRU encoder, and time mask, with no
Transformer input.

## What the completed results actually show

| Scenario | Current Proposed success | Paper Proposed success | Delta | Current successful time | Paper successful time | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Unprotected Left Turn | 0.980 (5 seeds, 250 episodes) | 0.940 (50 test episodes reported) | +0.040 | 11.87 s | 12.5 s | Small difference; sampling and aggregation are not matched. |
| Double Merging | 0.984 (5 seeds, 250 episodes) | 0.960 | +0.024 | 45.72 s | 28.6 s | Not stronger on efficiency; success is dominated by the longer time limit. |
| Roundabout-A | 0.972 (5 seeds, 250 episodes) | 0.880 | +0.092 | 12.86 s | 24.5 s | Abnormally fast; direct SUMO speed control is the principal confirmed cause. |
| Roundabout-B | 0.960 (one seed, 50 episodes) | 0.820 | +0.140 | 27.03 s | 33.7 s | Preliminary only; one seed is not a five-seed result. |
| Roundabout-C | not completed | 0.760 | - | not completed | 56.6 s | No conclusion is permitted. |

The paper reports one 50-episode result per method after selecting a policy by
training success.  The current table averages five independently trained
policies where available.  These are different statistical grains.  In
particular, a few percentage points at 50 episodes are not, by themselves,
strong evidence of a real improvement.

## Confirmed causes

### 1. Episode-limit mismatch creates a false Double Merging gain

Severity: **critical**. Confidence: **directly demonstrated**.

Paper Table VI specifies 400 maximum timesteps for Double Merging.  The released
`tools/test.py` instead uses 600, and v12 intentionally followed that released
source value.  Reclassifying the already recorded successful trajectories at
the paper's 400-step boundary gives:

| Seed | Reported success at 600 | Success completed by paper step 400 | Successful episodes after step 400 |
|---:|---:|---:|---:|
| 0 | 1.00 | 0.10 | 45/50 |
| 1 | 0.94 | 0.18 | 38/50 |
| 2 | 1.00 | 0.00 | 50/50 |
| 3 | 0.98 | 0.00 | 49/50 |
| 4 | 1.00 | 0.00 | 50/50 |

Thus the apparent 98.4% mean success is not comparable with the paper's 96%.
Most of those episodes would already have ended as stagnation at the paper
boundary.  This also explains why current completion time is much worse than
the paper despite the apparently higher success rate.

The same risk exists for Roundabout-C: paper Table VI gives 800 steps, while
the released test code and v12 use 1000.  No Roundabout-C result should be
compared before both limits are reported.

### 2. Direct SUMO control removes the original ego dynamics bottleneck

Severity: **critical for roundabouts**. Confidence: **direct source and rollout
evidence**.

The released SMARTS action is handled by `LaneFollowingController` on an
`AckermannChassis`.  It computes lane waypoints, curvature, PID-like throttle
and braking, filtered steering, and then applies physical throttle/brake/steer.
On curved roads it clamps desired speed to 6.94 m/s or 5.56 m/s.

The migrated SUMO environment currently calls TraCI `setSpeed(target_speed)`
on every raw step and disables SUMO speed/right-of-way and lane-change safety
checks with `speedMode=0` and `laneChangeMode=0`.  This correctly mirrors the
SUMO *shadow vehicle* settings used for a SMARTS externally controlled actor,
but it does not recreate the Bullet/Ackermann controller that generated that
actor's physical motion.

A deterministic diagnostic rollout of the completed Roundabout-A seed-0 model
gave:

- successful completion in 112 raw steps (11.2 s);
- mean observed ego speed 9.843 m/s;
- 95th percentile speed 9.971 m/s;
- maximum speed 9.975 m/s;
- fraction of sampled speeds above 6.94 m/s: 1.00.

The released route is approximately 91.66 m after its fixed 1 m departure
offset.  The current five-seed completion mean therefore corresponds to about
7.13 m/s over the whole task, while the paper's 24.5 s corresponds to about
3.74 m/s.  This is exactly the direction expected when the original
curve-aware Ackermann controller is bypassed.  It explains both the unusually
short Roundabout-A time and part of the success gain.

This effect is scenario-dependent.  Double Merging is not made faster: its
current 45.72 s is much slower than the paper's 28.6 s.  The diagnosis is
therefore not the vague claim that "SUMO is always easier"; it is specifically
that the current ego-control transfer is much more ideal on curved routes,
while the learned Double Merging policy is slow.

### 3. Test traffic is parameter-overlapping with training traffic

Severity: **high**. Confidence: **confirmed protocol difference**.

Every newly constructed training or evaluation environment starts the same
SMARTS-compatible rolled traffic-file cycle at episode zero.  Evaluation uses
new SUMO seeds, so episodes are not bit-for-bit duplicates, but it reuses the
same vehicle-type, flow, route, imperfection, impatience, and cooperation
parameter files seen repeatedly during training.

The paper states that testing keeps the road networks but changes randomness
and random seeds for the Table VI parameters.  The corresponding held-out test
traffic assets or their generation seed are not included in the release.
Consequently, the current evaluation is in-distribution at the traffic-file
parameter level, while the paper claims a changed test setting.

Observed evaluation coverage confirms the limited pool:

| Scenario | Unique traffic XMLs in each 50-episode test | Repeats within test | Same XML order for every trained seed |
|---|---:|---:|---|
| Left Turn | 50 of 59 available | 0 | yes |
| Double Merging | 50 of 118 available | 0 | yes |
| Roundabout-A | 30 of 30 available | 20 | yes |
| Roundabout-B | 40 of 40 available | 10 | yes |
| Roundabout-C | 15 available | at least 35 in a 50-episode test | expected by the same cycle |

This is not a duplicate-row error: SUMO's per-episode seed still changes random
draws.  It is a train/test parameter-overlap problem and should be corrected by
a frozen held-out traffic split or by independently regenerated test traffic.

### 4. The paper says random ego starts; the released source fixes them

Severity: **high for generalization, moderate for the observed delta**.
Confidence: **confirmed source/paper conflict**.

The paper says the starting point is assigned randomly along the starting
route.  The released scenario missions use fixed lane/position values, for
example 40 m in Left Turn, 10 m in Double Merging, and 1 m in all three
roundabouts.  v12 follows the released source.  A fixed start reduces state
coverage and allows repeated specialization to the same approach geometry.
The unreleased start distribution cannot be reconstructed exactly, so a future
paper-comparable profile must state its chosen distribution rather than claim
literal equivalence.

## Secondary contributors and countervailing differences

### Source-first hyperparameters differ from the paper

The current complete method follows released source defaults where the paper
and source conflict:

- entropy temperature starts at 0.2 in source, versus 1 in paper Table VII;
- source uses NAdam for actor/critic/representation optimization, while the
  paper says Adam;
- source global predictive horizon defaults to 1, while the paper gives 3.

Lower initial entropy and NAdam can plausibly improve short-budget convergence
in the fixed SUMO tasks, but their contribution has not yet been isolated by a
controlled ablation.  The horizon-1 difference should normally weaken the
proposed auxiliary objective, so it cannot be used as a blanket explanation
for stronger numbers.

### Policy selection and result aggregation do not match

The paper says it tests the policy with the highest training success.  The
release does not provide a runnable best-success saver, and v12 tests the latest
periodic policy for every seed.  The paper presents one 50-episode percentage;
v12 reports every seed and then averages them.  These differences affect the
comparison, but latest-policy and all-seed averaging would normally make v12
more conservative, not explain the large Roundabout-A speed gain.

### Simulator and numerical backends are not identical

The released traffic XMLs were generated by SUMO 1.10.0 and originally run
through SMARTS 0.4.x with Bullet ego dynamics.  The migration runs direct SUMO
1.25.0.  Junction behavior, lane changes, numerical integration, PyTorch versus
TensorFlow initialization/RNG order, and optimizer kernels cannot be bitwise
equivalent.  These are real residual sources of variance, but they are weaker
explanations than the measured time-limit and controller differences above.

## Required correction before SAC/PPO comparison

SAC and PPO must be trained under the same corrected SUMO contract; otherwise
a baseline table would preserve the same unfair gain.  The baseline protocol
will therefore:

1. report a paper-limit evaluation (400 for Double Merging and 800 for
   Roundabout-C) separately from the released-source-limit evaluation;
2. add a curve-aware, acceleration-limited SUMO proxy for the released SMARTS
   Ackermann lane-following controller, while retaining the direct-control
   profile for traceability;
3. freeze non-overlapping training and evaluation traffic-file sets and label
   this as a reconstruction because the authors did not release their exact
   held-out test generation;
4. use the same environment profile, five seeds, 100,000 raw simulator steps,
   and 50 evaluation episodes for Proposed, SAC, and PPO;
5. report both latest-source checkpoint and paper-style best-training-success
   selection when the latter can be captured prospectively;
6. keep every paper/source conflict visible instead of silently mixing the two
   protocols.

Only the corrected profile is suitable for judging whether the complete method
matches the paper.  The existing v12 artifacts remain valid evidence for the
released-source/direct-SUMO profile and must not be deleted or relabelled.
