# Stage 5 prediction primitives — engineering status, not a passed method gate

Updated 2026-10-03. This is an additive implementation record. M0/M1 do not import
`scene_event_prediction`; no M2/M3 training or prediction result is recorded here.

## Implemented independently

`scene_event_prediction/progress_process.py` contains:

- `ProgressTransitionHead`: branch context → conditional nonnegative progress
  increments on a fixed physical time/distance grid.
- `ProgressProcess`: forward filtering, factual observation likelihoods, and
  two-time joint region probabilities under that same Markov process.
- `EventDistributionProjector`: vehicle-center entry/clearance thresholds →
  first-passage masses, occupancy marginals, and horizon survival probabilities.

The current default engineering grid is 10 × 0.3 s, 0.5 m spacing, 181 states,
and increments 0…18 bins per step. The last state is an absorbing **overflow
bucket at or beyond 90 m**, not an exact location. These are versioned prototype
settings, not tuned or validated model hyperparameters. The decoder sees fixed
physical scales, so changing spacing/dt changes its coordinate inputs.

For progress state distribution p_t(s) and conditional increments P_t(d|s),
the recurrence is p_(t+1)(s') = sum_(s,d: clip(s+d)=s') p_t(s) P_t(d|s).
An entry threshold e has CDF F_e(t)=P(S_t >= e). Monotonicity makes future
first-entry mass F_e(t)-F_e(t-1), and occupancy between e and c is
P(e <= S_t < c). Both thresholds use one process; there are no independent ETA
heads. The two-time region function retains process dependence, whereas a
product of marginal probabilities can assign mass to impossible reverse order.
This is an exact computation **within the discrete monotone-process assumption**,
not a claim that traffic follows this model or that it is calibrated.

Missing observations integrate over latent states and supply no negative label.
A *known* right-censor fact (observed not to reach a threshold by a given time)
can supply a survival-region emission at that time. The future-label collector
has not yet been connected to construct those emissions. Unknown disappearance
must not be converted into known nonarrival. All-missing sequences report zero
observations and must be excluded from the auxiliary loss denominator.

An impossible route prefix retains log likelihood minus infinity. Its branch
gradient is safely zero, allowing a compatible alternative in a mixture to train
without 0 × infinity contaminating gradients. An entirely impossible scene still
requires explicit out-of-set handling; this primitive does not implement it.

## Verification actually run

`D:\Programs\Anaconda\envs\pytorch\python.exe -m unittest test_scene_event_progress_process -v`

Result: **10 tests passed**. Analytic cases cover binomial progress, probability
mass, absorbing overflow, missing versus known-censored labels, impossible route
prefixes and mixture gradients, current/past events, separate entry/clearance
validity, impossible cross-zone reverse order, batch/route axes, permutation, and
auxiliary/projector gradients. These are pure CPU tensor tests; no simulator
episode, training run, or calibration dataset was consumed.

## Still required for Stage 5

1. Public candidate-path projection and compatible-prefix route labels; explicit
   unknown/out-of-set outcomes and actor-specific observation-loss reasons.
2. Mature factual-label sampling without future-input leakage, and interval/
   censor semantics respecting the 0.3 s sampling resolution.
3. K=1 predictor integration before event encoding. TD receives detached
   predicted distributions; auxiliary loss updates the prediction head and
   shared stem. Shared stem receives a single combined optimizer step.
4. Optimizer/target ownership tests, TD-versus-aux gradients, actual updates,
   prediction drift and normal collection coverage.
5. Episode-held-out calibration against CV/CA, stratum coverage and cost, then
   matched closed-loop comparison. Predictive improvement and useful joint
   dependence are empirical gates before introducing M3 scene modes.

**Stage 5 remains pending.** These primitives neither introduce a registered M2
method nor authorize bypassing the Stage 0–4 environment/geometry acceptance.
