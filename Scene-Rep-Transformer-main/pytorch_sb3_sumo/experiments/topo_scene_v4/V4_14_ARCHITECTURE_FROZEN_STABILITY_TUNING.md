# v4.14 architecture-frozen stability tuning

Status: design draft after the v4.13 promotion audit; no v4.14 scientific run
has been launched and no formal/test data have been accessed.

## Phase decision

v4.13 is the architecture candidate to freeze.  v4.14 is not a new network or
loss family: it is a versioned tuning-only successor whose purpose is to reduce
Cross training-seed and late-checkpoint instability while retaining the
Roundabout-medium and CARLA gains.

The triggering evidence is one failing promotion cell, Cross seed 20 at
Success/Collision 0.70/0.2667, against a Cross seed-21 result of 1.0/0.0 and
development seeds 10/11 at 0.85/0.15 and 0.95/0.05.  Seed 20's exact-final
checkpoint also collapses toward timeout on train-only calibration, while its
route/action integrity remains intact.  This is an optimization-stability
hypothesis, not a justification for another representation or decoder module.

## Frozen method surface

The following must remain identical to v4.13 throughout tuning:

- topology parser, MERGE relation, route/heading query, masks, slots, encoder,
  and Graph-SLT topology;
- masked categorical lane intent plus conditional learned speed mixture;
- shared learned risk encoder and online/target twin reward and collision
  critics;
- learned target reward-risk-uncertainty-support score and target-only decoder;
- deployed lane head, gradient-isolated tempered replay support, and all loss
  definitions;
- optimizer ownership and update order;
- checkpoint candidates and train-only checkpoint selection;
- reward, observation/action space, traffic split, action repeat, and terminal
  semantics;
- absence of kinematic projection, TTC/headway thresholds, geometry rules,
  traffic-risk masks, vetoes, confidence gates, shields, fallbacks, semantic
  tie overrides, and post-decoder rewrites.

Any change to this surface must be a separately justified structural version
and cannot be mixed into v4.14 tuning evidence.

## Freeze-verification backlog

Before calling the architecture unconditionally frozen, use validation data
only to close two evidence gaps with existing method components:

1. a clean return-horizon comparison including 1, 8, and 16 steps on matched
   Cross seeds; this is a scale/horizon ablation, not a network change;
2. an isolated route-bias removal ablation that leaves MERGE, compatible top-k,
   reverse mask, and all other v4.13 components unchanged.

Continuous-vs-Hybrid, MERGE, balance, and decoder evidence already exists and
should not be rerun merely to search for a more favorable number.  The two
backlog diagnostics may reopen architecture only if they expose a reproducible
mechanism contradiction across matched seeds; ordinary variance does not.

## Tuning order and scope

The tuning contract must be sealed before execution.  It should use a small,
factor-at-a-time or compact factorial design rather than an unrestricted grid:

1. training correctness and exact update/normalization audit, with no parameter
   selection;
2. RL stability: learning rate and return horizon/discount;
3. if the instability remains, warm-up/batch-scale stability;
4. only then, existing representation/risk/support loss coefficients;
5. learned deployment-score coefficients last.

Only 1–2 factors may change in a round.  Every round must contain the v4.13
default as a matched control, use identical training/evaluation seed blocks per
configuration, report every attempted job, and aggregate after all jobs finish.
The selection target is a stable region across settings and seeds, not the
single highest run.

## Development and promotion gates

- Primary diagnostic scenario: Cross, because it contains the only v4.13
  promotion failure and the largest candidate seed variance.
- Roundabout-medium and CARLA remain noncompensatory retention guards.
- Required outcomes: Success, Collision, Off-route, Timeout, Return, completion
  time/steps, and seed dispersion.
- Required mechanisms: collision-label exposure, reward/collision twin
  disagreement, policy expected risk, Q/score margins, entropy, support NLL,
  route/MERGE attention, feasible/exact action rates, and checkpoint drift.
- No candidate can pass through an aggregate gain that hides a failed per-seed
  safety/Success guard.
- A fresh paired promotion is mandatory after tuning.  Formal testing remains
  locked until that complete gate passes.

## Reopening rule

Reopen structural development only if matched evidence shows a persistent
directional mechanism failure—such as route information becoming unreadable,
Hybrid no longer outperforming its continuous ablation, target scores failing
to distinguish feasible lane intents, or a scenario-specific topology relation
being systematically absent.  Seed-specific late collapse, sparse collision
labels, and diffuse but nonzero optimization margins remain tuning problems.

