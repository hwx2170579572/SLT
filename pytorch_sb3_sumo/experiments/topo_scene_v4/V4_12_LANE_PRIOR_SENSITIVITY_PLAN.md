# v4.12 frozen lane-prior sensitivity plan

Status: locked before the `lane_prior_coef=0.10` closed-loop rollout.

Created locally: 2026-09-02 09:37 +08:00.

## Evidence available before the planned rollout

- The v4.12 development gate failed only D3 (`roundabout_medium`): success
  `0.80`, collision `0.20`, against the v4.11 aggregate reference `0.90/0.10`.
- Immutable trace attribution SHA-256:
  `43927d35ccb326878a02872327e4a5d6e69eae0dd78d9fdb0e43f3729712b077`.
- The same-checkpoint, same-seed `lane_prior_coef=0.00` rollout completed at
  `0.60/0.40`; receipt SHA-256:
  `8aea990543d5d13b5d4cd7bf75c807744e6b42a961a3ca87695416baa6d811f7`.
- Offline score sensitivity for `0.10` changes `6.289%` of stored D3 lane
  choices, lowers lane-switch rate from `0.155397` to `0.126837`, and raises
  actor-mode agreement from `0.869638` to `0.932528`. These are ranking
  diagnostics only and make no closed-loop outcome claim.

## Planned intervention

- Source checkpoint: the immutable v4.12 D3 `selected_model.zip`.
- Evaluation: validation split, seeds `163000..163019`, 20 episodes.
- Intervention: set the single global learned-model coefficient
  `policy.lane_prior_coef` from `0.05` to `0.10`.
- Learned tensor state must have identical SHA-256 before and after the change.
- The coefficient is global, not scenario-conditioned.
- No TTC/headway threshold, geometry label, lane veto, confidence gate,
  kinematic projection, shield, semantic tie override, or action rewrite is
  permitted.
- Formal-test data remains locked and untouched.

## Precommitted interpretation

- Consider a higher-prior v4.13 route only if D3 reaches success at least
  `0.85`, collision at most `0.15`, and improves the stored `0.05` result by at
  least `+0.05` success or `-0.05` collision.
- Reject the higher-prior route if any integrity/no-rule check fails, or if the
  outcome threshold above is not met.
- Even if it passes D3, it is only a hypothesis for v4.13. Fresh D1--D4
  training and development gates remain mandatory before promotion.
- The separately running `0.00` D1--D4 matrix is attribution evidence only and
  does not unlock any development, promotion, or formal stage.
