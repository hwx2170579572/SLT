# Bootstrap protocol audit: `intersection_sorted_depart4p0`

This is a source-backed, offline check of the active 4-step replay target. The reproducible synthetic-buffer probe and exact source hashes are in [the companion JSON](bootstrap_protocol_audit_20261003.json); the probe script is [bootstrap_protocol_probe_20261003.py](bootstrap_protocol_probe_20261003.py). It ran on CPU with hand-written transitions only. It did not create an environment, model, optimizer, SUMO process, or training/evaluation run.

## What the current learner computes

For an actual sequence of `k` decision transitions up to four or the first episode boundary, replay returns

`R_k = r_0 + γ r_1 + … + γ^(k−1) r_(k−1)`.

The SAC critic target is

`Y = R_k + (1 − done_mask) γ [min(Q1_target,Q2_target) − α log π(a_next|s_next)]`.

Here `γ=0.99` is the multiplier **for every bootstrap**, regardless of whether the accumulated window contains 1, 2, 3, or 4 decision transitions. A standard k-step bootstrap would instead multiply by `γ^k` (for a full four-step window, `γ^4 = 0.96059601`). This is a confirmed implementation difference. The replay code explicitly comments that using one `γ` preserves the released SAC behavior; that comment establishes local intent, not independent verification of the upstream release source.

Each replay transition is one policy decision, normally three 0.1-second SUMO ticks. Therefore a full four-transition window spans nominally 12 raw ticks or 1.2 seconds, while the replay discount remains per decision. It is not converted to a per-raw-tick discount. When a terminal event interrupts a held action, the final decision may contain fewer than three raw ticks and still counts as one transition. At a run-budget cutoff, an incomplete nonterminal held action is skipped by the buffer.

The hand-built probe produced the same four-step reward `9.8014965` and endpoint observation 4 for the nonterminal, timeout-on-step-4, and true-terminal-on-step-4 cases. Their bootstrap discount was `.99`; the timeout mask was 0, while the true-terminal mask was 1. For shortened `k=1,2,3` tails, rewards and endpoint observations stopped at the actual boundary and did not include a synthetic next-episode reward. Timeout still bootstrapped with `.99`; terminal did not bootstrap. A one-step `DictReplayBuffer` sample has `discounts=None`, so SAC falls back to `.99`.

SAC applies one entropy adjustment at the bootstrap endpoint: `min(Q1,Q2) − α log π(a_next|s_next)`. The n-step replay return itself contains environment rewards only: it does not add per-intermediate entropy terms and it has no intermediate importance-sampling or trace correction. Consequently, this is the project’s replay n-step SAC target, not an exact current-policy soft n-step return. That boundary alone does not show a defect; off-policy n-step replay commonly makes this approximation. The decisive local distinction is the extra bootstrap discount factor.

For scale, using the synthetic return above and a fixed soft endpoint value of `3.1`, a timeout target is `12.8704965` under the current one-`γ` protocol and `12.7793441` under `γ^4`. A true terminal target is `9.8014965` because its mask removes the bootstrap. These are arithmetic probes, not learned Q values or a claim about measured training impact.

## Timeout and task definition

The sorted scenario registry caps an episode at 600 raw ticks; the SUMO configuration and `PaperSumoSceneEnv` command use a 0.1-second step, so the cap is 60 seconds. D1 configures `n_steps=4`, and the base trainer sets `ACTION_REPEAT=3` and `DISCOUNT=0.99`. For the active SMARTS observation contract, `SumoSceneEnv.step` reports this max-time event as `truncated=True`, not `terminated=True`. The n-step buffer treats it as a reward-sequence boundary, removes the timeout flag from its terminal mask, and bootstraps from the boundary transition’s next observation. The shared v2 reward wrapper also assigns timeout reward −5; that penalty is included in `R_k` while the cap remains bootstrap-eligible.

The base observation builder contains trajectory/map and optional LSTM-history fields. A bounded static scan of the SUMO env, SAC code and D1 trainer found no remaining-time field. This fact does **not** decide whether 60 seconds is part of the task or an external collection limit. Gymnasium’s time-limit guidance and Pardo et al. distinguish these cases: if finishing within 60 seconds is intrinsic to the task, the deadline is a termination and remaining time is needed in the state for a Markov finite-horizon formulation; if the cap is external to a continuing task, truncation and bootstrapping are appropriate. Do not conclude that every timeout should be terminal. The research/task contract must choose which case `intersection_sorted_depart4p0` represents.

## How to interpret and verify the discrepancy

- **Behavioral fact:** current source returns one bootstrap `γ`; the 10-test CPU probe checks full windows, shorter tails, timeout versus terminal masks, episode-boundary isolation, a one-step control, and SAC target arithmetic.
- **Protocol judgment:** `γ` versus `γ^k` is not an accidental mismatch inside this implementation: the source contains an explicit released-compatibility comment. The audit has not independently traced that statement to a specific upstream release revision.
- **Task-definition decision:** the 60-second timeout semantics cannot be selected by a replay-buffer unit test. Decide whether the cap is task-intrinsic or externally imposed, then test the corresponding `terminated`/`truncated` flag and, for a finite-horizon task, the remaining-time observation.
- **If adopting standard decision-level n-step semantics:** use `γ^k` for the actual number of included decision transitions, including shortened tails; do not merely replace `.99` with `.99**4`, because boundaries can occur at `k<4`. Keep such a protocol version separate from the existing compatibility runs and use it uniformly for comparisons. This audit did not modify that protocol or evaluate its performance.

The previous [2026-09-30 audit](p02_p03_100k_analysis_20260930.md) and its [original probe JSON](p02_p03_nstep_probe_20260930.json) are retained unchanged; this artifact adds the missing shortened-tail and target-arithmetic checks.

## Source locations

- [`replay_buffer.py`](../../algos/sb3_torch/replay_buffer.py): boundary tail staging at line 113; n-step sample/return logic at 183; released-compatibility comment at 203; one-`γ` bootstrap at 206; discounted rewards at 209; timeout mask at 242.
- [`sac.py`](../../algos/sb3_torch/sac.py): fallback discount at 382; endpoint entropy adjustment at 458; target at 459.
- [`train_intersection_yield_v2.py`](../train_intersection_yield_v2.py): `DISCOUNT=.99` and `ACTION_REPEAT=3` at lines 71–72.
- [`train_intersection_yield_v2_d1.py`](../train_intersection_yield_v2_d1.py): n-step 4 and replay gamma/action-repeat wiring at lines 1032–1037.
- [`paper_scenario_registry.py`](../../envs/sumo/paper_scenario_registry.py): sorted scenario’s 600-raw-tick cap at lines 100–102; [`scenario.sumocfg`](../../envs/sumo/scenarios/intersection_sorted/scenario.sumocfg) and [`paper_env.py`](../../envs/sumo/paper_env.py) set 0.1-second SUMO ticks.
- [`sumo_env.py`](../../envs/sumo/sumo_env.py): max episode steps are raw 0.1-second simulator ticks at lines 382–385; truncation/termination split at 630–636; observation keys at 299–326 and builder at 1966 onward.
- [`reward_shaping_v2.py`](../reward_shaping_v2.py): timeout coefficient −5 and step/progress shaping at lines 58–68 and 85–115.

External theory references: [Gymnasium time-limit handling](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/), [Pardo et al., *Time Limits in Reinforcement Learning*](https://arxiv.org/abs/1712.00378), and [Stable-Baselines3 SAC source documentation](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/sac/sac.html).
