# v4.11 — Proper-Calibrated Ranked Risk Model (PRCR)

## Decision

v4.10 is retained unchanged as a failed, fully evidenced iteration. v4.11 is a new model version. It does not add a kinematic safety projection, traffic rule, veto, shield, confidence gate, or action rewrite.

The environment may still mask a lane action when the adjacent driving lane does not physically exist. That mask defines the valid action space. It does not inspect traffic risk and is not a safety policy.

## What the real v4.10 evidence says

- D1 and D2 evaluated the same ordered 20 Cross traffic files, but their outcomes agreed on only 9/20 files. Ten D1 collisions fell on files that did not collide in D2, while D2's only collision occurred on a D1 success file.
- Replay collision-label prevalence was nearly equal (0.035958 versus 0.037993). A simple positive-label-count explanation is unsupported.
- The train-only selector chose `exact_final` in both cells and reduced calibration collision rate relative to `highest_training_success`; selector error is unsupported.
- D1 collision value ranks imminent-10 collision with AUROC 0.6374. The signal exists but is weak at the horizon where the policy must act.
- In D1, 48/50 final-five collision decisions used a learned non-keep lane command, versus 0/50 in matched success tails. This localizes the defect to learned action-conditioned transition risk. It does not justify a lane-change rule.
- Shared representation and proposal statistics differ materially across seeds, including merge-attention mass, effective lane count, lane entropy, and proposal spread.

The complete source-backed analysis is in `results_topo_v4_10_dev/development/attribution/d1_d2_cross_instability/attribution.json`.

## Model change

Let each twin collision critic emit a logit \(z_k(s,a)\) and probability \(p_k=\sigma(z_k)\). The v4.10 TD collision-return target \(y\in[0,1]\) is retained exactly, but v4.11 replaces probability MSE with the proper soft Bernoulli negative log-likelihood:

\[
L_{\mathrm{BCE}} = -y\log\sigma(z_k) -(1-y)\log(1-\sigma(z_k)).
\]

For all replay pairs in a minibatch, v4.11 also learns continuous risk ordering:

\[
L_{\mathrm{rank}} =
\frac{\sum_{i,j}|y_i-y_j|\,\operatorname{softplus}[-\operatorname{sign}(y_i-y_j)(z_i-z_j)]}
{\sum_{i,j}|y_i-y_j|}.
\]

There is no hard positive/negative threshold and no fixed ranking margin. Pairs with identical soft targets contribute zero weight.

A low-weight temporal-ensemble term aligns each online collision probability with the detached EMA target critic on the same replay observation and action. This is a learned representation stabilizer; it does not modify inference.

The frozen coefficients are:

- pairwise ranking: 0.25;
- EMA risk consistency: 0.05;
- collision risk in actor and deterministic score: 1.0 (unchanged);
- all v4.10 proposal, support, component-prior, and twin-uncertainty coefficients unchanged.

## Inference equation and projection boundary

Inference is unchanged from v4.10:

1. the actor emits three learned speed-component means for each lane command;
2. the physical lane-existence mask removes only impossible lane commands;
3. learned reward and collision target critics score every feasible learned proposal;
4. `torch.argmax` selects the maximum learned score;
5. the selected lane/speed proposal is applied exactly.

No step computes TTC, headway, a geometry unsafe label, a rule threshold, or an external projected action. No diagnostic speed grid is deployed.

## Evidence gates

Engineering must prove finite loss/gradients, active continuous ranking pairs, detached EMA teachers, disjoint optimizer ownership, exact model-score argmax, exact action equality, and zero rule/projection indicators on a real SUMO smoke run.

Fresh D1–D4 development must all pass the locked v4.10 outcome guards. Every job is attempted even if another fails, then one aggregate decision is written. Only a complete primary pass permits two preregistered ablations; only development plus ablation completion permits the fresh 12-job promotion. Formal test data remains locked until promotion passes.

## Causal restraint

This iteration tests whether a better learned collision-return objective and representation stabilizer repair Cross instability. A pass supports that package, not any individual term. The preregistered ablations separate soft BCE, ranking, and consistency only after the primary development gate passes.
