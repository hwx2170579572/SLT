"""Finite tabular sanity check for a proposed critic, NOT a driving experiment.

No project imports, checkpoints, simulator, training, or network access.
First terminal event is success/collision/timeout; time index j=1 means
terminal reward on the current transition. Entropy starts at the next action
as in the soft Q Bellman equation. The environment and numbers below are
synthetic and must never be reported as empirical driving results.
"""
import json
import math


def evaluate(policy, horizon=7, gamma=0.99, alpha=0.1):
    # (probability, next transient state or None, terminal event or None)
    dynamics = {
        (0, 0): [(0.9, 0, None), (0.1, None, 1)],
        (0, 1): [(0.65, 1, None), (0.25, None, 1), (0.10, None, 0)],
        (1, 0): [(0.65, 1, None), (0.25, None, 1), (0.10, None, 0)],
        (1, 1): [(0.75, None, 0), (0.20, None, 1), (0.05, 0, None)],
    }
    weights = (10.0, -10.0, -5.0)
    p, q_dense, q_direct = {}, {}, {}
    max_error, max_mass_error = 0.0, 0.0
    for remaining in range(1, horizon + 1):
        for state in range(2):
            for action in range(2):
                key = (remaining, state, action)
                law = [[0.0] * horizon for _ in weights]
                dense = -0.01 + 0.05 * action
                qd, qs = dense, dense
                for prob, nxt, event in dynamics[state, action]:
                    if event is not None or remaining == 1:
                        terminal = event if event is not None else 2
                        law[terminal][0] += prob
                        qs += prob * weights[terminal]
                        continue
                    for next_action, pa in enumerate(policy[nxt]):
                        next_key = (remaining - 1, nxt, next_action)
                        entropy = -alpha * math.log(pa)
                        qd += prob * pa * gamma * (q_dense[next_key] + entropy)
                        qs += prob * pa * gamma * (q_direct[next_key] + entropy)
                        for e in range(len(weights)):
                            for j in range(horizon - 1):
                                law[e][j + 1] += prob * pa * p[next_key][e][j]
                recovered = qd + sum(
                    weights[e] * gamma**j * law[e][j]
                    for e in range(len(weights)) for j in range(horizon)
                )
                max_error = max(max_error, abs(qs - recovered))
                max_mass_error = max(max_mass_error, abs(sum(map(sum, law)) - 1.0))
                p[key], q_dense[key], q_direct[key] = law, qd, qs
    return p, q_dense, q_direct, max_error, max_mass_error


def main():
    behavior = ((0.8, 0.2), (0.8, 0.2))
    target = ((0.2, 0.8), (0.2, 0.8))
    pb, _, qb, eb, mb = evaluate(behavior)
    pt, _, qt, et, mt = evaluate(target)
    # Same state, initial action, dynamics and deadline; only continuation differs.
    key = (7, 0, 1)
    old_event_probs = list(map(sum, pb[key]))
    new_event_probs = list(map(sum, pt[key]))
    continuation_gap = sum(abs(x - y) for x, y in zip(old_event_probs, new_event_probs))
    assert max(eb, et, mb, mt) < 1e-12
    assert continuation_gap > 0.1
    # Sum of separately clipped components need not equal clipping full critics.
    componentwise_min = min(10.0, 5.0) + min(-4.0, 0.0)
    whole_critic_min = min(10.0 - 4.0, 5.0 + 0.0)
    assert componentwise_min != whole_critic_min
    print(json.dumps({
        "scope": "synthetic_exact_dynamic_programming_only",
        "checks": {
            "soft_q_decomposition_max_abs_error": max(eb, et),
            "event_probability_mass_max_abs_error": max(mb, mt),
            "same_initial_action_continuation_policy_event_L1_gap": continuation_gap,
            "componentwise_min_is_not_whole_critic_min": True,
        },
        "event_order": ["success", "collision", "timeout"],
        "behavior_continuation_event_probabilities": old_event_probs,
        "target_continuation_event_probabilities": new_event_probs,
        "behavior_soft_q": qb[key],
        "target_soft_q": qt[key],
        "limitations": [
            "not a convergence proof for neural SAC",
            "not evidence of better sample efficiency or driving success",
            "not an off-policy coverage guarantee",
            "finite horizon and fully observed synthetic state only",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
