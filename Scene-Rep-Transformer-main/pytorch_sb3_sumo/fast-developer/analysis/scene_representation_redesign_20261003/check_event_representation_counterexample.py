"""Synthetic representation counterexample; no SUMO, training, or project results."""

import json


def summarize(worlds):
    assert abs(sum(p for _, p in worlds) - 1.0) < 1e-12
    return {
        "marginal_free_zone_1": sum(p * x[0] for x, p in worlds),
        "marginal_free_zone_2": sum(p * x[1] for x, p in worlds),
        "joint_both_free": sum(p * x[0] * x[1] for x, p in worlds),
    }


def main():
    correlated = [((0, 0), 0.5), ((1, 1), 0.5)]
    anticorrelated = [((0, 1), 0.5), ((1, 0), 0.5)]
    a, b = summarize(correlated), summarize(anticorrelated)
    for key in ("marginal_free_zone_1", "marginal_free_zone_2"):
        assert a[key] == b[key] == 0.5
    assert a["joint_both_free"] == 0.5
    assert b["joint_both_free"] == 0.0
    assert summarize(list(reversed(correlated))) == a
    print(json.dumps({"kind": "synthetic_exact_counterexample", "A": a, "B": b,
                      "claim": "Marginals alone cannot determine joint passage availability.",
                      "limitation": "No claim about actual traffic frequency or a specific GNN's limitations."},
                     indent=2))


if __name__ == "__main__":
    main()
