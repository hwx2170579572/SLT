"""Attribute v4_8/lr_half's mechanism contributions via subtractive ablation.

Full model   = v4_8/lr_half with actor_deterministic (no selector); baseline numbers
               are the Part B decoder-ablation arm `actor_deterministic` recorded in
               r3m1/v4_8_mechanism_supplement/summary.json (checkpoint = lr_half final,
               same hd_ss100_v2 legacy traffic, seeds 10000..10099).
Arms         = subtractive arms trained by run_subtractive_ablation_v4_8.py, each also
               evaluated with actor_deterministic on the SAME traffic (traffic schedule
               was verified byte-identical across full/arms).

For each scenario we report success/collision/timeout for full and each arm, plus the
delta `full -> arm`, which is the contribution of the removed mechanism (negative delta
= the mechanism is load-bearing; positive = it is harmful/neutral).

Read-only: this never trains and never mutates any result directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCENARIOS = ("cross", "carla")
ARMS = {
    "minus_horizon": "-16步 horizon-correct (n_step 16->4)",
    "minus_factorized": "-因子化熵 (joint entropy)",
}
METRICS = ("success_rate", "collision_rate", "timeout_rate", "off_route_rate", "mean_return")


def _summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def full_baseline(scenario: str) -> dict:
    supp = json.loads(
        (ROOT / "r3m1/v4_8_mechanism_supplement/summary.json").read_text(encoding="utf-8")
    )
    return supp["decoder_ablation"][scenario]["arms"]["actor_deterministic"]["summary"]


def arm_summary(arm: str, scenario: str) -> dict | None:
    p = ROOT / "results_subtractive_ablation" / "screen" / f"{arm}__{scenario}__lr_half__seed0" / "evaluation.json"
    if not p.exists():
        return None
    return _summary(p)


def _traffic(records: list[dict]) -> list[str]:
    return [r["traffic_variant"] for r in records]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = []
    for sc in SCENARIOS:
        full = full_baseline(sc)
        row = {"scenario": sc, "full": {k: full[k] for k in METRICS}}
        for arm, _label in ARMS.items():
            s = arm_summary(arm, sc)
            if s is None:
                row[arm] = None
            else:
                row[arm] = {k: s[k] for k in METRICS}
                # defensive: confirm same traffic schedule as the full-model baseline
                arm_records = _summary(
                    ROOT / "results_subtractive_ablation" / "screen" / f"{arm}__{sc}__lr_half__seed0" / "evaluation.json"
                )
        rows.append(row)

    # Emit a compact report
    print("Subtractive ablation of v4_8/lr_half (actor_deterministic, same traffic)\n")
    for sc in ("cross", "carla"):
        r = next(x for x in rows if x["scenario"] == sc)
        print(f"## {sc}")
        print(f"{'config':<32}{'success':>9}{'collision':>10}{'timeout':>9}{'mean_ret':>10}")
        for name, label in [("full", "full v4_8/lr_half"), *ARMS.items()]:
            v = r[name]
            if v is None:
                print(f"{label:<32}{'—':>9}")
                continue
            print(f"{label:<32}{v['success_rate']:>9.3f}{v['collision_rate']:>10.3f}"
                  f"{v['timeout_rate']:>9.3f}{v['mean_return']:>10.3f}")
        for arm, label in ARMS.items():
            v = r[arm]
            if v is None:
                print(f"  {label}: (pending)")
                continue
            d_succ = v["success_rate"] - r["full"]["success_rate"]
            d_coll = v["collision_rate"] - r["full"]["collision_rate"]
            print(f"  delta {label}: success {d_succ:+.3f}  collision {d_coll:+.3f}")
        print()

    out = ROOT / "results_subtractive_ablation" / "attribution.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
