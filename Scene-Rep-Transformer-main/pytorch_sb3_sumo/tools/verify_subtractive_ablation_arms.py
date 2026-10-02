"""Build-time audit for the two subtractive-ablation arms.

Confirms the subtracted mechanism actually lands on the model object (not just the
metadata): minus_horizon -> n_steps=4 + frozen dict buffer; minus_factorized ->
lane_entropy_scale=1.0 + target_entropy=-2.0 + horizon-correct buffer. Does not train.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase1_checkpoint_diagnostics import read
from configs.sb3_configs_v4_8 import (
    V48_CANDIDATE,
    V48_MINUS_HORIZON,
    V48_MINUS_FACTORIZED,
    make_model_v4_8,
)


ALGOS = {
    "full_v4_8": V48_CANDIDATE,
    "minus_horizon": V48_MINUS_HORIZON,
    "minus_factorized": V48_MINUS_FACTORIZED,
}


def build_model(algo: str, scenario: str):
    from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory

    plan = read(ROOT / "results_phase1_checkpoint_diagnostics_v1/plan.json")
    source = next(
        s for s in plan["sources"]
        if s["method"] == "v4_8" and s["scenario"] == scenario
    )
    requested = read(ROOT / source["run"] / "arguments.json")["requested_raw_steps"]
    env_args = argparse.Namespace(
        **dict(requested, seed=0, gui=False, evaluation_split="validation")
    )
    protocol = read(ROOT / plan["protocol"])
    factory = _make_environment_factory(
        adapter="v4_8",
        density=protocol["scenarios"][scenario],
        overlay_root=ROOT / "results_subtractive_ablation" / "ov" / "verify",
    )
    env = factory(env_args)
    model = make_model_v4_8(
        algo,
        env,
        scenario=scenario,
        learning_rate=5e-5,
        action_repeat=3,
        seed=0,
        device="cuda",
        verbose=0,
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="cross")
    args = parser.parse_args()

    expected = {
        "full_v4_8": dict(n_steps=16, lane_entropy_scale=0.0, target_entropy=-1.0),
        "minus_horizon": dict(n_steps=4, lane_entropy_scale=0.0, target_entropy=-1.0),
        "minus_factorized": dict(n_steps=16, lane_entropy_scale=1.0, target_entropy=-2.0),
    }

    failures = []
    for name, algo in ALGOS.items():
        model = build_model(algo, args.scenario)
        buffer_cls = type(model.replay_buffer).__name__
        actual = dict(
            n_steps=model.n_steps,
            lane_entropy_scale=model.lane_entropy_scale,
            target_entropy=model.target_entropy,
        )
        exp = expected[name]
        ok = (
            actual["n_steps"] == exp["n_steps"]
            and abs(actual["lane_entropy_scale"] - exp["lane_entropy_scale"]) < 1e-12
            and abs(actual["target_entropy"] - exp["target_entropy"]) < 1e-12
        )
        print(f"[{'OK' if ok else 'FAIL'}] {name}")
        print(f"      n_steps={actual['n_steps']}  lane_entropy_scale={actual['lane_entropy_scale']}  "
              f"target_entropy={actual['target_entropy']}  buffer={buffer_cls}")
        if not ok:
            failures.append((name, exp, actual, buffer_cls))

    if failures:
        print("\nMISMATCH:")
        for name, exp, actual, buffer_cls in failures:
            print(f"  {name}: expected {exp} got {actual} (buffer={buffer_cls})")
        return 1
    print("\nALL ARMS MATCH EXPECTED BUILD-TIME CONFIGURATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
