"""Short deterministic SUMO diagnostics-on/off transparency smoke test."""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "fast-developer"))

from envs.sumo.paper_env import PaperSumoSceneEnv  # noqa: E402
from behavior_diagnostics import wrap_behavior_diagnostics  # noqa: E402


OUT = Path(__file__).resolve().parent / "behavior_smoke_20260929"


def make_env():
    return PaperSumoSceneEnv(
        scenario="intersection_sorted",
        traffic_partition="all",
        history_steps=4,
        neighbors=5,
        path_length=5,
        path_spacing=5.0,
        neighbor_radius=120.0,
        action_repeat=3,
        reward_discount=0.99,
        ego_control_profile="direct",
    )


def assert_obs_equal(left, right):
    assert set(left) == set(right), (set(left), set(right))
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def assert_info_equal(left, right):
    # The recorder adds provenance keys only while enabled. Compare simulator
    # and traffic provenance that both paths expose.
    for key in (
        "scenario",
        "simulation_seed",
        "traffic_variant",
        "traffic_variant_seed",
        "source_traffic_cycle_episode",
        "source_traffic_roll",
        "active_vehicles",
        "raw_simulation_steps",
    ):
        if key in left or key in right:
            assert left.get(key) == right.get(key), (key, left.get(key), right.get(key))


def run_episode_pair(on_env, off_env, seed, decisions):
    obs_on, info_on = on_env.reset(seed=seed)
    obs_off, info_off = off_env.reset(seed=seed)
    assert_obs_equal(obs_on, obs_off)
    assert_info_equal(info_on, info_off)
    assert info_on["simulation_seed"] == seed

    raw_ticks = 0
    terminal = False
    for index in range(decisions):
        # Same fixed action sequence on both paths; vary the requested speed
        # slightly while retaining the keep-lane command.
        action = np.asarray([((index % 5) - 2) * 0.1, 0.0], dtype=np.float32)
        out_on = on_env.step(action)
        out_off = off_env.step(action)
        assert_obs_equal(out_on[0], out_off[0])
        assert out_on[1] == out_off[1], (index, out_on[1], out_off[1])
        assert out_on[2:4] == out_off[2:4], (index, out_on[2:4], out_off[2:4])
        for key in ("raw_steps_executed", "undiscounted_reward", "is_success", "collision", "off_route", "max_time"):
            assert out_on[4].get(key) == out_off[4].get(key), (
                index,
                key,
                out_on[4].get(key),
                out_off[4].get(key),
            )
        raw_ticks += int(out_on[4]["raw_steps_executed"])
        if out_on[2] or out_on[3]:
            terminal = True
            break
    return raw_ticks, terminal


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output: {OUT}")
    OUT.mkdir(parents=True)
    base_on = make_env()
    on_env = wrap_behavior_diagnostics(
        base_on,
        output_dir=OUT,
        phase="transparency",
        method="paper_intersection_sorted",
        metadata={"purpose": "diagnostic_on_off_transparency_smoke", "seed": 20260929},
    )
    off_env = make_env()
    total_raw = 0
    try:
        first_raw, first_terminal = run_episode_pair(on_env, off_env, 20260929, 40)
        total_raw += first_raw
        second_raw, _ = run_episode_pair(on_env, off_env, 20260930, 4)
        total_raw += second_raw
    finally:
        on_env.close()
        off_env.close()

    diag_dir = OUT / "diagnostics" / "transparency"
    with gzip.open(diag_dir / "raw_steps.jsonl.gz", "rt", encoding="utf-8") as stream:
        raw_rows = [json.loads(line) for line in stream if line.strip()]
    with gzip.open(diag_dir / "decisions.jsonl.gz", "rt", encoding="utf-8") as stream:
        decision_rows = [json.loads(line) for line in stream if line.strip()]
    episode_rows = [
        json.loads(line)
        for line in (diag_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(episode_rows) == 2, len(episode_rows)
    assert [row["seed"] for row in episode_rows] == [20260929, 20260930]
    assert len(raw_rows) == total_raw, (len(raw_rows), total_raw)
    assert len(decision_rows) >= 40, len(decision_rows)
    assert all(row.get("step_seconds") == 0.1 for row in raw_rows), {
        row.get("step_seconds") for row in raw_rows
    }

    error_count = sum(len(row.get("errors") or []) for row in raw_rows)
    # Validate world-frame velocity direction against measured actor center
    # displacement over consecutive SUMO ticks. Omit stationary/removed actors.
    direction_cosines = []
    previous = {}
    for row in raw_rows:
        now = row.get("sim_time")
        dt = row.get("step_seconds")
        if now is None or dt is None or dt <= 0:
            previous.clear()
            continue
        actors = [row.get("ego"), *row.get("vehicles", [])]
        current = {}
        for actor in actors:
            if not actor or actor.get("state_source") != "current":
                continue
            key = actor["key"]
            current[key] = actor
            old = previous.get(key)
            if old is None:
                continue
            elapsed = now - old["sim_time"]
            if elapsed <= 0:
                continue
            p0, p1 = np.asarray(old["position"], dtype=float), np.asarray(actor["position"], dtype=float)
            v = np.asarray(actor["velocity"], dtype=float)
            displacement = (p1 - p0) / elapsed
            if np.linalg.norm(v) > 0.5 and np.linalg.norm(displacement) > 0.1:
                cosine = float(np.dot(v, displacement) / (np.linalg.norm(v) * np.linalg.norm(displacement)))
                direction_cosines.append(cosine)
        previous = {
            key: {"position": actor["position"], "sim_time": now}
            for key, actor in current.items()
        }
    assert direction_cosines, "No moving actors found for physical-frame check"
    median_cosine = float(np.median(direction_cosines))
    assert median_cosine > 0.8, median_cosine

    result = {
        "status": "passed",
        "scene": "intersection_sorted",
        "reset_seeds": [20260929, 20260930],
        "first_episode_terminal": first_terminal,
        "raw_ticks_total_on_off_matched": total_raw,
        "diagnostic_raw_rows": len(raw_rows),
        "diagnostic_decision_rows": len(decision_rows),
        "diagnostic_episodes": len(episode_rows),
        "step_seconds": sorted({row["step_seconds"] for row in raw_rows}),
        "diagnostic_error_count": error_count,
        "physical_velocity_direction_pairs": len(direction_cosines),
        "physical_velocity_direction_median_cosine": median_cosine,
        "episode_traffic_provenance": [
            {
                key: row.get("reset_info", {}).get(key)
                for key in (
                    "simulation_seed",
                    "requested_reset_seed",
                    "traffic_variant",
                    "traffic_variant_seed",
                    "source_traffic_cycle_episode",
                    "source_traffic_roll",
                )
            }
            for row in episode_rows
        ],
        "output_dir": str(OUT),
    }
    (OUT / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
