"""Serial CPU telemetry collection from frozen actors, with episode-level resume."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

from .common import cpu_environment
RESOURCES = cpu_environment()
import numpy as np
import torch

from .common import ROOT, OUTPUT, configure_torch, metadata, load_json, write_json, describe_file, sha256
from .models import model_path, load_model, filter_observation, tensor_digest

BEHAVIORS = ("mst_slt", "v4_8_lr_half", "late_decay")
ENCODERS = (*BEHAVIORS, "batch64")
PROTOCOL = ROOT / "experiments/independent_v2_five_methods_six_scenarios_100ep_v1/protocol.json"


def make_environment(scene, namespace):
    from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory
    manifest = load_json(ROOT / "r48s1/manifest.json")
    source = manifest["sources"][f"{scene}__lr_half"]
    args = load_json(ROOT / source["arguments"])["inherited_environment_arguments"].copy()
    args.update(seed=0, gui=False, evaluation_split="validation")
    density = load_json(PROTOCOL)["scenarios"][scene]
    return _make_environment_factory(adapter="v4_8", density=density,
        overlay_root=OUTPUT / "ov" / namespace)(argparse.Namespace(**args), evaluation=True)


def protocol(smoke=False):
    return dict(contract="v48-mechanism-control-trajectories/v2", smoke=smoke,
        scenes=["cross", "carla"], behaviors=list(BEHAVIORS), encoders=list(ENCODERS),
        stable_behavior_choice="late_decay is the first predeclared candidate, not a selected winner",
        episodes_per_behavior=1 if smoke else 40, seed_start=930300 if smoke else 530000,
        split_seed=73, future_horizons_decisions=[1, 4, 16, 32], sumo_raw_step_seconds=.1,
        evaluation="development diagnostic; common V4 environment; final deterministic actors",
        behavior_pairing="same traffic seed and template index across three behavior policies",
        no_training=True, no_cuda=True, max_parallel_sumo=1,
        group_unit="traffic seed across ALL behavior policies and action branches",
        code={str(Path(p).relative_to(ROOT)): sha256(Path(p)) for p in
              [__file__, ROOT / "tools/v48_mechanism_v2/models.py", ROOT / "tools/v48_stability_v1/driving.py"]},
        checkpoints={f"{s}__{m}": describe_file(model_path(s,m)) for s in ("cross", "carla") for m in ENCODERS},
        environment_protocol=describe_file(PROTOCOL))


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seal(path, value):
    if path.exists():
        if load_json(path) != value:
            raise ValueError(f"Immutable diagnostic identity changed: {path}")
    else:
        write_json(path, value)


def initialize(smoke=False):
    audit = load_json(OUTPUT / "stage0/audit.json")
    assert audit["historical_diagnostics_gate"] == "PASS_WITH_DECLARED_LIMITATIONS"
    output = OUTPUT / ("smoke/control" if smoke else "stage2/control")
    value = protocol(smoke)
    seal(output / "protocol.json", value)
    return output, value


def collect_episode(env, wrapped, model, episode_index, seed):
    from tools.run_latent_probes import _transition_targets
    from algos.sb3_torch.evaluation import source_evaluation_augmentation
    np.random.seed(seed + 600000)
    torch.manual_seed(seed + 600000)
    env._traffic_episode_index, env._traffic_roll = episode_index, None
    observation, _ = wrapped.reset(seed=seed)
    traffic = wrapped.traffic_identity()
    assert traffic["seed"] == seed
    rows, actions, next_rows, scalars = [], [], [], []
    step = raw_step = 0
    with source_evaluation_augmentation(model):
        while True:
            before = wrapped.initial if not wrapped.samples else wrapped.samples[-1]
            connection, ego = env._connection, env.specification.ego_id
            route_index = int(connection.vehicle.getRouteIndex(ego))
            action, _ = model.predict(filter_observation(model, observation), deterministic=True)
            start_sample = len(wrapped.samples)
            next_observation, reward, terminated, truncated, info = wrapped.step(action)
            executed = int(info["raw_steps_executed"])
            assert executed == len(wrapped.samples) - start_sample
            samples = [s for s in wrapped.samples[start_sample:] if s is not None]
            after = samples[-1] if samples else before
            targets = _transition_targets(observation, next_observation)
            rows.append({k: np.array(v, copy=True) for k,v in observation.items()})
            next_rows.append({k: np.array(v, copy=True) for k,v in next_observation.items()})
            actions.append(np.array(action, copy=True))
            scalars.append(dict(decision_index=step, raw_start=raw_step, raw_executed=executed,
                reward=float(reward), terminated=terminated, truncated=truncated,
                collision=bool(info.get("collision", False)), timeout=bool(info.get("max_time", False)),
                success=bool(info.get("is_success", False)), off_route=bool(info.get("off_route", False)),
                route_index=route_index, route_intent=int(info.get("pre_action_route_intent", 0)),
                route_intent_valid=bool(info.get("pre_action_route_intent_valid", False)),
                lane_command=int(info["lane_command"]), target_speed=float(info["effective_target_speed"]),
                distance_start=float(before["distance"]), distance_end=float(after["distance"]),
                distance_end_observed=bool(wrapped.samples[-1] is not None),
                speed=float(before["speed"]), acceleration=float(before["acceleration"]), lane=int(before["lane"]),
                following_gap=float(before["leader_gap"]) if before["leader_gap"] is not None else np.nan,
                following_ttc=float(before["ttc"]) if before["ttc"] is not None else np.nan,
                local_label_valid=targets is not None,
                local_dx=float(targets[0][0]) if targets else np.nan,
                local_dy=float(targets[0][1]) if targets else np.nan,
                local_vx=float(targets[0][2]) if targets else np.nan,
                local_vy=float(targets[0][3]) if targets else np.nan,
                observed_distance=float(targets[1]) if targets else np.nan,
                closest_approach_time=float(targets[2]) if targets else np.nan))
            step += 1
            raw_step += executed
            observation = next_observation
            if terminated or truncated:
                break
            if step > 650:
                raise AssertionError("Episode did not terminate under source budget")
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    arrays.update({"next_"+k: np.stack([r[k] for r in next_rows]) for k in next_rows[0]})
    arrays["action"] = np.stack(actions)
    arrays.update({key: np.asarray([row[key] for row in scalars]) for key in scalars[0]})
    record = dict(episode=episode_index, seed=seed, decisions=step, raw_steps=raw_step,
                  episode_return=sum(row["reward"] for row in scalars),
                  **{k: scalars[-1][k] for k in ("success", "collision", "timeout", "off_route")})
    return arrays, dict(record=record, traffic=traffic, driving=wrapped.metrics(raw_step)), dict(
        initial=wrapped.initial, samples=wrapped.samples, actions=wrapped.actions)


def run(smoke=False):
    from tools.v48_stability_v1.driving import DrivingTrace
    configure_torch()
    output, frozen = initialize(smoke)
    fingerprint = identity_hash(frozen)
    started = time.time()
    for scene in frozen["scenes"]:
        for name in BEHAVIORS:
            directory = output / scene / name
            directory.mkdir(parents=True, exist_ok=True)
            env = make_environment(scene, ("smoke_" if smoke else "control_") + scene + "_" + name)
            wrapped = DrivingTrace(env)
            model = load_model(scene, name)
            original = tensor_digest(model)
            try:
                for index in range(frozen["episodes_per_behavior"]):
                    receipt_path = directory / f"e{index:03d}.json"
                    if receipt_path.exists():
                        receipt = load_json(receipt_path)
                        if receipt["protocol_sha256"] != fingerprint or sha256(directory / receipt["data"]) != receipt["data_sha256"]:
                            raise ValueError("Completed episode identity mismatch")
                        continue
                    arrays, receipt, trace = collect_episode(env, wrapped, model, index, frozen["seed_start"] + index)
                    data_path = directory / f"e{index:03d}.npz"
                    with data_path.with_suffix(".tmp").open("wb") as stream:
                        np.savez_compressed(stream, **arrays)
                    data_path.with_suffix(".tmp").replace(data_path)
                    trace_path = directory / f"e{index:03d}.json.gz"
                    with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
                        json.dump(trace, stream, allow_nan=False)
                    receipt.update(protocol_sha256=fingerprint, data=data_path.name, data_sha256=sha256(data_path),
                                   trace=trace_path.name, trace_sha256=sha256(trace_path))
                    seal(receipt_path, receipt)
                    write_json(output / "progress.json", dict(scene=scene, behavior=name, episode=index+1,
                        episodes_per_behavior=frozen["episodes_per_behavior"], elapsed_seconds=time.time()-started))
                    print(f"collected {scene}/{name} {index+1}/{frozen['episodes_per_behavior']}", flush=True)
                assert tensor_digest(model) == original
            finally:
                env.close()
    # Verify physical pairing across every behavior, not just matching seed names.
    pairs = []
    for scene in frozen["scenes"]:
        for index in range(frozen["episodes_per_behavior"]):
            traffic = [load_json(output / scene / n / f"e{index:03d}.json")["traffic"] for n in BEHAVIORS]
            keys = ("seed", "route_sha256", "network_sha256")
            assert all(all(t[k] == traffic[0][k] for k in keys) for t in traffic)
            pairs.append(dict(scene=scene, seed=traffic[0]["seed"], passed=True))
    write_json(output / "complete.json", dict(**metadata(), protocol_sha256=fingerprint,
        episodes=len(pairs)*len(BEHAVIORS), traffic_pairing_verified=True, tensors_unchanged=True,
        resource_policy=RESOURCES, elapsed_seconds=time.time()-started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
