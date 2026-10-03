"""One bounded, fixed-action environment contract check; no model training.

The check is a separate artifact and is never a policy-performance episode.
"""
import argparse
from pathlib import Path
import numpy as np

from scene_event.env import make_scene_env
from scene_event.protocol import ExperimentConfig
from scene_event.provenance import RunArtifacts, save_json
from scene_event.trainer import EpisodeAccount


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    config = ExperimentConfig(device="cpu", raw_steps=600, eval_episodes=0)
    artifacts = RunArtifacts(args.run_root, config, "deadline_contract")
    env = None
    try:
        env = make_scene_env(config, artifacts.root / "environment", evaluation=True)
        observation, _ = env.reset(seed=10000)
        artifacts.snapshot(training_started=False)
        account = EpisodeAccount("deadline_contract", 0, 10000)
        trace = []
        initial_tick = int(env.last_collection["raw_tick"])
        initial_remaining = float(observation["remaining_time_s"][0])
        if abs(initial_remaining - 60.0) > 1e-5:
            raise AssertionError("Warm-up consumed the ego task deadline")
        for _ in range(201):
            if not env.observation_space.contains(observation):
                raise AssertionError("Real observation violates declared schema/space")
            observation, reward, terminated, truncated, info = env.step(np.asarray([-1.0, 0.0], dtype=np.float32))
            account.add(float(reward), info)
            tick = int(env.last_collection["raw_tick"])
            remaining = float(observation["remaining_time_s"][0])
            if tick - initial_tick != account.raw_steps:
                raise AssertionError("Collector tick axis disagrees with executed raw steps")
            if abs(remaining - max(0.0, 60.0 - 0.1 * account.raw_steps)) > 2e-5:
                raise AssertionError("Remaining time disagrees with physical control time")
            trace.append({"raw": account.raw_steps, "collector_tick": tick, "remaining_seconds": remaining})
            if terminated or truncated:
                break
        if not (terminated and not truncated and info.get("max_time") and account.raw_steps == 600):
            raise AssertionError(f"Expected true terminal deadline at 600 raw ticks; got {info}")
        if not env.observation_space.contains(observation) or float(observation["remaining_time_s"][0]) != 0.0:
            raise AssertionError("Final deadline observation must be valid with zero remaining time")
        if info.get("TimeLimit.truncated", False) or info.get("source_terminal_for_bootstrap") is not True:
            raise AssertionError("Terminal metadata disagrees with finite-horizon protocol")
        result = {"passed": True, "kind": "deadline_contract", "protocol": config.protocol,
                  "training_started": False, "additional_policy_evaluation_episodes": 0,
                  "contract_test_episodes": 1, "trace": trace,
                  **account.result(terminated, truncated, info, True)}
        save_json(artifacts.root / "result.json", result)
        artifacts.status("completed", raw_steps=account.raw_steps, decision_steps=account.decisions,
                         training_started=False)
        print("Deadline contract passed: 600 raw ticks, zero remaining time, terminal/no bootstrap.")
    except BaseException as exc:
        artifacts.status("failed", error=repr(exc), training_started=False)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
