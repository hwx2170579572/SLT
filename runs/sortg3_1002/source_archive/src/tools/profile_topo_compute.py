"""Profile the topology extractor on a real SUMO observation batch.

This is an engineering diagnostic, not an experiment metric.  It loads an
actual checkpoint/environment pair, expands one valid observation to the
training batch size, and reports operator-level CPU/CUDA time for the feature
extractor forward and backward paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch import SceneRepresentationSAC, TopoTemporalGraphExtractor
from envs.sumo.paper_env import PaperSumoSceneEnv


def _event_row(event: Any) -> dict[str, float | int | str]:
    cuda_total = float(getattr(event, "self_cuda_time_total", 0.0))
    if cuda_total == 0.0:
        cuda_total = float(getattr(event, "self_device_time_total", 0.0))
    return {
        "operator": str(event.key),
        "calls": int(event.count),
        "self_cpu_ms": float(event.self_cpu_time_total) / 1000.0,
        "self_cuda_ms": cuda_total / 1000.0,
    }


def _profile(
    extractor: TopoTemporalGraphExtractor,
    observations: dict[str, torch.Tensor],
    *,
    backward: bool,
    iterations: int,
) -> dict[str, object]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if observations["trajectory"].is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    extractor.zero_grad(set_to_none=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        for _ in range(iterations):
            output = extractor(observations)
            if backward:
                output.square().mean().backward()
                extractor.zero_grad(set_to_none=True)
    if observations["trajectory"].is_cuda:
        torch.cuda.synchronize(observations["trajectory"].device)
    events = [_event_row(event) for event in profiler.key_averages()]
    sort_key = "self_cuda_ms" if observations["trajectory"].is_cuda else "self_cpu_ms"
    events.sort(key=lambda row: float(row[sort_key]), reverse=True)
    return {
        "backward": backward,
        "iterations": iterations,
        "top_operators": events[:25],
        "summed_self_cpu_ms": float(sum(float(row["self_cpu_ms"]) for row in events)),
        "summed_self_cuda_ms": float(sum(float(row["self_cuda_ms"]) for row in events)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    env = PaperSumoSceneEnv(
        scenario=args.scenario,
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="evaluation",
        episode_limit_profile="source",
    )
    try:
        model = SceneRepresentationSAC.load(args.model.resolve(), env=env, device=args.device)
        extractor = model.critic.features_extractor
        if not isinstance(extractor, TopoTemporalGraphExtractor):
            raise TypeError("Checkpoint does not contain TopoTemporalGraphExtractor")
        observation, _ = env.reset(seed=args.seed)
        tensor_observation, _ = model.policy.obs_to_tensor(observation)
        batch = {
            key: value.repeat((args.batch_size,) + (1,) * (value.ndim - 1))
            for key, value in tensor_observation.items()
        }
        extractor.train(True)
        extractor.set_source_augmentation(False)
        for _ in range(args.warmup):
            extractor.zero_grad(set_to_none=True)
            extractor(batch).square().mean().backward()
        if batch["trajectory"].is_cuda:
            torch.cuda.synchronize(batch["trajectory"].device)
        result = {
            "contract_version": 1,
            "engineering_diagnostic_only": True,
            "checkpoint": str(args.model.resolve()),
            "scenario": args.scenario,
            "device": str(batch["trajectory"].device),
            "batch_size": args.batch_size,
            "forward": _profile(
                extractor, batch, backward=False, iterations=args.iterations
            ),
            "forward_backward": _profile(
                extractor, batch, backward=True, iterations=args.iterations
            ),
        }
        return result
    finally:
        env.close()


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenario", default="left_turn")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("batch size/iterations must be positive and warmup non-negative")
    result = run(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
