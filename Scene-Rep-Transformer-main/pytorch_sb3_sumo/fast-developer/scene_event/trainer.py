"""Fresh-training/evaluation runner for the independent scene-event protocol."""
from dataclasses import replace
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from .diagnostics import HistoryAudit, JsonLines, ShadowBudget, run_shadow
from .future_targets import ObservedFutureQueue
from .policy import SceneSAC
from .protocol import ExperimentConfig, readonly_observation
from .provenance import RunArtifacts, array_mapping_sha256, save_json, sha256
from .replay import DecisionTransition, NStepAssembler, ObservationReplay


REWARD_COMPONENTS = ("reward_success", "reward_collision", "reward_off_route",
                     "reward_timeout", "reward_step_cost", "reward_progress")


class EpisodeAccount:
    def __init__(self, phase: str, episode: int, seed: int):
        self.phase, self.episode, self.seed = phase, episode, seed
        self.reward = 0.0
        self.raw_return = 0.0
        self.raw_reward_available = True
        self.components = {key: 0.0 for key in REWARD_COMPONENTS}
        self.raw_steps = 0
        self.decisions = 0
        self.max_component_step_error = 0.0

    def add(self, reward: float, info: dict[str, Any]) -> None:
        raw = int(info["raw_steps_executed"])
        if raw < 1:
            raise ValueError("Actual raw steps must be present and positive")
        # GeneralizedRewardShapingWrapper exposes cumulative components. Taking
        # a delta is required: summing those cumulative fields per step is wrong.
        current = {key: float(info[key]) for key in REWARD_COMPONENTS}
        if not math.isfinite(float(reward)) or not all(math.isfinite(v) for v in current.values()):
            raise ValueError("Non-finite environment reward/component")
        delta = sum(current.values()) - sum(self.components.values())
        error = abs(delta - reward)
        self.max_component_step_error = max(self.max_component_step_error, error)
        if error > 2e-5:
            raise ValueError(f"Six-component reward reconciliation failed: {delta} != {reward}")
        self.components = current
        self.reward += float(reward)
        if "undiscounted_reward" in info:
            if not math.isfinite(float(info["undiscounted_reward"])):
                raise ValueError("Non-finite raw environment reward")
            self.raw_return += float(info["undiscounted_reward"])
        else:
            self.raw_reward_available = False
        self.raw_steps += raw
        self.decisions += 1

    def result(self, terminated: bool, truncated: bool, info: dict[str, Any], completed: bool) -> dict[str, Any]:
        success = bool(info.get("is_success", False))
        collision = bool(info.get("collision", False))
        off_route = bool(info.get("off_route", False))
        timeout = bool(info.get("max_time", False))
        if completed and sum((success, collision, off_route, timeout)) != 1:
            raise ValueError("Completed episode must have exactly one mutually exclusive task outcome")
        return {"phase": self.phase, "episode": self.episode, "seed": self.seed,
            "completed": completed, "terminated": bool(terminated), "truncated": bool(truncated),
            "success": success, "collision": collision, "off_route": off_route, "timeout": timeout,
            "raw_steps": self.raw_steps, "decision_steps": self.decisions,
            "environment_step_reward_v2_return": self.reward,
            "raw_environment_return": self.raw_return if self.raw_reward_available else None,
            "raw_return_definition": "sum(info.undiscounted_reward); distinct from shaped decision reward",
            **self.components, "reward_component_error": abs(sum(self.components.values()) - self.reward),
            "max_component_step_error": self.max_component_step_error}


def _encoder(config: ExperimentConfig, public_map: dict[str, np.ndarray]) -> torch.nn.Module:
    from .encoder import SceneEncoder
    return SceneEncoder(method="m0" if config.method == "sac_scene_dualgraph_v1" else "m1",
                        width=config.width, z_dim=config.z_dim, public_map=public_map)


def _collect_audit(env: Any) -> dict[str, Any]:
    value = getattr(env, "last_audit", {})
    return value() if callable(value) else value


def _collection_identity(env: Any) -> dict[str, Any]:
    value = env.last_collection
    return value() if callable(value) else value


def _censor_reason(info: dict[str, Any], cutoff: bool) -> str:
    if info.get("is_success"):
        return "success"
    if info.get("collision"):
        return "collision"
    if info.get("off_route"):
        return "off_route"
    if info.get("max_time"):
        return "deadline"
    return "collection_cutoff" if cutoff else "external_truncation"


def _do_shadow(policy: SceneSAC, obs: dict[str, np.ndarray], budget: ShadowBudget,
               writer: JsonLines, phase: str, episode: int, decision: int, raw: int) -> None:
    if not budget.take(phase, episode, decision, raw):
        return
    try:
        rows = run_shadow(policy, obs)
        for row in rows:
            writer.write({"phase": phase, "episode": episode, "decision": decision,
                          "raw_steps": raw, **row})
            budget.variant_rows += 1
    except Exception as exc:
        budget.errors += 1
        writer.write({"phase": phase, "episode": episode, "decision": decision,
                      "raw_steps": raw, "error": repr(exc)})
        # A collection error is not silently accepted as absence of dependence.
        raise


def run(config: ExperimentConfig, root: Path, kind: str = "development") -> dict[str, Any]:
    config.validate()
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; do not silently substitute CPU")
    from .env import make_scene_env
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(1)
    artifacts = RunArtifacts(root, config, kind)
    env = None
    files: list[JsonLines] = []
    started = time.monotonic()
    raw, decisions, completed_episodes = 0, 0, 0
    replay_used_raw, omitted_tail_raw = 0, 0
    policy = None
    future_queue = None
    try:
        env = make_scene_env(config, artifacts.root / "environment", evaluation=False)
        observation, reset_info = env.reset(seed=config.seed)
        public_map = env.public_map
        public_map_sha = array_mapping_sha256(public_map)
        policy = SceneSAC(_encoder(config, public_map), config)
        artifacts.snapshot()
        metadata = getattr(env, "scene_metadata", lambda: {})()
        save_json(artifacts.root / "scene_metadata.json", {**metadata, "public_map_sha256": public_map_sha})
        save_json(artifacts.root / "model_size.json", {
            "encoder_parameters": sum(p.numel() for p in policy.encoder.parameters()),
            "online_trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
            "target_parameters": sum(p.numel() for p in policy.target_encoder.parameters()) + sum(p.numel() for p in policy.target_critic.parameters()),
            "map_shapes": {k: list(v.shape) for k, v in public_map.items() if isinstance(v, np.ndarray)}})
        episode_log = JsonLines(artifacts.root / "episodes.jsonl")
        training_log = JsonLines(artifacts.root / "updates.jsonl")
        observation_log = JsonLines(artifacts.root / "observation_audit.jsonl")
        action_log = JsonLines(artifacts.root / "actions.jsonl")
        shadow_log = JsonLines(artifacts.root / "policy_shadow_probes.jsonl")
        files.extend((episode_log, training_log, observation_log, action_log, shadow_log))
        replay = ObservationReplay(config.replay_capacity, config.seed)
        assembly = NStepAssembler(config.n_step, config.gamma)
        budget = ShadowBudget()
        history_audits = {"train": HistoryAudit(), "eval": HistoryAudit()}
        future_queue = ObservedFutureQueue(artifacts.root / "factual_train",
            horizon_ticks=int(round(config.prediction_seconds / config.raw_dt)),
            stride_ticks=int(round(config.prediction_dt / config.raw_dt)), require_tick_axis=True)
        episode, episode_decision = 0, 0
        account = EpisodeAccount("train", episode, config.seed)
        observation = readonly_observation(observation)
        artifacts.status("training", raw_steps=raw, decision_steps=decisions, updates=0)
        while raw < config.raw_steps:
            future_queue.observe(observation, _collection_identity(env), episode, episode_decision, add_anchor=True)
            observation_log.write({"phase": "train", "episode": episode, "decision": episode_decision,
                "raw_steps": raw, **history_audits["train"].observe(observation), "collector": _collect_audit(env)})
            _do_shadow(policy, observation, budget, shadow_log, "train", episode, episode_decision, raw)
            action = (np.random.uniform(-1, 1, size=policy.action_dim).astype(np.float32)
                      if raw < config.learning_starts_raw else policy.act(observation))
            # Exact raw budget. The final shortened action is explicitly an
            # external collection boundary, never a reset observation.
            prior_repeat = env.unwrapped.action_repeat
            env.unwrapped.action_repeat = min(config.action_repeat, config.raw_steps - raw)
            try:
                next_observation, reward, terminated, truncated, info = env.step(action)
            finally:
                env.unwrapped.action_repeat = prior_repeat
            executed = int(info["raw_steps_executed"])
            raw += executed
            decisions += 1
            if raw > config.raw_steps:
                raise RuntimeError("Environment exceeded the exact raw collection budget")
            completed = bool(terminated or truncated)
            cutoff = raw == config.raw_steps and not completed
            next_observation = readonly_observation(next_observation)
            future_queue.observe(next_observation, _collection_identity(env), episode, episode_decision + 1, add_anchor=False)
            transition = DecisionTransition(observation, action, float(reward), next_observation,
                bool(terminated), bool(truncated or cutoff), executed, episode, episode_decision)
            ready, omitted_fragment = assembly.append_collected(transition,
                action_repeat=config.action_repeat, collection_cutoff=cutoff)
            if omitted_fragment:
                omitted_tail_raw += executed
            else:
                replay_used_raw += executed
            for assembled in ready:
                replay.add(assembled)
            account.add(float(reward), info)
            action_log.write({"phase": "train", "episode": episode, "decision": episode_decision,
                "raw_steps": raw, "action": action, "raw_steps_executed": executed,
                "environment_step_reward_v2": float(reward),
                "terminated": bool(terminated), "truncated": bool(truncated or cutoff),
                "replay_included": not omitted_fragment,
                "behavior": info.get("scene_event_behavior", {}), "collector": _collect_audit(env)})
            episode_decision += 1
            if not omitted_fragment and raw >= config.learning_starts_raw and len(replay) >= config.batch_size:
                for _ in range(config.updates_per_decision):
                    audit = policy.updates == 0 or (policy.updates + 1) % 500 == 0
                    metrics = policy.update(replay.sample(config.batch_size), audit=audit)
                    if audit or policy.updates % 100 == 0:
                        diag = getattr(policy.encoder, "diagnostics", {})
                        training_log.write({"raw_steps": raw, "decision_steps": decisions, **metrics,
                            "encoder": diag() if callable(diag) else diag})
            if completed or cutoff:
                future_queue.end_episode(_censor_reason(info, cutoff))
                episode_log.write(account.result(bool(terminated), bool(truncated or cutoff), info, completed))
                if completed:
                    completed_episodes += 1
                if raw < config.raw_steps:
                    episode += 1
                    episode_decision = 0
                    observation, reset_info = env.reset(seed=config.seed + episode)
                    observation = readonly_observation(observation)
                    account = EpisodeAccount("train", episode, config.seed + episode)
                else:
                    observation = next_observation
            else:
                observation = next_observation
            if decisions % 25 == 0 or raw == config.raw_steps:
                artifacts.status("training", raw_steps=raw, decision_steps=decisions,
                    updates=policy.updates, completed_train_episodes=completed_episodes,
                    elapsed_seconds=time.monotonic() - started)
        if assembly.pending:
            raise RuntimeError("Final n-step queue was not flushed at the collection boundary")
        future_queue.close()
        future_queue = None
        checkpoint = artifacts.root / "final_model.pt"
        torch.save({**policy.checkpoint(), "raw_steps": raw, "decision_steps": decisions,
                    "replay_used_raw": replay_used_raw, "omitted_tail_raw": omitted_tail_raw,
                    "training_seed": config.seed, "fresh": True,
                    "public_map": public_map}, checkpoint)
        checkpoint_sha = sha256(checkpoint)
        # Reload before evaluation, and verify deterministic identity on an
        # already-collected physical observation without an additional step.
        payload = torch.load(checkpoint, map_location=config.device, weights_only=False)
        evaluation_policy = SceneSAC(_encoder(config, public_map), config)
        evaluation_policy.load_state_dict(payload["model"])
        evaluation_policy.eval()
        policy.eval()
        if any(not torch.equal(value, evaluation_policy.state_dict()[name]) for name, value in policy.state_dict().items()):
            raise RuntimeError("Reloaded weights/buffers differ from the final training checkpoint")
        reload_action_error = float(np.max(np.abs(policy.act(observation, True) - evaluation_policy.act(observation, True))))
        if reload_action_error > 1e-6:
            raise RuntimeError("Checkpoint reload changed the deterministic action")
        env.close()
        env = None
        evaluation_results = []
        if config.eval_episodes:
            env = make_scene_env(config, artifacts.root / "evaluation_environment", evaluation=True)
            future_queue = ObservedFutureQueue(artifacts.root / "factual_eval",
                horizon_ticks=int(round(config.prediction_seconds / config.raw_dt)),
                stride_ticks=int(round(config.prediction_dt / config.raw_dt)), require_tick_axis=True)
            artifacts.status("evaluating", raw_steps=raw, decision_steps=decisions,
                             updates=policy.updates, checkpoint_sha256=checkpoint_sha)
            for eval_episode in range(config.eval_episodes):
                seed = config.eval_seed_start + eval_episode
                observation, reset_info = env.reset(seed=seed)
                if array_mapping_sha256(env.public_map) != public_map_sha:
                    raise RuntimeError("Evaluation changed the public map or ego-navigation contract")
                account = EpisodeAccount("eval", eval_episode, seed)
                decision = 0
                done = False
                while not done:
                    future_queue.observe(observation, _collection_identity(env), eval_episode, decision, add_anchor=True)
                    observation_log.write({"phase": "eval", "episode": eval_episode, "seed": seed,
                        "decision": decision, **history_audits["eval"].observe(observation), "collector": _collect_audit(env)})
                    _do_shadow(evaluation_policy, observation, budget, shadow_log, "eval", eval_episode, decision, account.raw_steps)
                    action = evaluation_policy.act(observation, True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    future_queue.observe(observation, _collection_identity(env), eval_episode, decision + 1, add_anchor=False)
                    account.add(float(reward), info)
                    action_log.write({"phase": "eval", "episode": eval_episode, "seed": seed,
                        "decision": decision, "raw_steps": account.raw_steps, "action": action,
                        "environment_step_reward_v2": float(reward), "raw_steps_executed": info["raw_steps_executed"],
                        "behavior": info.get("scene_event_behavior", {}), "collector": _collect_audit(env)})
                    decision += 1
                    done = bool(terminated or truncated)
                    if account.raw_steps > int(round(config.deadline_seconds / config.raw_dt)):
                        raise RuntimeError("Evaluation failed to respect the task deadline")
                row = {**account.result(bool(terminated), bool(truncated), info, True),
                       "checkpoint_sha256": checkpoint_sha}
                future_queue.end_episode(_censor_reason(info, False))
                evaluation_results.append(row)
                episode_log.write(row)
                artifacts.status("evaluating", raw_steps=raw, decision_steps=decisions,
                    updates=policy.updates, completed_eval_episodes=len(evaluation_results),
                    checkpoint_sha256=checkpoint_sha)
            future_queue.close()
            future_queue = None
        save_json(artifacts.root / "diagnostic_summary.json", {"history": {phase: dict(audit.counts) for phase, audit in history_audits.items()},
                                                             "shadow": budget.summary()})
        summary = {"kind": kind, "protocol": config.protocol, "method": config.method,
            "scenario": config.scenario, "seed": config.seed, "fresh": True,
            "raw_steps": raw, "decision_steps": decisions, "updates": policy.updates,
            "replay_used_raw": replay_used_raw, "omitted_tail_raw": omitted_tail_raw,
            "completed_train_episodes": completed_episodes, "completed_eval_episodes": len(evaluation_results),
            "eval_seeds": [r["seed"] for r in evaluation_results], "checkpoint_sha256": checkpoint_sha,
            "public_map_sha256": public_map_sha,
            "reload_action_max_abs_error": reload_action_error,
            "counts": {key: sum(int(r[key]) for r in evaluation_results) for key in ("success", "collision", "off_route", "timeout")},
            "mean_environment_step_reward_v2_return": float(np.mean([r["environment_step_reward_v2_return"] for r in evaluation_results])) if evaluation_results else None,
            "mean_raw_environment_return": float(np.mean([r["raw_environment_return"] for r in evaluation_results])) if evaluation_results and all(r["raw_environment_return"] is not None for r in evaluation_results) else None,
            "max_reward_component_error": max((r["reward_component_error"] for r in evaluation_results), default=None),
            "elapsed_seconds": time.monotonic() - started,
            "interpretation": "Correctness smoke, not a 100k performance experiment" if kind == "smoke" else "Single training seed development result"}
        save_json(artifacts.root / "result.json", summary)
        artifacts.status("completed", **summary)
        return summary
    except BaseException as exc:
        artifacts.status("failed", error=repr(exc), raw_steps=raw, decision_steps=decisions,
                         updates=policy.updates if policy else 0, elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        if future_queue is not None:
            future_queue.close()
        if env is not None:
            env.close()
        for writer in files:
            writer.close()
