"""Bounded, read-only diagnostics collected during ordinary environment work."""
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class JsonLines:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("x", encoding="utf-8")
        self.rows = 0

    def write(self, data: dict[str, Any]) -> None:
        self.file.write(json.dumps(json_value(data), allow_nan=False, ensure_ascii=False) + "\n")
        self.file.flush()
        self.rows += 1

    def close(self) -> None:
        self.file.close()


class ShadowBudget:
    """Count real source states, not intervention-variant output rows."""
    def __init__(self):
        self.train_keys: set[tuple[int, int]] = set()
        self.eval_keys: dict[int, set[int]] = {}
        self.next_raw_threshold = 5000
        self.errors = 0
        self.variant_rows = 0

    def take(self, phase: str, episode: int, decision: int, raw_steps: int) -> bool:
        if phase == "train":
            if raw_steps < self.next_raw_threshold or len(self.train_keys) >= 20:
                return False
            key = (episode, decision)
            if key in self.train_keys:
                return False
            self.train_keys.add(key)
            while self.next_raw_threshold <= raw_steps:
                self.next_raw_threshold += 5000
            return True
        if phase != "eval":
            raise ValueError("Unknown shadow phase")
        # Spread evaluation coverage instead of taking only the first four
        # decisions. Short episodes honestly yield fewer states.
        if decision not in (0, 10, 30, 60):
            return False
        keys = self.eval_keys.setdefault(episode, set())
        if len(keys) >= 4 or decision in keys:
            return False
        keys.add(decision)
        return True

    def summary(self) -> dict[str, Any]:
        return {"train_unique_states": len(self.train_keys),
                "eval_unique_states": sum(len(x) for x in self.eval_keys.values()),
                "eval_states_per_episode": {str(k): len(v) for k, v in self.eval_keys.items()},
                "variant_rows": self.variant_rows, "errors": self.errors,
                "additional_simulator_steps": 0}


class HistoryAudit:
    def __init__(self):
        self.counts: Counter = Counter()

    def observe(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        mask = np.asarray(observation["history_valid"], dtype=bool)
        active = np.asarray(observation["actor_valid"], dtype=bool)
        self.counts["decisions"] += 1
        self.counts["actor_slots"] += int(active.sum())
        counts = mask.sum(-1)
        last = np.where(mask, np.arange(mask.shape[-1]), -1).max(-1)
        nonempty = counts > 0
        short = nonempty & (counts < mask.shape[-1])
        old = counts - 1
        old_is_padding = nonempty & ~np.take_along_axis(mask, np.maximum(old, 0)[..., None], -1).squeeze(-1)
        first = np.where(mask, np.arange(mask.shape[-1]), mask.shape[-1]).min(-1)
        gaps = nonempty & ((last - first + 1) != counts)
        row = {"active_slots": int(active.sum()), "empty_active": int((active & ~nonempty).sum()),
               "short_nonempty": int((active & short).sum()),
               "left_padding": int((active & nonempty & (first > 0) & ~gaps).sum()),
               "right_padding": int((active & nonempty & (last < mask.shape[-1] - 1) & ~gaps).sum()),
               "internal_gaps": int((active & gaps).sum()),
               "old_selector_mismatch": int((active & nonempty & (old != last)).sum()),
               "old_selector_padding": int((active & old_is_padding).sum())}
        self.counts.update(row)
        return row


def tensor_statistics(encoder: torch.nn.Module) -> dict[str, float | None]:
    report = {}
    for name, module in encoder.named_children():
        parameters = list(module.parameters())
        if parameters:
            norm = sum(float(p.detach().square().sum()) for p in parameters) ** 0.5
            grads = [p.grad for p in parameters if p.grad is not None]
            report[f"{name}/parameter_norm"] = norm
            report[f"{name}/gradient_norm"] = sum(float(g.detach().square().sum()) for g in grads) ** 0.5 if grads else None
    return report


@torch.no_grad()
def run_shadow(policy: Any, observation: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Same-state functional probes; never step/reset the simulator.

    Preserve global torch RNG states so even a future stochastic encoder cannot
    perturb subsequent training samples through diagnostics.
    """
    devices = [policy.device_obj.index or 0] if policy.device_obj.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        action, log_std = policy.action_mean(observation)
        variants = [("identity", observation)]
        # This is a history-information probe, not a causally valid scene edit.
        history_only_current = {k: np.array(v, copy=True) for k, v in observation.items()}
        mask = history_only_current["history_valid"]
        last = np.where(mask, np.arange(mask.shape[-1]), -1).max(-1)
        mask[:] = False
        for index, at in enumerate(last):
            if at >= 0:
                mask[index, at] = True
        variants.append(("history_current_only", history_only_current))
        results = []
        for name, variant in variants:
            other, other_log_std = policy.action_mean(variant)
            results.append({"variant": name, "active": True, "error": None,
                "action_mean": other, "action_delta": other - action,
                "log_std_delta": other_log_std - log_std,
                "interpretation": "functional sensitivity only; no extra rollout"})
        if policy.config.method == "sac_scene_eventgraph_cv_v1":
            other, other_log_std = policy.action_mean(observation, disable_events=True)
            results.append({"variant": "event_branch_off", "active": True, "error": None,
                "action_mean": other, "action_delta": other - action,
                "log_std_delta": other_log_std - log_std,
                "interpretation": "functional branch intervention, not performance causation"})
        else:
            results.append({"variant": "event_branch_off", "active": False, "error": None,
                "reason": "NA: M0 has no event branch"})
        return results
