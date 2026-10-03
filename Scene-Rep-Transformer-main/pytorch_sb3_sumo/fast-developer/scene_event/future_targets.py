"""Delayed factual targets from later ordinary observations, never an oracle.

This Stage-1 collector does not train a predictor. Unknown future samples remain
masked. Labels are stored separately from the policy observation and are keyed
by collection identity. Simulator tracking keys are used only for joining rows.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import readonly_observation
from .provenance import save_json


@dataclass
class PendingFuture:
    episode: int
    decision: int
    tick: int
    keys: tuple[str | None, ...]
    observation: dict[str, np.ndarray]
    xy: np.ndarray
    valid: np.ndarray
    lane_ptr: np.ndarray


class ObservedFutureQueue:
    def __init__(self, root: Path, horizon_ticks: int = 30, stride_ticks: int = 3,
                 anchor_interval: int = 10, max_anchors: int = 5000, shard_size: int = 64,
                 require_tick_axis: bool = False):
        if horizon_ticks < 1 or stride_ticks < 1 or horizon_ticks % stride_ticks:
            raise ValueError("Future horizon must be a positive multiple of sample stride")
        if min(anchor_interval, max_anchors, shard_size) < 1:
            raise ValueError("Collection budgets must be positive")
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.offsets = np.arange(stride_ticks, horizon_ticks + 1, stride_ticks)
        self.horizon_ticks, self.anchor_interval = horizon_ticks, anchor_interval
        self.max_anchors, self.shard_size = max_anchors, shard_size
        self.require_tick_axis = require_tick_axis
        self.explicit_tick_axis_calls = 0
        self.missing_tick_axis_calls = 0
        self.pending: list[PendingFuture] = []
        self.ready: list[dict[str, Any]] = []
        self.accepted = 0
        self.written = 0
        self.shards = 0
        self.valid_samples = 0
        self.total_samples = 0
        self.censored_anchors = 0
        self.eligible_anchors = 0
        self.dropped_budget = 0
        self.last_identity: tuple[int, int] | None = None

    def observe(self, observation: dict[str, np.ndarray], sidecar: dict[str, Any],
                episode: int, decision: int, *, add_anchor: bool) -> None:
        tick = int(sidecar["raw_tick"])
        keys = tuple(sidecar["actor_keys"])
        if len(keys) != len(observation["actor_valid"]):
            raise ValueError("Tracking keys and actor slots disagree")
        # A future state can update previous anchors, but never their input.
        positions = np.asarray(observation["actor_history"])[..., :2]
        frame_valid = np.asarray(observation["history_valid"], dtype=bool)
        frame_ticks = sidecar.get("frame_raw_tick")
        if frame_ticks is None:
            self.missing_tick_axis_calls += 1
            if self.require_tick_axis:
                raise ValueError("Explicit raw-frame timestamps are required for factual labels")
        else:
            frame_ticks = np.asarray(frame_ticks)
            if frame_ticks.shape != frame_valid.shape:
                raise ValueError("Frame timestamp and history-mask shapes disagree")
            expected_ticks = np.broadcast_to(np.arange(tick - frame_valid.shape[1] + 1, tick + 1), frame_valid.shape)
            if np.any(frame_valid & (frame_ticks != expected_ticks)):
                raise ValueError("History compressed or reordered the physical raw-tick axis")
            self.explicit_tick_axis_calls += 1
        lookup = {key: i for i, key in enumerate(keys) if key is not None and bool(observation["actor_valid"][i])}
        retained = []
        for anchor in self.pending:
            if anchor.episode != episode:
                raise ValueError("Pending targets must be censored before episode reset")
            for slot, key in enumerate(anchor.keys):
                current_slot = lookup.get(key)
                if current_slot is None:
                    continue
                for future_index, at in enumerate(anchor.tick + self.offsets):
                    if frame_ticks is not None:
                        matches = np.flatnonzero((frame_ticks[current_slot] == int(at)) & frame_valid[current_slot])
                        if len(matches) > 1:
                            raise ValueError("Duplicate physical timestamp in actor history")
                        history_index = int(matches[0]) if len(matches) else -1
                    else:
                        history_index = positions.shape[1] - 1 - (tick - int(at))
                    if 0 <= history_index < positions.shape[1] and at <= tick and frame_valid[current_slot, history_index]:
                        anchor.xy[slot, future_index] = positions[current_slot, history_index]
                        anchor.valid[slot, future_index] = True
                        # Current observed lane only; no reconstruction using a
                        # hidden route and no guessed lane for an old frame.
                        if at == tick:
                            anchor.lane_ptr[slot, future_index] = int(observation["actor_lane_ptr"][current_slot])
            if tick >= anchor.tick + self.horizon_ticks:
                self._finish(anchor, "horizon")
            else:
                retained.append(anchor)
        self.pending = retained
        identity = (episode, tick)
        eligible = add_anchor and decision % self.anchor_interval == 0 and identity != self.last_identity
        if eligible:
            self.eligible_anchors += 1
            if self.accepted >= self.max_anchors:
                self.dropped_budget += 1
                self.last_identity = identity
        if eligible and self.accepted < self.max_anchors:
            actors = len(keys)
            self.pending.append(PendingFuture(episode, decision, tick, keys, readonly_observation(observation),
                np.zeros((actors, len(self.offsets), 2), dtype=np.float32),
                np.zeros((actors, len(self.offsets)), dtype=bool),
                np.full((actors, len(self.offsets)), -1, dtype=np.int32)))
            self.accepted += 1
            self.last_identity = identity

    def end_episode(self, reason: str) -> None:
        if reason not in {"success", "collision", "off_route", "deadline", "collection_cutoff", "external_truncation"}:
            raise ValueError("Censor reason must be explicit")
        for anchor in self.pending:
            self._finish(anchor, reason)
        self.pending.clear()

    def _finish(self, anchor: PendingFuture, reason: str) -> None:
        source_valid = np.asarray(anchor.observation["actor_valid"], dtype=bool)
        active_future = np.broadcast_to(source_valid[:, None], anchor.valid.shape)
        self.total_samples += int(active_future.sum())
        self.valid_samples += int((anchor.valid & active_future).sum())
        if not np.all(anchor.valid[active_future]):
            self.censored_anchors += 1
        self.ready.append({"anchor": anchor, "reason": reason})
        if len(self.ready) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self.ready:
            return
        rows = self.ready
        output = {
            "episode": np.asarray([r["anchor"].episode for r in rows], dtype=np.int64),
            "decision": np.asarray([r["anchor"].decision for r in rows], dtype=np.int64),
            "raw_tick": np.asarray([r["anchor"].tick for r in rows], dtype=np.int64),
            "future_offsets_raw": self.offsets,
            "future_xy": np.stack([r["anchor"].xy for r in rows]),
            "future_valid": np.stack([r["anchor"].valid for r in rows]),
            "future_observed_lane_ptr": np.stack([r["anchor"].lane_ptr for r in rows]),
            "censor_reason": np.asarray([r["reason"] for r in rows], dtype="U32"),
            "tracking_keys_audit_only": np.asarray([["" if key is None else str(key) for key in r["anchor"].keys] for r in rows]),
        }
        for key in rows[0]["anchor"].observation:
            output["observation/" + key] = np.stack([r["anchor"].observation[key] for r in rows])
        destination = self.root / f"factual_{self.shards:05d}.npz"
        if destination.exists():
            raise FileExistsError(destination)
        np.savez_compressed(destination, **output)
        self.written += len(rows)
        self.shards += 1
        self.ready.clear()

    def close(self) -> dict[str, Any]:
        if self.pending:
            self.end_episode("collection_cutoff")
        self._flush()
        manifest = {"accepted_anchors": self.accepted, "written_anchors": self.written,
            "valid_actor_future_samples": self.valid_samples, "active_actor_future_slots": self.total_samples,
            "censored_anchors": self.censored_anchors, "shards": self.shards,
            "eligible_anchors": self.eligible_anchors, "dropped_budget": self.dropped_budget,
            "explicit_tick_axis_calls": self.explicit_tick_axis_calls,
            "missing_tick_axis_calls": self.missing_tick_axis_calls,
            "explicit_tick_axis_required": self.require_tick_axis,
            "anchor_interval_decisions": self.anchor_interval, "max_anchors": self.max_anchors,
            "future_offsets_raw": self.offsets.tolist(), "input_contains_future": False,
            "hidden_route_queries": 0, "additional_simulation_steps": 0,
            "scope": "factual future observations under the collection policy; not counterfactual responses",
            "limitations": "Missing/culled actors remain unknown. Old-frame lane assignments are not inferred. Split by episode, not overlapping anchors."}
        save_json(self.root / "manifest.json", manifest)
        return manifest
