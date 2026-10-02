"""Frozen-policy, three-arm diagnostic for the sorted intersection goal-only model.

This runner reuses D1's saved-model evaluator.  It never calls ``learn`` and
does not write into the source training run.  The route-veto arm is an
observation-preserving action wrapper: it only changes an explicitly requested
lane action when fresh route data proves that the current lane can continue to
the planned next edge and the requested target lane cannot.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except Exception:  # The pure decision helper remains importable without Gymnasium.
    gym = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAST_DEVELOPER = Path(__file__).resolve().parent
METHOD = "sac_mlp_d1_st_rt_topo_goalonly_v1"
SCENARIO = "intersection_sorted"
DEPART_SCALE = 4.0
EVAL_SEED_START = 10_000
EVAL_EPISODES = 100
REFERENCE_RUN = (
    WORKSPACE_ROOT
    / "runs"
    / "sortg3_1002"
    / f"{METHOD}__{SCENARIO}_depart4p0"
)
REFERENCE_MODEL = REFERENCE_RUN / "final_model.zip"
REFERENCE_EVALUATION = REFERENCE_RUN / "evaluation_results.json"
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "runs" / "sortg3_causal_1002"
ARMS = ("frozen_control", "topology_goal_off", "route_lane_veto")
ARM_DIRS = {
    "frozen_control": "control",
    "topology_goal_off": "goaloff",
    "route_lane_veto": "routeveto",
}
MAX_PARALLEL = 2


def _as_label(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if numeric in (0, 1) else None


def route_veto_decision(
    action: Any,
    *,
    context_known: bool,
    context_fresh: bool,
    current_road: str | None,
    has_next_edge: bool,
    current_lane_label: Any,
    target_lane_label: Any,
    lane_command: Any,
    target_lane_id: str | None = None,
    target_lane_reason: str | None = None,
    target_lane_in_range: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Return a copied action and an auditable route-veto decision.

    Only a fresh, known eligible-current to known ineligible-target transition
    is vetoed.  Every uncertain or out-of-scope case passes the action through.
    The second action coordinate is changed to zero (SUMO hold); the speed
    coordinate and caller-owned input are preserved.
    """
    if isinstance(action, np.ndarray):
        action_copy = action.copy()
    elif isinstance(action, (list, tuple)):
        action_copy = np.asarray(action).copy()
    else:
        try:
            action_copy = np.asarray(action).copy()
        except Exception:
            action_copy = action
    try:
        flat = np.asarray(action_copy).reshape(-1)
    except Exception:
        flat = np.asarray([], dtype=np.float32)
    try:
        command = int(lane_command)
    except (TypeError, ValueError, OverflowError):
        command = None
    current_label = _as_label(current_lane_label)
    target_label = _as_label(target_lane_label)

    reason = "veto_not_applicable"
    if flat.size != 2:
        reason = "action_shape_unknown"
    elif not context_known:
        reason = "context_unknown"
    elif not context_fresh:
        reason = "stale_context"
    elif not current_road:
        reason = "current_road_unknown"
    elif str(current_road).startswith(":"):
        reason = "internal_road"
    elif not has_next_edge:
        reason = "no_next_edge"
    elif current_label is None:
        reason = "current_lane_unknown"
    elif current_label != 1:
        reason = "current_lane_ineligible"
    elif command is None:
        reason = "lane_command_unknown"
    elif command == 0:
        reason = "hold_command"
    elif not target_lane_in_range:
        reason = "target_lane_out_of_range"
    elif target_label is None:
        reason = "target_lane_unknown"
    elif target_label == 1:
        reason = "target_lane_eligible"
    elif target_label == 0:
        reason = "known_unreachable_target_lane_vetoed"

    veto_applied = reason == "known_unreachable_target_lane_vetoed"
    speed_unchanged = True
    if veto_applied:
        np.asarray(action_copy).reshape(-1)[1] = 0.0
        speed_unchanged = bool(
            np.asarray(action_copy).reshape(-1)[0] == np.asarray(action).reshape(-1)[0]
        )
    audit = {
        "protocol": "goalonly_route_lane_veto_v1",
        "veto_applied": veto_applied,
        "reason": reason,
        "context_known": bool(context_known),
        "context_fresh": bool(context_fresh),
        "current_road": current_road,
        "has_next_edge": bool(has_next_edge),
        "current_lane_label": current_label,
        "target_lane_label": target_label,
        "target_lane_id": target_lane_id,
        "target_lane_reason": target_lane_reason,
        "target_lane_in_range": bool(target_lane_in_range),
        "lane_command_requested": command,
        "lane_command_after_veto": 0 if veto_applied else command,
        "action_requested": _jsonable(action),
        "action_after_veto": _jsonable(action_copy),
        "speed_action_unchanged": speed_unchanged,
        "interpretation_limit": (
            "A veto changes only the requested lateral command; it does not guarantee "
            "that the vehicle changes lane or completes the route."
        ),
    }
    return action_copy, audit


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


@contextmanager
def _consume_one_step_refresh(raw):
    """Use the immediately refreshed decision context without duplicate reads.

    ``SumoSceneEnv.step`` has one decision-rate refresh immediately before
    control application. The wrapper refreshes just before classifying the
    proposed target lane, so consume that first inner refresh only. Any later
    refreshes still call the original method. Always restore the exact prior
    instance/class lookup behavior, including when ``step`` raises.
    """
    name = "_behavior_refresh_dynamic_context"
    original = getattr(raw, name)
    namespace = getattr(raw, "__dict__", {})
    had_instance_value = name in namespace
    old_instance_value = namespace.get(name)
    state = {"suppressed": False, "calls": 0}

    def consume_once():
        state["calls"] += 1
        if not state["suppressed"]:
            state["suppressed"] = True
            return None
        return original()

    setattr(raw, name, consume_once)
    try:
        yield state
    finally:
        if had_instance_value:
            setattr(raw, name, old_instance_value)
        else:
            try:
                delattr(raw, name)
            except AttributeError:
                setattr(raw, name, original)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_d1():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(FAST_DEVELOPER) not in sys.path:
        sys.path.insert(0, str(FAST_DEVELOPER))
    return importlib.import_module("train_intersection_yield_v2_d1")


def _reference_smoke_case() -> tuple[int, int]:
    """Choose the first existing goal-only timeout and its 0-based eval index."""
    payload = json.loads(REFERENCE_EVALUATION.read_text(encoding="utf-8"))
    records = payload.get("episode_records", [])
    for row in records:
        if bool(row.get("timeout")) and not bool(row.get("collision")):
            index = int(row.get("episode", 0))
            seed = row.get("episode_seed", row.get("seed", EVAL_SEED_START + index))
            seed = int(seed)
            if seed != EVAL_SEED_START + index:
                raise ValueError(
                    f"reference seed/index mismatch: seed={seed}, episode={index}"
                )
            return seed, index
    raise ValueError("reference evaluation contains no exclusive timeout smoke seed")


def _expected_config(d1) -> dict[str, Any]:
    cfg = d1.D1_CONFIG.get(METHOD)
    if cfg is None:
        raise RuntimeError(f"D1 method registry lacks required method {METHOD!r}")
    expected = {
        "use_route": True,
        "use_topology": True,
        "use_topology_actor_intent": False,
        "use_topology_relations": False,
        "use_topology_goal": True,
        "use_route_reachability": True,
        "use_slots": False,
        "use_graph_slt": False,
        "use_sbs": False,
        "representation_coef": 0.0,
        "slot_balance_coef": 0.0,
    }
    mismatches = {key: (cfg.get(key), value) for key, value in expected.items() if cfg.get(key) != value}
    if mismatches:
        raise RuntimeError(f"goal-only config differs from registered protocol: {mismatches}")
    return dict(cfg)


def _source_snapshot_paths() -> list[Path]:
    candidates = [
        Path(__file__).resolve(),
        FAST_DEVELOPER / "train_intersection_yield_v2_d1.py",
        FAST_DEVELOPER / "train_intersection_yield_v2.py",
        PROJECT_ROOT / "envs" / "sumo" / "sumo_env.py",
        PROJECT_ROOT / "envs" / "sumo" / "paper_env.py",
        PROJECT_ROOT / "envs" / "sumo" / "route_reachability_v1.py",
        PROJECT_ROOT / "algos" / "sb3_torch" / "incremental_topo_encoder.py",
        PROJECT_ROOT / "algos" / "sb3_torch" / "policy_shadow_probes.py",
        FAST_DEVELOPER / "behavior_diagnostics.py",
        FAST_DEVELOPER / "reward_shaping_v2.py",
    ]
    return [path for path in candidates if path.is_file()]


def _preflight(run_root: Path, smoke: bool) -> dict[str, Any]:
    d1 = _load_d1()
    cfg = _expected_config(d1)
    if not REFERENCE_MODEL.is_file() or not REFERENCE_EVALUATION.is_file():
        raise FileNotFoundError(f"missing frozen reference artifacts under {REFERENCE_RUN}")
    root = Path(run_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"refusing to reuse existing causal run root: {root}")
    if len(str(root / "routeveto" / "diagnostics" / "eval" / "route_lane_veto_actions.jsonl")) >= 240:
        raise ValueError("causal output path is too long for Windows; choose a shorter run root")
    eval_sig = inspect.signature(d1._evaluate_saved)
    if "policy_shadow_critical_eval_sample" not in eval_sig.parameters:
        raise RuntimeError("D1 critical evaluation sampler is not available yet")
    _apply_original_eval_protocol(d1)
    smoke_seed, smoke_index = _reference_smoke_case()
    evaluation = json.loads(REFERENCE_EVALUATION.read_text(encoding="utf-8"))
    identity = evaluation.get("identity", {})
    model_sha = _sha256(REFERENCE_MODEL)
    if identity.get("checkpoint_sha256") != model_sha:
        raise RuntimeError("reference evaluation checkpoint SHA does not match frozen model")
    seed_rows = [row.get("episode_seed", row.get("seed")) for row in evaluation.get("episode_records", [])]
    seed_rows = [int(seed) for seed in seed_rows if seed is not None]
    if len(evaluation.get("episode_records", [])) != EVAL_EPISODES:
        raise RuntimeError("reference evaluation is not the expected 100-episode evaluation")
    if seed_rows and seed_rows != list(range(EVAL_SEED_START, EVAL_SEED_START + EVAL_EPISODES)):
        raise RuntimeError("reference evaluation seed sequence is not 10000..10099")
    traffic_source = d1.PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1" / SCENARIO
    traffic_files = sorted((traffic_source / "traffic").glob("traffic_*.rou.xml"))
    if len(traffic_files) != 30:
        raise RuntimeError(f"expected 30 sorted traffic templates, found {len(traffic_files)}")
    return {
        "workspace_root": str(WORKSPACE_ROOT),
        "project_root": str(PROJECT_ROOT),
        "run_root": str(root),
        "mode": "smoke" if smoke else "formal",
        "arms": list(ARMS),
        "max_parallel_worker_processes": MAX_PARALLEL,
        "eval_device": "cpu (matches D1 frozen final evaluation)",
        "method": METHOD,
        "registered_config": cfg,
        "scenario": SCENARIO,
        "depart_scale": DEPART_SCALE,
        "eval_seed_start": smoke_seed if smoke else EVAL_SEED_START,
        "eval_episodes": 1 if smoke else EVAL_EPISODES,
        "smoke_reference_episode_index": smoke_index if smoke else None,
        "reference_model": str(REFERENCE_MODEL.resolve()),
        "reference_model_sha256": model_sha,
        "reference_evaluation": str(REFERENCE_EVALUATION.resolve()),
        "reference_eval_checkpoint_sha256": identity.get("checkpoint_sha256"),
        "reference_traffic_templates": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in traffic_files
        ],
        "d1_evaluator_signature": str(eval_sig),
        "critical_sampler": "policy_shadow_critical_eval_sample=True for control and route-veto; cap=3",
        "route_veto": "only known fresh eligible-current to known ineligible-target, non-hold requests; no speed or reward edits",
        "source_files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in _source_snapshot_paths()
        ],
    }


def _apply_original_eval_protocol(d1) -> None:
    """Patch only module-local scene globals; keep original sorted pool/reward path."""
    d1._apply_patch(DEPART_SCALE, SCENARIO, "validation")
    d1.ensure_sorted_scenario()
    parent = d1.PARENT.get(METHOD)
    if parent is None or parent not in d1.base.SEED_START:
        raise RuntimeError("goal-only method has no registered base evaluation seed block")
    if int(d1.base.SEED_START[parent]) != EVAL_SEED_START:
        raise RuntimeError(f"unexpected original evaluation seed start: {d1.base.SEED_START[parent]}")


def _state_fingerprint(model) -> dict[str, Any]:
    import torch

    digest = hashlib.sha256()
    tensors = 0
    for name, value in sorted(model.policy.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        tensors += 1
    return {
        "sha256": digest.hexdigest(),
        "state_dict_tensors": tensors,
        "updates": int(getattr(model, "_n_updates", 0)),
    }


def _wrap_goal_off_predict(model) -> None:
    extractor = getattr(getattr(model, "actor", None), "features_extractor", None)
    probe = getattr(extractor, "policy_shadow_probe", None)
    if not callable(probe):
        raise TypeError("goal-only extractor does not expose policy_shadow_probe()")
    original_predict = model.predict

    def predict_with_goal_off(observation, *args, **kwargs):
        with probe("topology_goal_off"):
            return original_predict(observation, *args, **kwargs)

    model.predict = predict_with_goal_off


def _install_loader_capture(
    d1,
    arm: str,
    fingerprints: list[dict[str, Any]],
    loaded_models: list[Any],
):
    from algos.sb3_torch.sac import SceneRepresentationSAC

    cls = SceneRepresentationSAC
    had_local = "load" in cls.__dict__
    old_local = cls.__dict__.get("load")
    original_bound = cls.load

    def load_capture(_cls, *args, **kwargs):
        model = original_bound(*args, **kwargs)
        before = _state_fingerprint(model)
        fingerprints.append({"phase": "before_eval", **before})
        loaded_models.append(model)
        if arm == "topology_goal_off":
            _wrap_goal_off_predict(model)
        original_predict = getattr(model, "predict")

        def finalize_fingerprint(*_args, **_kwargs):
            after = _state_fingerprint(model)
            fingerprints.append({"phase": "after_eval", **after})
            return after

        model._causal_original_predict = original_predict
        model._causal_fingerprint_after = finalize_fingerprint
        return model

    cls.load = classmethod(load_capture)

    def restore():
        if had_local:
            setattr(cls, "load", old_local)
        else:
            try:
                delattr(cls, "load")
            except AttributeError:
                pass

    return restore


def _install_eval_env_wrapper(
    d1,
    arm: str,
    arm_dir: Path,
    *,
    smoke_episode_index: int | None,
):
    if gym is None:
        raise RuntimeError("Gymnasium is required for frozen-evaluation wrappers")
    original_wrap = d1._wrap_route_reachability_env

    class ReferenceTrafficIndexWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.raw = getattr(env, "unwrapped", env)
            self._smoke_episode_index = smoke_episode_index
            self._traffic_meta = {}
            traffic_path = arm_dir / "diagnostics" / "eval" / "traffic_selection.jsonl"
            traffic_path.parent.mkdir(parents=True, exist_ok=True)
            self._traffic_handle = traffic_path.open("x", encoding="utf-8", newline="\n")
            self._wrapper_closed = False

        def reset(self, *, seed=None, options=None):
            prior_index = getattr(self.raw, "_traffic_episode_index", None)
            if self._smoke_episode_index is not None:
                # Pair smoke arms on the same old template and logical seed.
                self.raw._traffic_episode_index = int(self._smoke_episode_index)
                self.raw._traffic_roll = None
            observation, info = self.env.reset(seed=seed, options=options)
            self._traffic_meta = {
                "traffic_index_before_reset": (
                    int(self._smoke_episode_index)
                    if self._smoke_episode_index is not None else prior_index
                ),
                "traffic_roll": getattr(self.raw, "_traffic_roll", None),
                "selected_traffic_path": str(getattr(self.raw, "_selected_traffic_path", "")),
                "reset_info_traffic_cycle_episode": (info or {}).get("source_traffic_cycle_episode"),
            }
            selected_path = Path(self._traffic_meta["selected_traffic_path"])
            try:
                selected_sha256 = _sha256(selected_path) if selected_path.is_file() else None
            except OSError:
                selected_sha256 = None
            self._traffic_handle.write(
                json.dumps(
                    _jsonable({
                        "episode_seed": int(seed) if seed is not None else None,
                        **self._traffic_meta,
                        "selected_traffic_sha256": selected_sha256,
                    }),
                    ensure_ascii=False,
                    allow_nan=False,
                ) + "\n"
            )
            self._traffic_handle.flush()
            return observation, info

        def close(self):
            if self._wrapper_closed:
                return None
            self._wrapper_closed = True
            if not self._traffic_handle.closed:
                self._traffic_handle.flush()
                self._traffic_handle.close()
            return self.env.close()

    class RouteLaneVetoWrapper(ReferenceTrafficIndexWrapper):
        def __init__(self, env):
            super().__init__(env)
            out = arm_dir / "diagnostics" / "eval" / "route_lane_veto_actions.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            self._handle = out.open("x", encoding="utf-8", newline="\n")
            self._episode_index = None
            self._episode_seed = None
            self._decision = 0

        def _write_row(self, row):
            self._handle.write(json.dumps(_jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
            self._handle.flush()

        def reset(self, *, seed=None, options=None):
            self._episode_seed = int(seed) if seed is not None else None
            self._episode_index = (
                self._episode_seed - EVAL_SEED_START
                if self._episode_seed is not None else None
            )
            self._decision = 0
            return super().reset(seed=seed, options=options)

        def step(self, action):
            self._decision += 1
            raw = self.raw
            refresh = getattr(raw, "_behavior_refresh_dynamic_context", None)
            context = {}
            current_facts = {}
            target_facts = {}
            route_helpers_present = callable(refresh)
            refresh_error = None
            consumed = {"done": False}
            original_action = np.asarray(action).copy()
            executed_action = original_action.copy()
            action_clipped = np.clip(original_action.reshape(-1), -1.0, 1.0)
            lane_command = None
            target_lane_id = None
            target_lane_in_range = False
            target_lane_reason = "route_helpers_unavailable"

            if route_helpers_present:
                try:
                    # SUMO step also refreshes once before applying control. Move
                    # that refresh to this boundary so the route decision is age 0.
                    refresh()
                    context = raw._behavior_current_ego_route_context()
                    current_facts = raw._behavior_route_lane_facts(context)
                except Exception as exc:
                    refresh_error = f"{type(exc).__name__}: {exc}"
                    context = {}
                    current_facts = {}

            current_raw = getattr(raw, "_raw_steps", None)
            sampled_raw = context.get("context_sample_raw_step")
            try:
                age = int(current_raw) - int(sampled_raw) if current_raw is not None and sampled_raw is not None else None
            except (TypeError, ValueError, OverflowError):
                age = None
            context_fresh = age == 0
            context_known = bool(
                current_facts.get("route_lane_context_known")
                and current_facts.get("route_lane_status_known")
            )

            try:
                _target_speed, lane_command = raw.adapt_action(action_clipped)
            except Exception as exc:
                refresh_error = refresh_error or f"action_decode:{type(exc).__name__}: {exc}"

            road_id = context.get("road_id")
            lane_id = context.get("lane_id")
            target_rank = None
            target_lane_in_range = False
            if lane_command not in (None, 0) and road_id and lane_id and callable(getattr(raw, "_driving_lanes", None)):
                try:
                    lanes = tuple(raw._driving_lanes(str(road_id)))
                    suffix = str(lane_id).rsplit("_", 1)[-1]
                    current_index = int(suffix) if suffix.isdigit() else None
                    if current_index is None or current_index not in lanes:
                        target_lane_reason = "current_lane_not_in_driving_lanes"
                    else:
                        sumo_offset = int(raw._sumo_lane_offset(int(lane_command)))
                        current_rank = lanes.index(current_index)
                        target_rank = current_rank + sumo_offset
                        if target_rank < 0 or target_rank >= len(lanes):
                            target_lane_reason = "target_lane_out_of_range"
                        else:
                            target_lane_in_range = True
                            target_lane_id = f"{road_id}_{int(lanes[target_rank])}"
                            target_context = dict(context, lane_id=target_lane_id)
                            target_facts = raw._behavior_route_lane_facts(target_context)
                            target_lane_reason = target_facts.get("route_lane_status_reason")
                except Exception as exc:
                    target_lane_reason = f"target_lane_classification_error:{type(exc).__name__}"
            elif lane_command == 0:
                target_lane_reason = "hold_command"

            current_label = current_facts.get("current_lane_reachability_label")
            target_label = target_facts.get("current_lane_reachability_label")
            executed_action, decision = route_veto_decision(
                original_action,
                context_known=context_known,
                context_fresh=context_fresh,
                current_road=str(road_id) if road_id is not None else None,
                has_next_edge=bool(current_facts.get("planned_next_edge")),
                current_lane_label=current_label,
                target_lane_label=target_label,
                lane_command=lane_command,
                target_lane_id=target_lane_id,
                target_lane_reason=target_lane_reason,
                target_lane_in_range=target_lane_in_range if lane_command not in (None, 0) else True,
            )

            row = {
                "schema_version": "goalonly_route_lane_veto_actions_v1",
                "arm": "route_lane_veto",
                "episode_index": self._episode_index,
                "episode_seed": self._episode_seed,
                "traffic": self._traffic_meta,
                "decision_step": self._decision,
                "raw_step_before_action": current_raw,
                "context_sample_raw_step": sampled_raw,
                "context_age_raw_steps": age,
                "context_refresh_error": refresh_error,
                "inner_step_refresh_consumed": False,
                "route_context": context,
                "current_route_facts": current_facts,
                "target_route_facts": target_facts,
                **decision,
            }
            refresh_context = (
                _consume_one_step_refresh(raw)
                if route_helpers_present and refresh_error is None
                else None
            )
            refresh_state = {"suppressed": False}
            try:
                if refresh_context is not None:
                    with refresh_context as refresh_state:
                        observation, reward, terminated, truncated, info = self.env.step(executed_action)
                else:
                    observation, reward, terminated, truncated, info = self.env.step(executed_action)
            except BaseException as exc:
                row["step_error"] = f"{type(exc).__name__}: {exc}"
                row["inner_step_refresh_consumed"] = bool(refresh_state.get("suppressed"))
                self._write_row(row)
                raise
            info = info if isinstance(info, dict) else {}
            row.update(
                {
                    "inner_step_refresh_consumed": bool(refresh_state.get("suppressed")),
                    "reward_returned_unchanged": _jsonable(reward),
                    "terminated_returned_unchanged": bool(terminated),
                    "truncated_returned_unchanged": bool(truncated),
                    "lane_change_request_sent": info.get("lane_change_applied"),
                    "lane_control_request_status": info.get("lane_control_request_status"),
                    "lane_control_request_reason": info.get("lane_control_request_reason"),
                    "previous_decision_transition": info.get("actual_lane_transition_since_previous_decision"),
                    "returned_current_lane_id": info.get("current_lane_id"),
                    "returned_current_lane_can_reach_next_edge": info.get("current_lane_can_reach_next_edge"),
                }
            )
            info["causal_intervention"] = decision
            self._write_row(row)
            return observation, reward, terminated, truncated, info

        def close(self):
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()
            return super().close()

    def add_eval_wrapper(env, method):
        env = original_wrap(env, method)
        if arm == "route_lane_veto":
            return RouteLaneVetoWrapper(env)
        return ReferenceTrafficIndexWrapper(env)

    d1._wrap_route_reachability_env = add_eval_wrapper
    return lambda: setattr(d1, "_wrap_route_reachability_env", original_wrap)


def _worker_main(args) -> int:
    arm = args.worker_arm
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    run_root = Path(args.run_root).resolve()
    arm_dir = run_root / ARM_DIRS[arm]
    if arm_dir.exists():
        raise FileExistsError(f"arm output already exists: {arm_dir}")
    arm_dir.mkdir(parents=True)
    status_path = arm_dir / "worker_status.json"
    _write_json(status_path, {"status": "running", "arm": arm, "pid": os.getpid(), "started_unix": time.time()})
    d1 = _load_d1()
    _apply_original_eval_protocol(d1)
    d1.base.RESULT_ROOT = FAST_DEVELOPER
    d1.RESULT_ROOT = FAST_DEVELOPER
    parent = d1.PARENT[METHOD]
    d1.base.SEED_START[parent] = int(args.seed_start)
    d1.EVAL_EPISODES_TOTAL = int(args.episodes)
    d1.SMOKE_EVAL_EPISODES = int(args.episodes)
    restore_wrapper = _install_eval_env_wrapper(
        d1,
        arm,
        arm_dir,
        smoke_episode_index=(
            int(args.smoke_episode_index)
            if args.smoke and args.smoke_episode_index is not None else None
        ),
    )

    fingerprints: list[dict[str, Any]] = []
    loaded_models: list[Any] = []
    restore_loader = _install_loader_capture(d1, arm, fingerprints, loaded_models)
    try:
        behavior = True
        probes = arm != "topology_goal_off"
        critical = arm != "topology_goal_off"
        if args.smoke and arm == "route_lane_veto" and args.smoke_episode_index is None:
            raise ValueError("route-veto functional smoke requires the reference timeout episode index")
        result = d1._evaluate_d1(
            METHOD,
            arm_dir,
            REFERENCE_MODEL,
            smoke=False,
            behavior_diagnostics=behavior,
            policy_shadow_probes=probes,
            policy_shadow_eval_max_per_episode=3,
            policy_shadow_critical_eval_sample=critical,
        )
        for model in loaded_models:
            fingerprints.append({"phase": "after_eval", **_state_fingerprint(model)})
        before_rows = [row for row in fingerprints if row.get("phase") == "before_eval"]
        after_rows = [row for row in fingerprints if row.get("phase") == "after_eval"]
        if len(before_rows) != 1 or len(after_rows) != 1:
            raise RuntimeError(
                "expected exactly one frozen model fingerprint before and after evaluation; "
                f"got before={len(before_rows)} after={len(after_rows)}"
            )
        if (
            before_rows[0].get("sha256") != after_rows[0].get("sha256")
            or before_rows[0].get("updates") != after_rows[0].get("updates")
        ):
            raise RuntimeError(
                "frozen evaluation changed model parameters/buffers or update count: "
                f"before={before_rows[0]} after={after_rows[0]}"
            )
        extra = {
            "diagnostic_experiment": "sorted_goalonly_causal_v1",
            "causal_arm": arm,
            "base_checkpoint_sha256": _sha256(REFERENCE_MODEL),
            "eval_device": "cpu",
            "learning_or_replay_updates": 0,
            "input_reward_observation_done_protocol": "unchanged D1 evaluator; no reward/obs/terminal edits",
            "policy_shadow_probes": probes,
            "policy_shadow_critical_eval_sample": critical,
            "policy_shadow_eval_max_per_episode": 3,
            "route_veto_sidecar": (
                str((arm_dir / "diagnostics" / "eval" / "route_lane_veto_actions.jsonl").resolve())
                if arm == "route_lane_veto" else None
            ),
            "model_policy_fingerprint": fingerprints,
        }
        result["identity"].update(extra)
        result["identity"]["checkpoint_sha256"] = _sha256(REFERENCE_MODEL)
        d1.base._write_json_atomic(arm_dir / "evaluation_results.json", result)
        status = {
            "status": "complete",
            "arm": arm,
            "pid": os.getpid(),
            "episodes": len(result.get("episode_records", [])),
            "checkpoint_sha256": result["identity"]["checkpoint_sha256"],
            "completed_unix": time.time(),
        }
        _write_json(status_path, status)
        return 0
    except BaseException as exc:
        _write_json(
            status_path,
            {
                "status": "failed",
                "arm": arm,
                "pid": os.getpid(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "failed_unix": time.time(),
            },
        )
        raise
    finally:
        restore_loader()
        restore_wrapper()


def _outcome(row: dict[str, Any]) -> str:
    flags = [
        name for name in ("success", "collision", "timeout", "off_route")
        if bool(row.get(name))
    ]
    if len(flags) == 1:
        return flags[0]
    if not flags:
        return "unknown"
    return "multi:" + "+".join(flags)


def _pair_results(run_root: Path) -> dict[str, Any]:
    outputs = {}
    traffic_by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in ARMS:
        path = run_root / ARM_DIRS[arm] / "evaluation_results.json"
        if path.is_file():
            outputs[arm] = json.loads(path.read_text(encoding="utf-8"))
        traffic_path = run_root / ARM_DIRS[arm] / "diagnostics" / "eval" / "traffic_selection.jsonl"
        if traffic_path.is_file():
            selected = {}
            for line_number, line in enumerate(traffic_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                seed = item.get("episode_seed")
                if seed is None:
                    continue
                seed = int(seed)
                if seed in selected:
                    raise ValueError(f"duplicate traffic selection for {arm} seed {seed} at line {line_number}")
                selected[seed] = item
            traffic_by_arm[arm] = selected
    control_rows = outputs.get("frozen_control", {}).get("episode_records", [])
    pairs = {}
    for arm, result in outputs.items():
        rows_by_seed = {}
        for row in result.get("episode_records", []):
            seed = row.get("episode_seed", row.get("seed"))
            if seed is not None:
                rows_by_seed[int(seed)] = row
        matrix: dict[str, dict[str, int]] = {}
        matched = 0
        mismatches = []
        traffic_checked = 0
        traffic_unknown = []
        traffic_mismatches = []
        for reference in control_rows:
            seed = reference.get("episode_seed", reference.get("seed"))
            if seed is None or int(seed) not in rows_by_seed:
                continue
            seed = int(seed)
            candidate = rows_by_seed[seed]
            matched += 1
            left = _outcome(reference)
            right = _outcome(candidate)
            matrix.setdefault(left, {}).setdefault(right, 0)
            matrix[left][right] += 1
            left_traffic = traffic_by_arm.get("frozen_control", {}).get(seed)
            right_traffic = traffic_by_arm.get(arm, {}).get(seed)
            if left_traffic is None or right_traffic is None:
                traffic_unknown.append(seed)
            else:
                left_key = (
                    left_traffic.get("traffic_index_before_reset"),
                    left_traffic.get("traffic_roll"),
                    left_traffic.get("selected_traffic_sha256"),
                )
                right_key = (
                    right_traffic.get("traffic_index_before_reset"),
                    right_traffic.get("traffic_roll"),
                    right_traffic.get("selected_traffic_sha256"),
                )
                if any(value is None for value in left_key + right_key):
                    traffic_unknown.append(seed)
                else:
                    traffic_checked += 1
                    if left_key != right_key:
                        traffic_mismatches.append(
                            {"seed": seed, "control": left_key, "candidate": right_key}
                        )
        if arm != "frozen_control":
            pairs[arm] = {
                "reference_arm": "frozen_control",
                "matched_seed_pairs": matched,
                "outcome_matrix_reference_rows_candidate_columns": matrix,
                "traffic_identity": {
                    "fully_checked_pairs": traffic_checked,
                    "unknown_pairs": traffic_unknown,
                    "mismatches": traffic_mismatches,
                    "identity_fields": [
                        "episode_seed",
                        "traffic_index_before_reset",
                        "traffic_roll",
                        "selected_traffic_sha256",
                    ],
                },
                "interpretation": "paired frozen-policy diagnostic; does not identify performance effects across training seeds",
            }
    return {"schema_version": "sorted_goalonly_causal_pairs_v1", "pairs_vs_frozen_control": pairs}


def _run_coordinator(args) -> int:
    run_root = Path(args.run_root).expanduser().resolve()
    plan = _preflight(run_root, smoke=bool(args.smoke))
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if not args.start:
        raise ValueError("pass --dry-run to preview or --start to execute; no implicit start")
    run_root.mkdir(parents=True, exist_ok=False)
    if args.smoke:
        seed, episode_index = _reference_smoke_case()
        episodes = 1
        seed_start = seed
        smoke_index = episode_index
    else:
        episodes = EVAL_EPISODES
        seed_start = EVAL_SEED_START
        smoke_index = None
    plan.update(
        {
            "status": "running",
            "supervisor_pid": os.getpid(),
            "started_unix": time.time(),
            "formal_or_smoke": "smoke_functional_only" if args.smoke else "formal_diagnostic",
            "seed_start": seed_start,
            "episodes_per_arm": episodes,
            "no_training": True,
            "resume": False,
            "smoke_seed_episode_index": smoke_index,
            "workers": {},
        }
    )
    _write_json(run_root / "experiment_manifest.json", plan)
    _write_json(run_root / "launcher_status.json", {"status": "running", "pid": os.getpid(), "workers": {}})
    script = Path(__file__).resolve()

    def launch(arm: str):
        arm_dir = run_root / ARM_DIRS[arm]
        stdout_path = run_root / f"{ARM_DIRS[arm]}.stdout.log"
        stderr_path = run_root / f"{ARM_DIRS[arm]}.stderr.log"
        command = [
            sys.executable, str(script), "--worker-arm", arm,
            "--run-root", str(run_root),
            "--episodes", str(episodes),
            "--seed-start", str(seed_start),
        ]
        if args.smoke:
            command.extend(["--smoke", "--smoke-episode-index", str(smoke_index)])
        _write_json(run_root / f"{ARM_DIRS[arm]}.command.json", {"command": command, "cwd": str(PROJECT_ROOT), "stdout": str(stdout_path), "stderr": str(stderr_path)})
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with stdout_path.open("w", encoding="utf-8", newline="\n") as out, stderr_path.open("w", encoding="utf-8", newline="\n") as err:
            proc = subprocess.Popen(command, cwd=str(PROJECT_ROOT), stdout=out, stderr=err, creationflags=flags)
            with status_lock:
                live_status["workers"][arm] = {
                    "pid": proc.pid,
                    "status": "running",
                    "command": command,
                    "stdout": str(stdout_path.resolve()),
                    "stderr": str(stderr_path.resolve()),
                }
                _write_json(run_root / "launcher_status.json", live_status)
            code = proc.wait()
        return arm, proc.pid, code

    outcomes = {}
    status_lock = threading.Lock()
    live_status = {"status": "running", "pid": os.getpid(), "workers": {}}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(launch, arm): arm for arm in ARMS}
        for future in concurrent.futures.as_completed(futures):
            arm, pid, code = future.result()
            outcomes[arm] = {"pid": pid, "exit_code": code, "status": "complete" if code == 0 else "failed"}
            with status_lock:
                live_status["workers"][arm] = dict(live_status["workers"].get(arm, {}), **outcomes[arm])
                live_status["status"] = "failed" if code != 0 else "running"
                _write_json(run_root / "launcher_status.json", live_status)
    complete = all(item["exit_code"] == 0 for item in outcomes.values()) and len(outcomes) == len(ARMS)
    if complete:
        _write_json(run_root / "paired_outcomes.json", _pair_results(run_root))
    final_status = {
        "status": "complete" if complete else "failed",
        "pid": os.getpid(),
        "workers": outcomes,
        "completed_unix": time.time(),
        "paired_outcomes": str((run_root / "paired_outcomes.json").resolve()) if complete else None,
    }
    _write_json(run_root / "launcher_status.json", final_status)
    return 0 if complete else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="validate identities/configuration and print plan without creating the root")
    parser.add_argument("--start", action="store_true", help="create a fresh root and run the three evaluation arms")
    parser.add_argument("--smoke", action="store_true", help="one known reference timeout seed; functional only, not a result")
    parser.add_argument("--worker-arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES, help=argparse.SUPPRESS)
    parser.add_argument("--seed-start", type=int, default=EVAL_SEED_START, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-episode-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_arm:
        return _worker_main(args)
    if args.smoke and not args.dry_run and not args.start:
        parser.error("--smoke requires --dry-run or --start")
    if args.start and args.dry_run:
        parser.error("choose one of --dry-run and --start")
    if not args.dry_run and not args.start:
        parser.error("choose --dry-run or --start")
    return _run_coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
