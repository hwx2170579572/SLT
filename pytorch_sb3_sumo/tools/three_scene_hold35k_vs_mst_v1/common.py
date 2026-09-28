"""Shared constants and environment construction for the three-scene curve study.

Two methods share the i5m6s100 high-density environment family so they observe
the exact same physical traffic:
  * hold35k (v4_8):  IndependentV2FiveBySixEnvV4V1  (adapter='v4_8')
  * mst_slt (base):  IndependentV2FiveBySixEnvV1    (adapter='base')

The new ``cross_left`` scenario was registered in the same paper/high-density
registries (see envs/sumo/paper_scenario_registry.py and high_density_env_v1.py),
so both adapters accept it unchanged.

``cross_left_unreg`` is a fourth *trained* scene (uncontrolled 4-way, 70 m lanes,
all three turning behaviours on every approach).  It runs at its base density
(0.2 veh/s per approach) with ``vehicle_scale=1.0`` (empty additive overlay).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results_three_scene_v1"
PROTOCOL_PATH = (
    ROOT / "experiments" / "three_scene_hold35k_vs_mst_50ep_v1" / "protocol.json"
)

SCENES = ("cross", "carla", "cross_left", "cross_left_unreg")
METHODS = ("hold35k", "mst_slt")

# --------------------------------------------------------------------------- #
# New SUMO scenarios (this project) added to the *visual / video* pipeline as
# zero-shot transfers.  They have no trained checkpoints of their own, so their
# visual verification and video capture load the source scene's final model via
# ``ZERO_SHOT_SOURCE``.
# --------------------------------------------------------------------------- #
EXTRA_SCENES = ("merge", "intersection")
ALL_SCENES = SCENES + EXTRA_SCENES
ZERO_SHOT_SOURCE = {
    "merge": "cross",          # single-lane highway + on-ramp merge -> double merge
    "intersection": "cross_left",  # unsignalised unprotected left -> unprotected left
}

# Density regime for scenes added *after* the sealed protocol.json.  ``merge`` /
# ``intersection`` are zero-shot visual-only and inherit the source scene's
# regime; ``cross_left_unreg`` is a *trained* fourth scene whose base demand
# (0.2 veh/s per approach) is already its experiment density, so it uses a
# no-overlay ``vehicle_scale`` of 1.0.
NEW_SCENE_DENSITY: dict[str, dict[str, Any]] = {
    "merge": {
        "display_label": "Single-lane highway + on-ramp merge (urban)",
        "scenario_id": "merge_zero_shot_visual_v1",
        "traffic_mode": "high_density_v2",
        "vehicle_scale": 1.5,
        "pedestrian_scale": 1.0,
        "clone_depart_jitter_seconds": [8.0, 10.0],
        "rationale": (
            "Mirrors the cross (double merge) regime: 50% extra social vehicles "
            "with delayed clones so ego departs before merge pressure builds."
        ),
    },
    "intersection": {
        "display_label": "4-way unsignalised intersection left turn (urban)",
        "scenario_id": "intersection_zero_shot_visual_v1",
        "traffic_mode": "high_density_calibrated_v1",
        "vehicle_scale": 1.4,
        "pedestrian_scale": 1.0,
        "clone_depart_jitter_seconds": [4.0, 6.0],
        "rationale": (
            "Mirrors the cross_left (unprotected left turn) regime: 40% extra "
            "social vehicles with 4-6 s clone jitter, keeping the dense-but-"
            "solvable band between left_turn and cross."
        ),
    },
    "cross_left_unreg": {
        "display_label": "4-way uncontrolled cross, 70 m lanes, all turns (0.2 veh/s/arm)",
        "scenario_id": "cross_left_unreg_base_density_v1",
        "traffic_mode": "base_density_no_overlay_v1",
        "vehicle_scale": 1.0,
        "pedestrian_scale": 1.0,
        "clone_depart_jitter_seconds": [0.0, 0.0],
        "rationale": (
            "Authored at 0.2 veh/s per approach (16 flows x 180 veh/h = 2880 "
            "veh/h total), already above every other scene's overlay-applied "
            "demand.  vehicle_scale=1.0 keeps the additive overlay empty so the "
            "scenario runs at its base density; any overlay would push an "
            "already-saturated uncontrolled junction into gridlock."
        ),
    },
}

RAW_TRAINING_STEPS = 50000
LEARNING_STARTS_RAW_STEPS = 5000
CHECKPOINT_FREQUENCY = 2000
RAW_STEPS = tuple(
    range(CHECKPOINT_FREQUENCY, RAW_TRAINING_STEPS + 1, CHECKPOINT_FREQUENCY)
)
EVAL_EPISODES = 50

HOLD35K_CANDIDATE = "tau0025_floor2e5_hold35k"
HOLD35K_CHECKPOINT_PREFIX = "ckpt"
MST_CHECKPOINT_PREFIX = "scene_rep"

TRAIN_ROOT = RESULT_ROOT / "train"
OVERLAY_ROOT = RESULT_ROOT / "overlays_three_scene_v1"
EVAL_CURVE_ROOT = RESULT_ROOT / "eval_curve"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def seal(path: Path, value: Any) -> None:
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Immutable artifact differs: {path}")
    else:
        write(path, value)


@contextmanager
def lock(path: Path):
    """Fail fast on a live owner; recover only a demonstrably dead PID."""
    import psutil

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        owner = read(path)
        try:
            alive = (
                abs(psutil.Process(owner["pid"]).create_time() - owner["create_time"])
                < 0.01
            )
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError(f"Already running (PID {owner['pid']}): {path}")
        path.unlink()
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            dict(pid=os.getpid(), create_time=psutil.Process().create_time()), handle
        )
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


_protocol_cache: dict[str, Any] | None = None


def load_protocol() -> dict[str, Any]:
    global _protocol_cache
    if _protocol_cache is None:
        _protocol_cache = read(PROTOCOL_PATH)
    return _protocol_cache


def density(scene: str) -> dict[str, Any]:
    protocol_scenarios = load_protocol()["scenarios"]
    if scene in protocol_scenarios:
        return protocol_scenarios[scene]
    if scene in NEW_SCENE_DENSITY:
        return NEW_SCENE_DENSITY[scene]
    raise ValueError(
        f"No density definition for scenario {scene!r}; known: "
        f"{sorted(set(protocol_scenarios) | set(NEW_SCENE_DENSITY))}"
    )


def checkpoint_source_scene(scene: str) -> str:
    """Return the scene whose trained checkpoints back a zero-shot transfer.

    ``merge`` / ``intersection`` have no trained models of their own; their
    visual/video playback loads the source scene's final model instead.
    """
    return ZERO_SHOT_SOURCE.get(scene, scene)


def method_scenario_cell(method: str, scene: str) -> str:
    return f"{method}__{scene}"


def train_dir(method: str, scene: str, smoke: bool = False) -> Path:
    return (
        RESULT_ROOT / ("smoke/train" if smoke else "train") / method_scenario_cell(method, scene)
    )


def hold35k_checkpoint(scene: str, raw_step: int) -> Path:
    return (
        train_dir("hold35k", scene)
        / "checkpoints"
        / f"{HOLD35K_CHECKPOINT_PREFIX}_raw_{raw_step}_steps.zip"
    )


def mst_checkpoint(scene: str, raw_step: int) -> Path:
    return (
        train_dir("mst_slt", scene)
        / "checkpoints"
        / f"{MST_CHECKPOINT_PREFIX}_raw_{raw_step}_steps.zip"
    )


def _environment_namespace(scene: str) -> argparse.Namespace:
    return argparse.Namespace(
        scenario=scene,
        history_steps=10,
        neighbors=5,
        path_length=10,
        action_repeat=3,
        discount=0.99,
        ego_control_profile="direct",
        episode_limit_profile="source",
        gui=False,
        evaluation_split="validation",
    )


def _build_env_factory(
    adapter: str, scene: str, overlay_root: Path
) -> Callable[..., Any]:
    from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory

    return _make_environment_factory(
        adapter=adapter,
        density=density(scene),
        overlay_root=overlay_root,
    )


def make_hold35k_env(
    scene: str, namespace: str, *, training: bool = False
) -> Any:
    """Construct the v4_8 high-density env under a named overlay namespace."""
    factory = _build_env_factory(
        "v4_8", scene, OVERLAY_ROOT / "hold35k" / f"ns_{namespace}"
    )
    return factory(_environment_namespace(scene), evaluation=not training)


def make_mst_env_factory(scene: str) -> Callable[..., Any]:
    """Return the base-adapter factory consumed by train_sb3 / evaluation."""
    return _build_env_factory(
        "base", scene, OVERLAY_ROOT / "mst_slt" / "seed_0"
    )


# --------------------------------------------------------------------------- #
# Six comparison baselines (HSAC has mlp/lstm variants -> seven method names).
# These are registered alongside ``METHODS``; their training runners produce
# checkpoints under the same ``train/{method}__{scene}/checkpoints`` layout.
# --------------------------------------------------------------------------- #
BASELINE_METHODS = (
    "hsac_mlp",
    "hsac_lstm",
    "gnn_sac",
    "hybrid_dt",
    "decision_transformer",
    "hyar",
    "mst",  # MST without SLT
)
ALL_METHODS = METHODS + BASELINE_METHODS

# Sequence / latent baselines train offline (gradient steps) and checkpoint as
# torch ``.pt`` files; the SB3-native baselines checkpoint as ``.zip`` archives.
SEQUENCE_BASELINE_METHODS = ("hybrid_dt", "decision_transformer", "hyar")

# Checkpoint filename prefix per baseline (the training runner must match it).
BASELINE_CHECKPOINT_PREFIX = {
    "hsac_mlp": "hsac_mlp",
    "hsac_lstm": "hsac_lstm",
    "gnn_sac": "gnn_sac",
    "hybrid_dt": "hybrid_dt",
    "decision_transformer": "decision_transformer",
    "hyar": "hyar",
    "mst": "scene_rep",
}

# Environment contract per baseline: v4_8 exposes ``lane_action_mask`` for the
# hybrid action head; base is the continuous Box(2) contract shared with mst_slt.
BASELINE_ENV_CONTRACT = {
    "hsac_mlp": "v4_8",
    "hsac_lstm": "v4_8",
    "gnn_sac": "base",
    "hybrid_dt": "v4_8",
    "decision_transformer": "base",
    "hyar": "v4_8",
    "mst": "base",
}


def baseline_checkpoint(method: str, scene: str, raw_step: int) -> Path:
    """Checkpoint path for a baseline method under the shared train layout.

    Sequence baselines (``hybrid_dt`` / ``decision_transformer`` / ``hyar``)
    checkpoint as ``.pt``; the SB3-native baselines checkpoint as ``.zip``.
    ``raw_step`` is the training-step index (environment steps for SB3-native,
    gradient steps for sequence baselines).
    """
    prefix = BASELINE_CHECKPOINT_PREFIX[method]
    suffix = ".pt" if method in SEQUENCE_BASELINE_METHODS else ".zip"
    return (
        train_dir(method, scene)
        / "checkpoints"
        / f"{prefix}_raw_{raw_step}_steps{suffix}"
    )


def baseline_data_source(method: str) -> str:
    """Return the trained policy that seeds offline data for a sequence baseline.

    v4_8 hybrid-action baselines (Hybrid-DT / HyAR) imitate HSAC-MLP, the plain
    hybrid-head baseline on the same v4_8 contract; the base continuous-action
    Decision Transformer imitates GNN+SAC, the continuous-head baseline.  Each
    sequence baseline therefore inherits data from a source in its own action-
    head / encoder family rather than from the TASAC / MST+SLT reference points.
    """
    if method in ("hybrid_dt", "hyar"):
        return "hsac_mlp"
    if method == "decision_transformer":
        return "gnn_sac"
    raise ValueError(f"{method!r} is not a sequence baseline")


def make_baseline_env_factory(method: str, scene: str) -> Callable[..., Any]:
    """Return the environment factory for a baseline method.

    Hybrid-head baselines (v4_8) reuse the hold35k adapter; continuous-head
    baselines (base) reuse the base adapter under a method-specific overlay.
    """
    contract = BASELINE_ENV_CONTRACT[method]
    if contract == "v4_8":
        return _build_env_factory("v4_8", scene, OVERLAY_ROOT / method / "seed_0")
    return _build_env_factory("base", scene, OVERLAY_ROOT / method / "seed_0")


def make_baseline_env(
    method: str, scene: str, namespace: str, *, training: bool = False
) -> Any:
    """Construct a baseline environment that pairs with its strong baseline.

    v4_8 baselines share the hold35k overlay so they observe the exact physical
    traffic of TASAC(hold35k); base baselines share the MST+SLT overlay.  This
    keeps each ablation arm on the identical traffic as the method it isolates.
    """
    contract = BASELINE_ENV_CONTRACT[method]
    if contract == "v4_8":
        return make_hold35k_env(scene, namespace, training=training)
    factory = _build_env_factory("base", scene, OVERLAY_ROOT / "mst_slt" / "seed_0")
    return factory(_environment_namespace(scene), evaluation=not training)
