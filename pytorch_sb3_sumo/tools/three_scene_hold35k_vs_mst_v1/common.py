"""Shared constants and environment construction for the three-scene curve study.

Two methods share the i5m6s100 high-density environment family so they observe
the exact same physical traffic:
  * hold35k (v4_8):  IndependentV2FiveBySixEnvV4V1  (adapter='v4_8')
  * mst_slt (base):  IndependentV2FiveBySixEnvV1    (adapter='base')

The new ``cross_left`` scenario was registered in the same paper/high-density
registries (see envs/sumo/paper_scenario_registry.py and high_density_env_v1.py),
so both adapters accept it unchanged.
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

SCENES = ("cross", "carla", "cross_left")
METHODS = ("hold35k", "mst_slt")

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
    return load_protocol()["scenarios"][scene]


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
