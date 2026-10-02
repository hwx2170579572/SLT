"""Reproducible, CPU-only support for isolated mechanism diagnostics."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "r48m2"
SCENES = ("cross", "carla")
MODEL_IDS = ("mst_slt", "v4_8_lr_half")


def cpu_environment(threads: int = 1) -> dict:
    if threads != 1:
        raise ValueError("This diagnostic protocol reserves one CPU thread.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    low_priority = None
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        low_priority = bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000))
    return {"device": "cpu", "threads": threads, "max_parallel_sumo": 1,
            "windows_below_normal_priority": low_priority}


def configure_torch(seed: int = 73) -> dict:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return {"torch": torch.__version__, "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads()}


def sha256(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_file(path: Path, with_hash: bool = True) -> dict:
    path = path.resolve()
    item = {"path": str(path), "exists": path.is_file()}
    if item["exists"]:
        stat = path.stat()
        item.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if with_hash:
            item["sha256"] = sha256(path)
    return item


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value) -> None:
    path = path.resolve()
    if OUTPUT.resolve() not in path.parents:
        raise ValueError("Diagnostic output must remain inside r48m2.")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)


def metadata() -> dict:
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "protocol": "v48-mechanisms-v2",
            "training_seed": 0, "not_training_seed_uncertainty": True}


def checkpoint_path(scene: str, model_id: str) -> Path:
    if scene not in SCENES or model_id not in MODEL_IDS:
        raise ValueError((scene, model_id))
    if model_id == "mst_slt":
        return ROOT / "results_hd_ss100_v2" / "comparison" / "runs" / f"hd_ss100_v2__comparison__mst_slt__{scene}__seed0" / "final_model.zip"
    return ROOT / "results_phase2_runtime_v2" / "screen" / f"v4_8__{scene}__lr_half__seed0" / "final_model.zip"


def probe_dataset_path(scene: str) -> Path:
    if scene not in SCENES:
        raise ValueError(scene)
    return ROOT / "r3m1" / "representation_probe" / scene / "dataset.npz"
