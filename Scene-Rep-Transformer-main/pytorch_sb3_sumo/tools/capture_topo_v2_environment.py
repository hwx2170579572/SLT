"""Capture the exact runtime dependency receipt for the v2 experiment freeze."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium
import numpy
import stable_baselines3
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sumo_receipt() -> dict[str, Any]:
    executable = shutil.which("sumo")
    if executable is None:
        return {"executable": None, "version_output": None, "available": False}
    completed = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "available": completed.returncode == 0,
        "executable": str(Path(executable).resolve()),
        "returncode": completed.returncode,
        "version_output": (completed.stdout or completed.stderr).strip(),
    }


def build_receipt() -> dict[str, Any]:
    distributions = sorted(
        (
            {
                "name": distribution.metadata.get("Name") or "UNKNOWN",
                "version": distribution.version,
            }
            for distribution in importlib.metadata.distributions()
        ),
        key=lambda row: str(row["name"]).lower(),
    )
    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        cuda.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_0_name": torch.cuda.get_device_name(0),
                "device_0_total_memory_bytes": properties.total_memory,
                "device_0_compute_capability": [properties.major, properties.minor],
            }
        )
    return {
        "schema_version": "topology-temporal-v2-runtime-environment/v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_runtime": True,
        "project_root": str(PROJECT_ROOT),
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "conda": {
            "default_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "prefix": os.environ.get("CONDA_PREFIX"),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_processors": os.cpu_count(),
        },
        "key_packages": {
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__,
            "numpy": numpy.__version__,
            "pyyaml": yaml.__version__,
        },
        "cuda": cuda,
        "sumo": _sumo_receipt(),
        "relevant_environment": {
            "SUMO_HOME": os.environ.get("SUMO_HOME"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
        "installed_distributions": distributions,
        "installed_distribution_count": len(distributions),
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "topo_v2" / "runtime_environment.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_receipt()
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "python": payload["python"],
                "key_packages": payload["key_packages"],
                "cuda": payload["cuda"],
                "sumo": payload["sumo"],
                "installed_distribution_count": payload[
                    "installed_distribution_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
