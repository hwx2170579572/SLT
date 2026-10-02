"""Non-scientific probe for Windows background PyTorch process startup."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results_topo_v4_13_dev" / "engineering" / "background_torch_probe.json"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(
        {
            "status": "ok",
            "created_at_local": datetime.now().astimezone().isoformat(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
