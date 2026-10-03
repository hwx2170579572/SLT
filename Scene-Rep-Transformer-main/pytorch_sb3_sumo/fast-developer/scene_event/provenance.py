"""Independent run roots, source snapshots, and artifact identity."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from datetime import datetime, timezone
import zipfile

import numpy as np
import torch

from .diagnostics import json_value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def array_mapping_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key, array in sorted(arrays.items()):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Public map field {key} must be an ndarray")
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(json_value(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunArtifacts:
    def __init__(self, root: Path, config: object, kind: str):
        self.root = root.resolve()
        # Do not reuse a partial/failed run or race a second launcher. Preserve
        # failed roots as evidence and require an explicitly different retry.
        self.root.mkdir(parents=True, exist_ok=False)
        self.started = utc_now()
        self.kind = kind
        self.config = config
        save_json(self.root / "config.json", config.to_dict())
        self.status("initializing", raw_steps=0, decision_steps=0, updates=0)

    def status(self, state: str, **fields: object) -> None:
        save_json(self.root / "status.json", {"state": state, "kind": self.kind,
            "pid": os.getpid(), "started_utc": self.started, "updated_utc": utc_now(),
            "config_sha256": self.config.digest(), **fields})

    def snapshot(self, extra_files: list[Path] | None = None, *, training_started: bool = True) -> dict:
        fd_root = Path(__file__).resolve().parents[1]
        repository = fd_root.parents[1]
        paths = set((fd_root / "scene_event").glob("*.py"))
        paths.add(fd_root / "train_scene_event_v1.py")
        paths.update(extra_files or [])
        for module in tuple(sys.modules.values()):
            name = getattr(module, "__file__", None)
            if name:
                source = Path(name).resolve()
                if source.is_relative_to(repository) and source.suffix == ".py":
                    paths.add(source)
        manifest = []
        archive = self.root / "source_snapshot.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
            for path in sorted(paths):
                if not path.is_file():
                    continue
                relative = path.relative_to(repository).as_posix() if path.is_relative_to(repository) else "external/" + path.name
                manifest.append({"path": str(path), "archive_path": relative, "sha256": sha256(path)})
                output.write(path, relative)
        runtime = {"timestamp_utc": utc_now(), "python": sys.version, "executable": sys.executable,
            "command_line": sys.argv, "pid": os.getpid(), "parent_pid": os.getppid(),
            "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(), "requested_device": self.config.device,
            "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "sources": manifest, "archive_sha256": sha256(archive), "fresh_training": training_started,
            "training_started": training_started,
            "resume": False, "protocol": self.config.protocol}
        save_json(self.root / "runtime_provenance.json", runtime)
        return runtime
