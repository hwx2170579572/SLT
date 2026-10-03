from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile

import pytest


FAST_DEVELOPER = Path(__file__).resolve().parents[1] / "fast-developer"
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

from scene_event.protocol import ExperimentConfig  # noqa: E402
from scene_event.provenance import RunArtifacts, sha256  # noqa: E402


def test_run_artifact_identity_snapshots_sources_and_rejects_root_reuse(tmp_path: Path) -> None:
    config = ExperimentConfig(device="cpu")
    root = tmp_path / "run"
    artifacts = RunArtifacts(root, config, "smoke")
    snapshot = artifacts.snapshot()

    assert snapshot["fresh_training"] is True
    assert snapshot["resume"] is False
    assert snapshot["protocol"] == config.protocol
    assert sha256(root / "source_snapshot.zip") == snapshot["archive_sha256"]
    assert (root / "status.json").is_file()
    assert json.loads((root / "config.json").read_text(encoding="utf-8"))["protocol"] == config.protocol

    sources = {Path(row["path"]).name: row for row in snapshot["sources"]}
    assert "provenance.py" in sources
    assert "trainer.py" in sources
    assert "train_scene_event_v1.py" in sources
    for row in sources.values():
        assert sha256(Path(row["path"])) == row["sha256"]

    with zipfile.ZipFile(root / "source_snapshot.zip") as archive:
        archived_names = set(archive.namelist())
    assert {row["archive_path"] for row in snapshot["sources"]} <= archived_names

    with pytest.raises(FileExistsError):
        RunArtifacts(root, config, "smoke")

