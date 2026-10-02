"""Verify that no file present before the migration was modified."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PORT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PORT_ROOT.parent
MANIFEST = PORT_ROOT / "MIGRATION_ORIGINAL_SHA256.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing: list[str] = []
    changed: list[dict[str, object]] = []
    for expected in data["files"]:
        relative = expected["path"]
        path = SOURCE_ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual_length = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_length != expected["length"] or actual_hash != expected["sha256"]:
            changed.append(
                {
                    "path": relative,
                    "expected_length": expected["length"],
                    "actual_length": actual_length,
                    "expected_sha256": expected["sha256"],
                    "actual_sha256": actual_hash,
                }
            )
    result = {
        "source_root": str(SOURCE_ROOT),
        "manifest": str(MANIFEST),
        "baseline_files": len(data["files"]),
        "unchanged_files": len(data["files"]) - len(missing) - len(changed),
        "missing": missing,
        "changed": changed,
        "ok": not missing and not changed,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
