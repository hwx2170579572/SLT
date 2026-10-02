from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "start_all_pending_promotions_v4_13_background.ps1"


def test_background_launcher_is_hidden_and_discovery_based() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "run_all_pending_promotions_v4_12.py" in source
    assert "-WindowStyle Hidden" in source
    assert "-RedirectStandardOutput" in source
    assert "-RedirectStandardError" in source
    assert "Get-Process -Id $previous.pid" in source
    assert "discovery_supports_current_and_future_versioned_pipelines" in source
    assert "failure_does_not_cancel_remaining_versions_or_jobs" in source
    assert "formal_stage_launched = $false" in source
