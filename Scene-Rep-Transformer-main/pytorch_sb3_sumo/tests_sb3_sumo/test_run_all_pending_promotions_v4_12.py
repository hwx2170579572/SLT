from __future__ import annotations

from pathlib import Path

from tools import run_all_pending_promotions_v4_12 as subject


def test_latest_patch_supersedes_earlier_engineering_runners(monkeypatch) -> None:
    function = lambda **kwargs: (0, {})
    monkeypatch.setattr(
        subject.legacy,
        "discover_pipelines",
        lambda: (
            ("v4.11", function, Path("v411.json")),
            ("v4.11.1", function, Path("v4111.json")),
            ("v4.11.2", function, Path("v4112.json")),
            ("v4.12", function, Path("v412.json")),
        ),
    )
    selected, superseded = subject.latest_family_pipelines()
    assert [row[0] for row in selected] == ["v4.11.2", "v4.12"]
    assert superseded["v4.11"] == "v4.11.1"
    assert superseded["v4.11.1"] == "v4.11.2"


def test_pipeline_script_names_follow_discovered_version() -> None:
    assert subject._pipeline_script("v4.12") == "run_all_v4_12_pipeline.py"
    assert subject._pipeline_script("v4.11.2") == "run_all_v4_11_2_pipeline.py"


def test_current_v412_detached_pipeline_is_detected() -> None:
    pid = subject.running_pipeline_pid("v4.12")
    assert isinstance(pid, int) and pid > 0
