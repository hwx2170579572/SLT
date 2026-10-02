from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_all_pending_promotions_v4_9_2_1 as registry
from tools import run_all_v4_9_2_1_promotion as automation
from tools import run_topo_v4_9_2_1_experiments as patched
from tools import run_topo_v4_9_2_experiments as frozen


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _evidence_root(
    tmp_path: Path,
    *,
    split: str,
    explicit: object = "missing",
    receipt: str | None = None,
    unlock: dict | None = None,
) -> Path:
    detailed = {
        "evaluation_split": split,
        "evaluation_provenance": {
            "evaluation_split": split,
            "traffic_partition": split,
        },
    }
    if explicit != "missing":
        detailed["formal_test_accessed"] = explicit
    _write(tmp_path / "paper_evaluation_detailed.json", detailed)
    _write(
        tmp_path / "arguments.json",
        {
            "requested_raw_steps": {
                "evaluation_split": split,
                "formal_unlock_receipt": receipt,
            },
            "formal_unlock": unlock,
        },
    )
    return tmp_path


def test_missing_redundant_marker_is_derived_for_promotion(tmp_path: Path) -> None:
    root = _evidence_root(tmp_path, split="validation")
    job = SimpleNamespace(stage="promotion", evaluation_split="validation")
    evidence = patched._recorded_formal_access(root, job)
    assert evidence["formal_test_accessed"] is False
    assert evidence["explicit_marker_present"] is False
    assert evidence["derivation"] == "split_and_unlock_provenance"


def test_explicit_marker_must_agree_with_provenance(tmp_path: Path) -> None:
    root = _evidence_root(tmp_path, split="validation", explicit=True)
    job = SimpleNamespace(stage="promotion", evaluation_split="validation")
    with pytest.raises(
        patched.V4921EngineeringError,
        match="explicit formal access marker contradicts",
    ):
        patched._recorded_formal_access(root, job)


def test_promotion_cannot_record_formal_unlock(tmp_path: Path) -> None:
    root = _evidence_root(
        tmp_path,
        split="validation",
        receipt="promotion_gate.json",
    )
    job = SimpleNamespace(stage="promotion", evaluation_split="validation")
    with pytest.raises(
        patched.V4921EngineeringError,
        match="promotion run records formal unlock receipt",
    ):
        patched._recorded_formal_access(root, job)


def test_formal_access_requires_test_split_and_verified_unlock(tmp_path: Path) -> None:
    root = _evidence_root(
        tmp_path,
        split="test",
        receipt="promotion_gate.json",
        unlock={"promotion_results_sha256": "abc"},
    )
    job = SimpleNamespace(stage="formal", evaluation_split="test")
    evidence = patched._recorded_formal_access(root, job)
    assert evidence["formal_test_accessed"] is True
    assert evidence["formal_unlock_receipt_present"] is True
    assert evidence["formal_unlock_binding_present"] is True


def test_patch_is_bound_to_unchanged_frozen_runner() -> None:
    assert frozen._sha256(Path(frozen.__file__)) == patched.FROZEN_RUNNER_SHA256


def test_wrapper_uses_patch_result_after_frozen_metadata_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        automation.frozen_automation,
        "run_all",
        lambda **_: (1, {"status": "promotion_jobs_incomplete_after_all_attempts"}),
    )
    monkeypatch.setattr(
        automation.patched,
        "revalidate_promotion",
        lambda: (
            0,
            {
                "promotion_matrix_complete": True,
                "promotion_gate_decision": "pass",
                "accepted_jobs": 12,
                "expected_jobs": 12,
            },
        ),
    )
    code, report = automation.run_all(
        device="cpu",
        report_path=tmp_path / "automation.json",
    )
    assert code == 0
    assert report["status"] == "promotion_gate_passed"
    assert report["frozen_training_automation_returncode"] == 1
    assert report["accepted_jobs"] == 12


def test_registry_runs_later_versions_after_failure(tmp_path: Path) -> None:
    calls: list[str] = []

    def first(*, device: str):
        calls.append(f"first:{device}")
        raise RuntimeError("failed")

    def second(*, device: str):
        calls.append(f"second:{device}")
        return 0, {
            "status": "promotion_gate_passed",
            "accepted_jobs": 12,
            "expected_jobs": 12,
            "promotion_gate_decision": "pass",
        }

    code, report = registry.run_registered(
        device="cpu",
        report_path=tmp_path / "registry.json",
        registry=(("first", first), ("second", second)),
    )
    assert calls == ["first:cpu", "second:cpu"]
    assert code == 1
    assert len(report["versions"]) == 2
    assert report["versions"][0]["status"] == "automation_exception"
    assert report["versions"][1]["status"] == "promotion_gate_passed"

