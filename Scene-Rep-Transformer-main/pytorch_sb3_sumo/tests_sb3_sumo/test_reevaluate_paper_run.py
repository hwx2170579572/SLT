from __future__ import annotations

import json

from tools import reevaluate_paper_run


def test_adopt_valid_primary_evaluation_preserves_evidence(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "paper__sac__cross__seed0"
    run_dir.mkdir()
    canonical_path = run_dir / "paper_evaluation_detailed.json"
    detailed = {
        "summary": {"episodes": 50, "success_rate": 0.8},
        "evaluation_provenance": {"contract_version": 1},
    }
    canonical_path.write_text(json.dumps(detailed), encoding="utf-8")
    monkeypatch.setattr(
        reevaluate_paper_run,
        "_existing_protocol_evaluation",
        lambda *args, **kwargs: detailed,
    )

    protocol_dir = run_dir / "protocol_evaluations" / "frozen_80_20"
    adopted = reevaluate_paper_run._adopt_valid_primary_evaluation(
        run_dir,
        protocol_dir,
        algorithm="sac",
        scenario="cross",
        traffic_protocol="frozen_80_20",
        episode_limit_profile="source",
        episodes=50,
        evaluation_seed_start=10_000,
        model_sha256="MODEL",
        checkpoint_audit_sha256="AUDIT",
    )

    assert adopted == detailed
    adopted_path = protocol_dir / "paper_evaluation_detailed.json"
    assert adopted_path.read_bytes() == canonical_path.read_bytes()
    receipt = json.loads(
        (protocol_dir / "evaluation_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["kind"] == "adopted_valid_primary_evaluation_without_rerun"
    assert receipt["training_was_not_resumed_or_modified"] is True
    assert receipt["evaluation_was_not_rerun"] is True
    assert receipt["canonical_detailed_sha256"] == receipt["adopted_detailed_sha256"]


def test_adopt_primary_rejects_invalid_canonical(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "paper_evaluation_detailed.json").write_text(
        "{}", encoding="utf-8"
    )
    monkeypatch.setattr(
        reevaluate_paper_run,
        "_existing_protocol_evaluation",
        lambda *args, **kwargs: None,
    )

    adopted = reevaluate_paper_run._adopt_valid_primary_evaluation(
        run_dir,
        run_dir / "protocol_evaluations" / "frozen_80_20",
        algorithm="sac",
        scenario="cross",
        traffic_protocol="frozen_80_20",
        episode_limit_profile="source",
        episodes=50,
        evaluation_seed_start=10_000,
        model_sha256="MODEL",
        checkpoint_audit_sha256="AUDIT",
    )

    assert adopted is None
    assert not (run_dir / "protocol_evaluations").exists()
