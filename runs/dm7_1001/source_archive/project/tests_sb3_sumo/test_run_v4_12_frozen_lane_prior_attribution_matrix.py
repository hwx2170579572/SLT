from __future__ import annotations

import json
from pathlib import Path

from tools import run_v4_12_frozen_lane_prior_attribution_matrix as subject


def test_matrix_attempts_later_cells_after_failure(tmp_path: Path) -> None:
    calls: list[str] = []
    cells = tuple(
        (f"D{index}", tmp_path / f"source{index}", tmp_path / f"out{index}")
        for index in range(1, 4)
    )

    def fake_evaluate(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        calls.append(output_dir.name)
        if output_dir.name == "out1":
            raise RuntimeError("first cell failed")
        output_dir.mkdir(parents=True)
        evaluation = {
            "scenario": "cross",
            "evaluation_provenance": {"scenario": "cross"},
            "source_summary": {"success_rate": 0.5},
            "ablation_summary": {"success_rate": 0.6},
            "ablation_minus_source": {"success_rate": 0.1},
        }
        evaluation_path = output_dir / "closed_loop_evaluation.json"
        evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
        (output_dir / "receipt.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "formal_test_accessed": False,
                    "evaluation_sha256": subject._sha256(evaluation_path),
                }
            ),
            encoding="utf-8",
        )
        return evaluation

    payload = subject.run_matrix(
        device="cpu",
        summary_path=tmp_path / "summary.json",
        run_evaluation=fake_evaluate,
        cells=cells,
    )
    assert calls == ["out1", "out2", "out3"]
    assert payload["accepted_cells"] == 2
    assert payload["failed_cells"] == 1
    assert payload["complete"] is False
    assert [row["status"] for row in payload["rows"]] == [
        "failed",
        "accepted",
        "accepted",
    ]
