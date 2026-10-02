from __future__ import annotations

from tools import summarize_dual_protocol_evaluations as summary_module
from tools.summarize_dual_protocol_evaluations import collect


def _protocol() -> dict:
    return {
        "supported_methods": {
            "sac": {"scenarios": ["cross"]},
            "ppo": {"scenarios": ["cross"]},
        }
    }


def test_collect_accounts_for_absent_expected_runs(tmp_path) -> None:
    runs, quality = collect(
        tmp_path,
        _protocol(),
        profile="paper",
        methods=["sac", "ppo"],
        scenarios=["cross"],
        seeds=[0, 1],
    )

    assert runs == []
    assert len(quality) == 8
    assert {row["status"] for row in quality} == {"missing_run"}
    assert {row["protocol"] for row in quality} == {
        "frozen_80_20",
        "source_all",
    }


def test_collect_ignores_method_scenario_combinations_not_supported(tmp_path) -> None:
    protocol = _protocol()
    protocol["supported_methods"]["sac"]["scenarios"] = []

    runs, quality = collect(
        tmp_path,
        protocol,
        profile="paper",
        methods=["sac", "ppo"],
        scenarios=["cross"],
        seeds=[0],
    )

    assert runs == []
    assert len(quality) == 2
    assert all(row["run"] == "paper__ppo__cross__seed0" for row in quality)


def test_write_csv_uses_latest_fallback_when_primary_is_locked(
    tmp_path, monkeypatch
) -> None:
    primary = tmp_path / "runs_by_seed.csv"
    real_writer = summary_module._write_csv_contents

    def locked_primary(path, rows) -> None:
        if path == primary:
            raise PermissionError("locked by viewer")
        real_writer(path, rows)

    monkeypatch.setattr(summary_module, "_write_csv_contents", locked_primary)
    written = summary_module._write_csv(primary, [{"seed": 1, "success_rate": 0.5}])

    assert written == tmp_path / "runs_by_seed.latest.csv"
    assert written.read_text(encoding="utf-8").splitlines() == [
        "seed,success_rate",
        "1,0.5",
    ]
