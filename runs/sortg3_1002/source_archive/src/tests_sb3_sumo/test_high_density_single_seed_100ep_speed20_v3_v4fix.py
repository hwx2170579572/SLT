from __future__ import annotations

import json
from pathlib import Path

from tools import train_high_density_single_seed_100ep_speed20_v3_v4fix as v4fix


def test_v4fix_short_path_is_stable_and_experiment_specific() -> None:
    long_path = (
        v4fix.PROJECT_ROOT
        / "results_hd_ss100_s20_v3"
        / "comparison"
        / "runs"
        / "hd_ss100_s20_v3__comparison__v4_8__roundabout__seed0"
        / "tensorboard"
    )
    first_run, first = v4fix._short_tensorboard_directory(str(long_path))
    second_run, second = v4fix._short_tensorboard_directory(str(long_path))

    assert first_run == long_path.parent.resolve()
    assert second_run == first_run
    assert first == second
    assert first.parent == v4fix.SHORT_TENSORBOARD_ROOT.resolve()
    assert len(first.name) == 16
    assert "speed20" not in first.name
    assert len(str(first)) < len(str(long_path))


def test_v4fix_redirect_receipt_records_no_experimental_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "run" / "tensorboard"
    target = tmp_path / "short" / "0123456789abcdef"
    run_dir = source.parent
    run_dir.mkdir(parents=True)
    v4fix._write_redirect_receipt(run_dir, source, target)

    receipt = json.loads(
        (run_dir / v4fix.REDIRECT_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    assert receipt["experimental_contract_changed"] is False
    assert receipt["original_tensorboard_log"] == str(source)
    assert receipt["redirected_tensorboard_log"] == str(target)
