from __future__ import annotations

from tools import run_v4_12_lane_prior_matrix_after_process as subject


def test_wait_returns_immediately_for_nonexistent_pid() -> None:
    result = subject.wait_for_process(
        2_147_483_647,
        poll_seconds=0.001,
        max_wait_seconds=0.01,
    )
    assert result["wait_status"] == "process_exited"
    assert result["observed_alive"] is False
