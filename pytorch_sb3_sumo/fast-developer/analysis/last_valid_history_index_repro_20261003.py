"""Pure-tensor reproduction of count-minus-one history indexing.

This script does not import the environment, construct a policy, launch SUMO,
or train a model.  The validity rule intentionally mirrors
algos.sb3_torch.features._nonzero_mask: trajectory[..., 0] != 0.
"""

from __future__ import annotations

import torch


HISTORY = 10
STATE_DIM = 5  # x, y, heading, vx, vy


def make_history(valid_indices: list[int], *, zero_x_indices: set[int] | None = None) -> torch.Tensor:
    history = torch.zeros((HISTORY, STATE_DIM), dtype=torch.float32)
    zero_x_indices = zero_x_indices or set()
    for index in valid_indices:
        # Distinct position and speed values make stale-row selection visible.
        history[index] = torch.tensor(
            [100.0 + index, -20.0 - index, 0.01 * index, 10.0 + index, -5.0 - index]
        )
        if index in zero_x_indices:
            # Model a physically valid point on the x=0 axis. The first-coordinate
            # sentinel nevertheless treats it as padding, exactly like production.
            history[index, 0] = 0.0
    return history


def summarize(
    name: str,
    valid_indices: list[int],
    *,
    global_history_timestep: int,
    zero_x_indices: set[int] | None = None,
) -> None:
    history = make_history(valid_indices, zero_x_indices=zero_x_indices)

    # Production contract: values[..., 0] != 0 (not any feature != 0).
    valid = history[:, 0] != 0
    count = int(valid.sum().item())
    count_minus_one = max(count - 1, 0)
    time_indices = torch.arange(HISTORY)
    max_valid = int(torch.where(valid, time_indices, -1).max().item())
    has_valid = bool(valid.any().item())
    env_current = min(global_history_timestep, HISTORY) - 1

    selected = history[count_minus_one]
    selected_valid = bool(valid[count_minus_one].item())
    lag = max_valid - count_minus_one if has_valid else None
    lag_seconds = lag * 0.1 if lag is not None else None
    mask_string = "".join("1" if item else "0" for item in valid.tolist())

    def state_text(row: torch.Tensor) -> str:
        return f"x={row[0].item():.1f},vx={row[3].item():.1f}"

    print(
        f"{name}: source_mask={mask_string} count={count} "
        f"sum-1={count_minus_one} (valid={selected_valid}; {state_text(selected)}) "
        f"maxvalid={max_valid} "
        f"lag={lag if lag is not None else 'n/a'} frames/"
        f"{lag_seconds if lag_seconds is not None else 'n/a'} s "
        f"env_current={env_current}"
    )


def main() -> None:
    summarize("mature_leftpad_5", [5, 6, 7, 8, 9], global_history_timestep=10)
    summarize("mature_leftpad_4", [6, 7, 8, 9], global_history_timestep=10)
    summarize("mature_leftpad_6", [4, 5, 6, 7, 8, 9], global_history_timestep=10)
    summarize("early_both_sides_pad", [4, 5], global_history_timestep=6)
    summarize("rightpad_prefix_5", [0, 1, 2, 3, 4], global_history_timestep=5)
    summarize("all_valid", list(range(HISTORY)), global_history_timestep=10)
    summarize("gaps", [0, 3, 4, 8], global_history_timestep=10)
    summarize("all_zero_stationary_origin", [], global_history_timestep=10)
    summarize(
        "valid_x_zero_current_frame",
        [8, 9],
        global_history_timestep=10,
        zero_x_indices={9},
    )


if __name__ == "__main__":
    main()
