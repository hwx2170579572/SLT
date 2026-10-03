from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


FAST_DEVELOPER = Path(__file__).resolve().parents[1] / "fast-developer"
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

from scene_event.future_targets import ObservedFutureQueue  # noqa: E402


def _observation(x: float, *, actor_valid: bool = True) -> dict[str, np.ndarray]:
    history = np.zeros((1, 3, 4), dtype=np.float32)
    history[0, -1, :2] = (x, 0.0)
    valid = np.zeros((1, 3), dtype=bool)
    valid[0, -1] = actor_valid
    return {
        "actor_history": history,
        "history_valid": valid,
        "actor_valid": np.asarray([actor_valid], dtype=bool),
        "actor_lane_ptr": np.asarray([7], dtype=np.int32),
    }


def _sidecar(raw_tick: int, actor_key: str | None = "vehicle.0",
             frame_ticks: list[int] | None = None) -> dict[str, object]:
    result: dict[str, object] = {"raw_tick": raw_tick, "actor_keys": [actor_key]}
    if frame_ticks is not None:
        result["frame_raw_tick"] = np.asarray([frame_ticks], dtype=np.int32)
    return result


def _load_one_shard(root: Path) -> dict[str, np.ndarray]:
    paths = list(root.glob("factual_*.npz"))
    assert len(paths) == 1
    with np.load(paths[0], allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def test_future_targets_keep_missing_actor_samples_unknown_not_zero(tmp_path: Path) -> None:
    root = tmp_path / "targets"
    queue = ObservedFutureQueue(root, horizon_ticks=6, stride_ticks=3,
                                anchor_interval=1, shard_size=1)
    queue.observe(_observation(1.0), _sidecar(0), 0, 0, add_anchor=True)
    # Tracking identity is absent from later ordinary observations. A missing
    # track is censored/unknown, never labeled as a zero displacement.
    queue.observe(_observation(0.0, actor_valid=False), _sidecar(3, None), 0, 1,
                  add_anchor=False)
    queue.observe(_observation(0.0, actor_valid=False), _sidecar(6, None), 0, 2,
                  add_anchor=False)
    manifest = queue.close()
    shard = _load_one_shard(root)

    assert shard["future_valid"].tolist() == [[[False, False]]]
    assert np.all(shard["future_xy"] == 0.0)
    assert manifest["valid_actor_future_samples"] == 0
    assert manifest["censored_anchors"] == 1
    assert shard["censor_reason"].tolist() == ["horizon"]


def test_episode_end_keeps_observed_future_and_censors_unseen_offsets(tmp_path: Path) -> None:
    root = tmp_path / "targets"
    queue = ObservedFutureQueue(root, horizon_ticks=6, stride_ticks=3,
                                anchor_interval=1, shard_size=1)
    queue.observe(_observation(1.0), _sidecar(0), 5, 0, add_anchor=True)
    queue.observe(_observation(4.0), _sidecar(3), 5, 1, add_anchor=False)
    queue.end_episode("collision")
    manifest = queue.close()
    shard = _load_one_shard(root)

    assert shard["future_valid"].tolist() == [[[True, False]]]
    assert shard["future_xy"][0, 0, 0].tolist() == pytest.approx([4.0, 0.0])
    assert shard["future_observed_lane_ptr"][0, 0, 0] == 7
    assert shard["future_observed_lane_ptr"][0, 0, 1] == -1
    assert shard["censor_reason"].tolist() == ["collision"]
    assert manifest["valid_actor_future_samples"] == 1
    assert manifest["active_actor_future_slots"] == 2


def test_future_queue_refuses_to_join_across_episode_and_keeps_inputs_separate(tmp_path: Path) -> None:
    root = tmp_path / "targets"
    queue = ObservedFutureQueue(root, horizon_ticks=6, stride_ticks=3,
                                anchor_interval=1, shard_size=4)
    anchor_observation = _observation(2.0)
    queue.observe(anchor_observation, _sidecar(0), 11, 0, add_anchor=True)
    anchor_observation["actor_history"][0, -1, 0] = 999.0

    with pytest.raises(ValueError, match="censored before episode reset"):
        queue.observe(_observation(123.0), _sidecar(0), 12, 0, add_anchor=True)

    queue.end_episode("external_truncation")
    queue.observe(_observation(123.0), _sidecar(0), 12, 0, add_anchor=True)
    queue.end_episode("collection_cutoff")
    manifest = queue.close()
    shard = _load_one_shard(root)

    assert shard["episode"].tolist() == [11, 12]
    assert shard["censor_reason"].tolist() == ["external_truncation", "collection_cutoff"]
    assert shard["observation/actor_history"][0, 0, -1, 0] == pytest.approx(2.0)
    assert shard["future_xy"].shape[0] == 2
    assert "future_xy" not in {key.removeprefix("observation/") for key in shard if key.startswith("observation/")}
    assert manifest["input_contains_future"] is False
    assert manifest["hidden_route_queries"] == 0


def test_strict_future_queue_rejects_compressed_valid_history_axis(tmp_path: Path) -> None:
    queue = ObservedFutureQueue(tmp_path / "targets", horizon_ticks=6, stride_ticks=3,
                                anchor_interval=1, require_tick_axis=True)
    observation = _observation(1.0)
    observation["history_valid"][:] = True
    # At raw tick 3 a three-frame history must refer to ticks [1, 2, 3].
    # This packed/repeated axis skips physical tick 2 and must fail closed.
    with pytest.raises(ValueError, match="compressed or reordered"):
        queue.observe(observation, _sidecar(3, frame_ticks=[1, 3, 3]),
                      0, 0, add_anchor=True)


@pytest.mark.parametrize("invalid_slot_tick", [2, -1])
def test_strict_future_queue_keeps_valid_raw_axis_across_masked_gap(
        tmp_path: Path, invalid_slot_tick: int) -> None:
    queue = ObservedFutureQueue(tmp_path / "targets", horizon_ticks=6, stride_ticks=3,
                                anchor_interval=1, shard_size=1, require_tick_axis=True)
    queue.observe(_observation(1.0), _sidecar(0, frame_ticks=[-2, -1, 0]),
                  0, 0, add_anchor=True)

    future = _observation(4.0)
    future["history_valid"][:] = np.asarray([[True, False, True]])
    future["actor_history"][0, 0, :2] = (2.0, 0.0)
    future["actor_history"][0, 1, :2] = (999.0, 0.0)  # masked data is never a label
    future["actor_history"][0, 2, :2] = (4.0, 0.0)
    queue.observe(future, _sidecar(3, frame_ticks=[1, invalid_slot_tick, 3]),
                  0, 1, add_anchor=False)
    queue.end_episode("collision")
    manifest = queue.close()
    shard = _load_one_shard(tmp_path / "targets")

    assert shard["future_valid"].tolist() == [[[True, False]]]
    assert shard["future_xy"][0, 0, 0].tolist() == pytest.approx([4.0, 0.0])
    assert manifest["explicit_tick_axis_calls"] == 2
    assert manifest["missing_tick_axis_calls"] == 0
