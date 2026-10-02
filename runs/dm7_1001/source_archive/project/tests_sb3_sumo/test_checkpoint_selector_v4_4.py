from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from configs import sb3_configs_v4_4 as config_v44
from tools.checkpoint_selector_v4_4 import select_checkpoint, sha256
from tools.train_sb3_v4_4 import (
    _seal_selector_then_construct_evaluation_env,
    _validate_formal_unlock,
)


def _checkpoint(path: Path, marker: str) -> Path:
    with ZipFile(path, "w") as archive:
        archive.writestr("marker.txt", marker)
    return path


def _summary(
    *,
    success: int,
    collision: int,
    off_route: int,
    timeout: int,
    mean_return: float,
    episodes: int = 12,
) -> dict:
    return {
        "episodes": episodes,
        "success_rate": success / episodes,
        "collision_rate": collision / episodes,
        "off_route_rate": off_route / episodes,
        "timeout_rate": timeout / episodes,
        "mean_return": mean_return,
    }


def _records(seed_start: int = 64_000, episodes: int = 12) -> list[dict]:
    return [
        {
            "episode": index,
            "seed": seed_start + index,
            "traffic_variant": f"train-{index % 3}",
        }
        for index in range(episodes)
    ]


def _candidate(
    path: Path,
    kind: str,
    summary: dict,
    *,
    records: list[dict] | None = None,
    partition: str = "train",
) -> dict:
    return {
        "checkpoint_kind": kind,
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256(path),
        "traffic_partition": partition,
        "calibration_seed_start": 64_000,
        "summary": summary,
        "episode_records": _records() if records is None else records,
        "calibration_result_sha256": kind[0] * 64,
    }


def test_selector_reproduces_t3_best_checkpoint_recovery(tmp_path: Path) -> None:
    best_path = _checkpoint(tmp_path / "best.zip", "best")
    final_path = _checkpoint(tmp_path / "final.zip", "final")
    candidates = [
        _candidate(
            best_path,
            "highest_training_success",
            _summary(
                success=10,
                collision=2,
                off_route=0,
                timeout=0,
                mean_return=2 / 3,
            ),
        ),
        _candidate(
            final_path,
            "exact_final",
            _summary(
                success=0,
                collision=2,
                off_route=0,
                timeout=10,
                mean_return=-1 / 6,
            ),
        ),
    ]
    receipt = select_checkpoint(candidates)
    assert receipt["selected_checkpoint_kind"] == "highest_training_success"
    assert receipt["selected_checkpoint_sha256"] == sha256(best_path)
    assert receipt["selection_partition"] == "train"
    assert receipt["validation_used_for_selection"] is False


def test_selector_complete_tie_prefers_exact_final(tmp_path: Path) -> None:
    best_path = _checkpoint(tmp_path / "best.zip", "best")
    final_path = _checkpoint(tmp_path / "final.zip", "final")
    tied = _summary(
        success=9,
        collision=3,
        off_route=0,
        timeout=0,
        mean_return=0.5,
    )
    receipt = select_checkpoint(
        [
            _candidate(best_path, "highest_training_success", tied),
            _candidate(final_path, "exact_final", tied),
        ]
    )
    assert receipt["selected_checkpoint_kind"] == "exact_final"
    assert receipt["selected_checkpoint_sha256"] == sha256(final_path)


@pytest.mark.parametrize("failure", ["pairing", "partition", "hash", "outcomes"])
def test_selector_rejects_protocol_or_integrity_drift(
    tmp_path: Path, failure: str
) -> None:
    best_path = _checkpoint(tmp_path / "best.zip", "best")
    final_path = _checkpoint(tmp_path / "final.zip", "final")
    valid = _summary(
        success=9,
        collision=3,
        off_route=0,
        timeout=0,
        mean_return=0.5,
    )
    best = _candidate(best_path, "highest_training_success", valid)
    final = _candidate(final_path, "exact_final", valid)
    if failure == "pairing":
        final["episode_records"][0]["traffic_variant"] = "different"
    elif failure == "partition":
        best["traffic_partition"] = "validation"
    elif failure == "hash":
        final["checkpoint_sha256"] = "0" * 64
    else:
        final["summary"] = _summary(
            success=9,
            collision=3,
            off_route=0,
            timeout=1,
            mean_return=0.5,
        )
    with pytest.raises(ValueError):
        select_checkpoint([best, final])


def test_missing_training_best_falls_back_to_exact_final(tmp_path: Path) -> None:
    final_path = _checkpoint(tmp_path / "final.zip", "final")
    final = _candidate(
        final_path,
        "exact_final",
        _summary(
            success=12,
            collision=0,
            off_route=0,
            timeout=0,
            mean_return=1.0,
        ),
    )
    receipt = select_checkpoint([final])
    assert receipt["candidate_count"] == 1
    assert receipt["selected_checkpoint_kind"] == "exact_final"


def test_selector_receipt_is_durable_before_evaluation_factory_call(
    tmp_path: Path,
) -> None:
    final_path = _checkpoint(tmp_path / "final.zip", "final")
    selector = select_checkpoint(
        [
            _candidate(
                final_path,
                "exact_final",
                _summary(
                    success=12,
                    collision=0,
                    off_route=0,
                    timeout=0,
                    mean_return=1.0,
                ),
            )
        ]
    )
    args = argparse.Namespace(
        experiment_contract_sha256="a" * 64,
        attribution_sha256="b" * 64,
        implementation_freeze_sha256="c" * 64,
    )
    sentinel = object()

    def instrumented_factory(received_args, *, evaluation=False):
        assert received_args is args
        assert evaluation is True
        receipt_path = tmp_path / "selector" / "receipt.json"
        assert receipt_path.is_file()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["selector_receipt_precedes_validation_environment"] is True
        assert receipt["selected_model_sha256"] == sha256(
            tmp_path / "selected_model.zip"
        )
        return sentinel

    selected, receipt_path, receipt_sha, env, constructed_at = (
        _seal_selector_then_construct_evaluation_env(
            selector=selector,
            run_dir=tmp_path,
            args=args,
            env_factory=instrumented_factory,
        )
    )
    assert selected == tmp_path / "selected_model.zip"
    assert receipt_sha == sha256(receipt_path)
    assert env is sentinel
    assert constructed_at >= selector["sealed_at_utc"]


def test_v4_4_registry_delegates_training_unchanged_and_adds_metadata(
    monkeypatch,
) -> None:
    calls: list[tuple[str, dict]] = []

    class FakeModel:
        v4_method_metadata = {
            "implementation_id": "parent",
            "deterministic_lane_decoder": "argmax_feasible_min_target_twin_q",
        }

    def parent_factory(parent, env, **kwargs):
        del env
        calls.append((parent, kwargs))
        return FakeModel()

    monkeypatch.setattr(config_v44, "make_model_v4_3", parent_factory)
    model = config_v44.make_model_v4_4(
        "topo_v4_4_train_only_selector",
        object(),
        scenario="cross",
        seed=3,
        learning_starts=123,
        batch_size=7,
    )
    assert calls[0][0] == "topo_v4_3_target_critic_decoder"
    assert calls[0][1]["scenario"] == "cross"
    assert calls[0][1]["seed"] == 3
    assert calls[0][1]["learning_starts"] == 123
    assert calls[0][1]["batch_size"] == 7
    assert model.v4_method_metadata["single_change"] == (
        "train_only_deployment_aware_checkpoint_selector"
    )
    assert model.v4_method_metadata["training_unchanged_from_v4_3"] is True
    assert model.v4_method_metadata["validation_used_for_checkpoint_selection"] is False


def test_formal_unlock_requires_hash_bound_promotion_receipt(tmp_path: Path) -> None:
    arguments = argparse.Namespace(
        evaluation_split="test",
        algo="topo_v4_4_train_only_selector",
        experiment_contract_sha256="a" * 64,
        implementation_freeze_sha256="b" * 64,
        formal_unlock_receipt=None,
    )
    with pytest.raises(ValueError, match="formal test locked"):
        _validate_formal_unlock(arguments)

    receipt = tmp_path / "promotion_gate.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.4.promotion-gate/v1",
                "decision": "pass",
                "experiment_contract_sha256": "a" * 64,
                "implementation_freeze_sha256": "b" * 64,
                "formal_algorithms": list(config_v44.V44_ALGORITHMS),
                "promotion_results_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    arguments.formal_unlock_receipt = receipt
    unlocked = _validate_formal_unlock(arguments)
    assert unlocked is not None
    assert unlocked["sha256"] == sha256(receipt)
