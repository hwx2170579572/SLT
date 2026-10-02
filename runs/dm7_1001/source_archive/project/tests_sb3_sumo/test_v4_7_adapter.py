from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from algos.sb3_torch.replay_buffer_v4_7 import (
    HorizonCorrectDictNStepReplayBufferV47,
)
from configs import sb3_configs_v4_7 as config
from tools import train_sb3_v4_7 as train


def test_v4_7_factory_changes_only_candidate_return_estimator() -> None:
    fake = mock.Mock()
    fake.replay_buffer = mock.Mock()
    fake.v4_method_metadata = {"implementation_id": "parent"}
    fake.graph_ablation = "parent"
    with mock.patch.object(config, "make_model_v4_6", return_value=fake), mock.patch.object(
        config, "_replace_empty_replay_buffer"
    ) as replace:
        observed = config.make_model_v4_7(
            config.V47_CANDIDATE,
            object(),
            scenario="cross",
            discount=0.99,
            action_repeat=3,
        )
    assert observed is fake
    replace.assert_called_once_with(fake, discount=0.99, action_repeat=3)
    assert observed.v4_method_metadata["single_change"] == (
        "horizon_correct_16_step_terminal_credit"
    )
    assert observed.v4_method_metadata["reward_changed"] is False
    assert observed.v4_method_metadata["network_changed"] is False


@pytest.mark.parametrize(
    ("algorithm", "n_step", "pairs", "changed"),
    [
        (config.TEMPORAL_GRAPH, 4, 2, False),
        (config.PARENT_CONTROL, 4, 4, False),
        (config.V47_CANDIDATE, 16, 4, True),
    ],
)
def test_v4_7_effective_hyperparameters_are_role_specific(
    algorithm: str, n_step: int, pairs: int, changed: bool
) -> None:
    values = train._method_hyperparameters_v4_7(algorithm)
    assert values["n_step"] == n_step
    assert values["checkpoint_decoder_candidate_pairs"] == pairs
    assert values["return_estimator_changed"] is changed
    assert values["reward_unchanged"] is True
    assert values["selector_changed"] is False


def test_v4_7_formal_unlock_rejects_parent_control_and_missing_receipt() -> None:
    common = dict(
        experiment_contract_sha256="a" * 64,
        implementation_freeze_sha256="b" * 64,
        formal_unlock_receipt=None,
    )
    with pytest.raises(ValueError, match="method pair"):
        train._validate_formal_unlock_v4_7(
            SimpleNamespace(
                evaluation_split="test", algo=config.PARENT_CONTROL, **common
            )
        )
    with pytest.raises(ValueError, match="missing promotion"):
        train._validate_formal_unlock_v4_7(
            SimpleNamespace(
                evaluation_split="test", algo=config.V47_CANDIDATE, **common
            )
        )


def test_v4_7_writer_persists_exact_runtime_return_diagnostics(tmp_path) -> None:
    replay = mock.Mock(spec=HorizonCorrectDictNStepReplayBufferV47)
    replay.return_estimator_diagnostics.return_value = {
        "schema_version": "topo-scene-v4.7.return-estimator-diagnostics/v1",
        "n_step": 16,
        "gamma": 0.99,
        "horizon_correct_bootstrap": True,
        "bootstrap_discount": "gamma_power_actual_horizon",
        "sampled_horizon_count": 64,
        "mean_sampled_actual_horizon": 15.0,
        "short_horizon_sample_rate": 0.1,
    }
    train._ACTIVE_MODEL = SimpleNamespace(replay_buffer=replay)
    path = tmp_path / "training_diagnostics.json"
    try:
        train._write_json_v4_7(
            path,
            {
                "schema_version": "topo-scene-v4.6.training-diagnostics/v1",
                "algorithm": config.V47_CANDIDATE,
                "raw_steps": 60,
            },
        )
    finally:
        train._ACTIVE_MODEL = None
    payload = json.loads(path.read_text(encoding="utf-8"))
    separate = json.loads(
        (tmp_path / "return_estimator_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["schema_version"].startswith("topo-scene-v4.7.")
    assert payload["return_estimator"] == separate
    assert separate["sampled_horizon_count"] == 64
    assert separate["implementation_id"] == config.V47_IMPLEMENTATION_IDS[
        config.V47_CANDIDATE
    ]
