from __future__ import annotations

from unittest import mock

from configs import sb3_configs_v4_6 as config
from tools.action_diagnostics_v4_6 import _target_integrity
from tools.train_sb3_v4_6 import (
    _method_hyperparameters_v4_6,
    _select_deployment_adapter,
)
from tools.checkpoint_selector_v4_4 import sha256
from pathlib import Path
from zipfile import ZipFile


def test_v4_6_factory_reuses_v45_learner_and_changes_only_metadata() -> None:
    fake = mock.Mock()
    fake.policy = mock.Mock()
    fake.v4_method_metadata = {"implementation_id": "parent", "reward_unchanged": True}
    fake.graph_ablation = "parent"
    with mock.patch.object(config, "make_model_v4_5", return_value=fake) as parent:
        observed = config.make_model_v4_6(
            "topo_v4_6_joint_checkpoint_decoder_selector",
            object(),
            scenario="cross",
            learning_rate=1e-4,
        )
    assert observed is fake
    assert parent.call_args.args[0] == "topo_v4_5_confident_actor_fusion"
    assert observed.v4_method_metadata["single_change"] == (
        "train_only_joint_checkpoint_decoder_selector"
    )
    assert observed.v4_method_metadata["training_unchanged_from_v4_5"] is True
    assert observed.v4_method_metadata["checkpoint_decoder_candidate_pairs"] == 4


def test_v4_6_effective_hyperparameters_freeze_training_and_decoders() -> None:
    values = _method_hyperparameters_v4_6(
        "topo_v4_6_joint_checkpoint_decoder_selector"
    )
    assert values["training_unchanged_from_v4_5"] is True
    assert values["lane_entropy_scale"] == 0.0
    assert values["speed_target_entropy"] == -1.0
    assert values["deployment_decoder_candidates"] == [
        "target_critic",
        "fusion_0_90",
    ]
    assert values["decoder_definitions_changed"] is False


def test_target_integrity_recomputes_argmax_keep_tie_mask_and_action() -> None:
    rows = [
        {
            "lane_command": 0,
            "target_critic_decoder": {
                "valid_lane_actions": [True, True, True],
                "minimum_target_twin_q": [0.0, 2.0, 2.0],
                "selected_lane_index": 1,
                "selected_lane": 0,
                "keep_was_exact_tied_maximum": True,
            },
        },
        {
            "lane_command": 1,
            "target_critic_decoder": {
                "valid_lane_actions": [False, True, True],
                "minimum_target_twin_q": [None, 1.0, 3.0],
                "selected_lane_index": 2,
                "selected_lane": 1,
                "keep_was_exact_tied_maximum": False,
            },
        },
    ]
    integrity = _target_integrity(rows)
    assert integrity["selected_decoder_exact_rule_match_rate"] == 1.0
    assert integrity["selected_decoder_exact_action_match_rate"] == 1.0
    assert integrity["selected_action_mask_feasible_rate"] == 1.0
    assert integrity["target_keep_tie_rule_match_rate"] == 1.0


def test_missing_training_best_reuses_exact_final_calibration(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final.zip"
    with ZipFile(checkpoint, "w") as archive:
        archive.writestr("payload", "final")
    summary = {
        "episodes": 1,
        "mean_return": 1.0,
        "std_return": 0.0,
        "mean_decision_steps": 1.0,
        "mean_raw_steps": 3.0,
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "off_route_rate": 0.0,
        "timeout_rate": 0.0,
    }
    records = [{
        "episode": 0,
        "seed": 10,
        "traffic_variant": "traffic_0.rou.xml",
        "success": True,
        "collision": False,
        "off_route": False,
        "timeout": False,
    }]
    pairs = []
    for decoder in ("target_critic", "fusion_0_90"):
        pairs.append({
            "checkpoint_kind": "exact_final",
            "deployment_decoder": decoder,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "traffic_partition": "train",
            "calibration_seed_start": 10,
            "summary": summary,
            "episode_records": records,
            "calibration_result_sha256": "a" * 64,
            "calibration_action_diagnostics_sha256": "b" * 64,
            "decoder_integrity": {"selected_deployment_decoder": decoder},
            "decoder_integrity_passed": True,
        })
    receipt = _select_deployment_adapter([{
        "checkpoint_kind": "exact_final",
        "deployment_candidates": pairs,
    }])
    assert receipt["training_best_missing_fallback_used"] is True
    assert receipt["candidate_count"] == 4
    assert receipt["selected_checkpoint_kind"] == "exact_final"
