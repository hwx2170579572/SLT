from __future__ import annotations

from contextlib import nullcontext
from unittest import mock

import pytest

from tools import run_all_v4_9_1_promotion as promotion
from tools import run_topo_v4_9_1_experiments as patch


def test_v4_9_1_contract_is_acceptance_only_and_formal_locked() -> None:
    contract = patch.validate_patch_contract()
    assert contract["scientific_protocol_changed"] is False
    assert contract["model_changed"] is False
    assert contract["training_changed"] is False
    assert contract["selection_changed"] is False
    assert contract["evaluation_changed"] is False
    assert contract["gate_thresholds_changed"] is False
    assert contract["formal_test"]["locked"] is True
    assert contract["formal_test"]["accessed"] is False
    receipt = patch.validate_preregistration()
    assert receipt["created_before_patch_implementation"] is True
    assert receipt["completed_M1_or_M2_will_be_rerun"] is False
    assert receipt["completed_run_artifacts_will_be_modified"] is False


def test_v4_9_1_derives_only_proved_train_only_metadata() -> None:
    derived = patch._derived_parent_metadata(
        {"selection_partition": "train"},
        {
            "selector_receipt_precedes_validation_environment": True,
            "selector_receipt_sha256": "a" * 64,
        },
    )
    assert derived == {
        "checkpoint_calibration_partition": "train",
        "validation_used_for_checkpoint_selection": False,
    }


@pytest.mark.parametrize(
    ("selector", "detailed", "message"),
    [
        (
            {"selection_partition": "validation"},
            {
                "selector_receipt_precedes_validation_environment": True,
                "selector_receipt_sha256": "a" * 64,
            },
            "cannot derive train selector partition",
        ),
        (
            {"selection_partition": "train"},
            {
                "selector_receipt_precedes_validation_environment": False,
                "selector_receipt_sha256": "a" * 64,
            },
            "cannot derive validation exclusion",
        ),
        (
            {"selection_partition": "train"},
            {
                "selector_receipt_precedes_validation_environment": True,
                "selector_receipt_sha256": None,
            },
            "selector receipt binding is missing",
        ),
    ],
)
def test_v4_9_1_rejects_unproved_metadata_derivation(
    selector, detailed, message
) -> None:
    with pytest.raises(patch.V491PatchError, match=message):
        patch._derived_parent_metadata(selector, detailed)


def test_v4_9_1_real_preflight_adopts_without_modifying_runs() -> None:
    result = patch.acceptance_preflight(write=False)
    assert set(result["original"]) == {"M1", "M2"}
    assert all(not row["accepted"] for row in result["original"].values())
    assert all(row == {"accepted": True, "reason": "accepted"} for row in result["patched"].values())
    assert result["completed_run_artifact_hashes_before"] == result[
        "completed_run_artifact_hashes_after"
    ]
    assert result["completed_run_artifacts_modified"] is False
    assert result["completed_jobs_rerun"] is False


def test_v4_9_1_acceptance_patch_restores_parent_context() -> None:
    original = patch.parent._parent_acceptance_view
    with patch.patched_acceptance():
        assert patch.parent._parent_acceptance_view is patch.patched_parent_acceptance_view
    assert patch.parent._parent_acceptance_view is original


def test_v4_9_1_promotion_wrapper_runs_all_parent_jobs_under_patch() -> None:
    sentinel = object()
    with (
        mock.patch.object(promotion.patch, "validate_patch_contract"),
        mock.patch.object(promotion.patch, "validate_preregistration"),
        mock.patch.object(promotion.patch, "validate_patch_freeze"),
        mock.patch.object(promotion.patch, "validate_engineering_receipt"),
        mock.patch.object(
            promotion.patch,
            "patched_acceptance",
            return_value=nullcontext(sentinel),
        ) as context,
        mock.patch.object(promotion.automation, "main", return_value=7) as main,
    ):
        assert promotion.main(["--device", "cpu"]) == 7
    context.assert_called_once_with()
    main.assert_called_once_with(["--device", "cpu"])
