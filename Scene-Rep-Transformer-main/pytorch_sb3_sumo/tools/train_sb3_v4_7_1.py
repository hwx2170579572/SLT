"""Engineering-only metadata binding patch for the frozen v4.7 trainer."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import train_sb3_v4_6 as deployment_parent
from tools import train_sb3_v4_7 as frozen


PATCH_ID = "v4_7_1_bind_sealed_selected_deployment_fields"
_ORIGINAL_WRITER = frozen._write_json_v4_7


def _selected_deployment_bindings() -> dict[str, str]:
    selector = deployment_parent._SELECTED_DEPLOYMENT
    if selector is None:
        raise RuntimeError("selected deployment is not sealed before detailed output")
    bindings = {
        "selected_deployment_decoder": selector["selected_deployment_decoder"],
        "selected_source_checkpoint_sha256": selector[
            "selected_checkpoint_sha256"
        ],
        "selected_model_policy_class": selector["selected_model_policy_class"],
        "selected_model_parameter_state_sha256": selector[
            "selected_model_parameter_state_sha256"
        ],
    }
    if not all(isinstance(value, str) and value for value in bindings.values()):
        raise ValueError("sealed selector contains an empty deployment binding")
    return bindings


def _write_json_v4_7_1(path: Path, payload: Any) -> None:
    value = copy.deepcopy(payload)
    if path.name == "paper_evaluation_detailed.json":
        if not isinstance(value, dict):
            raise TypeError("detailed evaluation payload must be a mapping")
        bindings = _selected_deployment_bindings()
        for key, expected in bindings.items():
            if key in value and value[key] != expected:
                raise ValueError(f"detailed evaluation conflicts with sealed {key}")
        value.update(bindings)
        value.update(
            {
                "engineering_patch_id": PATCH_ID,
                "selected_deployment_bindings_source": (
                    "selector_receipt_sealed_before_validation"
                ),
                "scientific_protocol_changed_by_engineering_patch": False,
            }
        )
    _ORIGINAL_WRITER(path, value)


def main(
    argv: list[str] | None = None,
    *,
    env_factory: Callable[..., Any] | None = None,
    default_output_dir: Path | None = None,
    require_paper_evaluation_contract: bool = False,
) -> int:
    original = frozen._write_json_v4_7
    frozen._write_json_v4_7 = _write_json_v4_7_1
    try:
        return frozen.main(
            argv,
            env_factory=env_factory,
            default_output_dir=default_output_dir,
            require_paper_evaluation_contract=require_paper_evaluation_contract,
        )
    finally:
        frozen._write_json_v4_7 = original


if __name__ == "__main__":
    raise SystemExit(main())
