from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from envs.sumo.high_density_env_v1 import (
    SUPPORTED_HIGH_DENSITY_SCENARIOS,
    build_high_density_overlay,
)
from envs.sumo.independent_v2_five_methods_six_scenarios_100ep_v1 import (
    IndependentV2FiveBySixEnvV1,
)
from envs.sumo.paper_scenario_registry import get_paper_scenario_spec
from tools import run_independent_v2_5m6s100e_v1 as runner
from tools import train_independent_v2_5m6s100e_v1 as trainer
from tools.independent_v2_5m6s100e_v1_common import (
    DEFAULT_PROTOCOL_PATH,
    METHOD_ADAPTERS,
    REMOVED_METHODS,
    SCENARIOS,
    build_cells,
    build_fresh_cells,
    load_protocol,
)


def test_protocol_is_exact_five_by_six_and_excludes_removed_methods() -> None:
    protocol = load_protocol()
    assert tuple(METHOD_ADAPTERS) == (
        "mst_slt",
        "temporal_graph",
        "full_balanced",
        "v4_8",
        "v4_13",
    )
    assert not set(REMOVED_METHODS) & set(METHOD_ADAPTERS)
    assert tuple(protocol["matrix"]["scenario_order"]) == SCENARIOS
    cells = build_cells(protocol, "comparison")
    assert len(cells) == 30
    assert sum(cell.execution == "adopt_parent_v2" for cell in cells) == 12
    assert len(build_fresh_cells(protocol, "comparison")) == 18
    assert protocol["matrix"]["total_evaluation_episodes"] == 3000


def test_all_six_scenarios_are_increased_with_calibrated_new_scales() -> None:
    protocol = load_protocol()
    # The high-density overlay now also covers cross_left / merge / intersection
    # (zero-shot scenes added after the six-scenario protocol was frozen), so it
    # is a superset of the protocol's scenario list rather than equal to it.
    assert set(SCENARIOS) <= set(SUPPORTED_HIGH_DENSITY_SCENARIOS)
    assert {
        scenario: protocol["scenarios"][scenario]["vehicle_scale"]
        for scenario in ("left_turn", "roundabout_easy", "roundabout_medium")
    } == {
        "left_turn": 1.30,
        "roundabout_easy": 1.25,
        "roundabout_medium": 1.20,
    }
    assert all(
        float(protocol["scenarios"][scenario]["vehicle_scale"]) > 1.0
        for scenario in SCENARIOS
    )


@pytest.mark.parametrize(
    ("scenario", "scale", "jitter", "expected_added"),
    (
        ("left_turn", 1.30, (4.0, 6.0), 150),
        ("roundabout_easy", 1.25, (2.0, 4.0), 205),
        ("roundabout_medium", 1.20, (1.5, 3.0), 190),
    ),
)
def test_new_scenario_overlays_add_demand_without_modifying_source(
    tmp_path: Path,
    scenario: str,
    scale: float,
    jitter: tuple[float, float],
    expected_added: int,
) -> None:
    source = get_paper_scenario_spec(scenario).traffic_paths[0]
    source_before = source.read_bytes()
    overlay = tmp_path / f"{scenario}_i5m6s100_v1.rou.xml"
    manifest = build_high_density_overlay(
        source,
        overlay,
        scenario=scenario,
        vehicle_scale=scale,
        pedestrian_scale=1.0,
        clone_depart_jitter_seconds=jitter,
    )
    assert manifest["additional_explicit_vehicles"] == expected_added
    assert manifest["original_assets_modified"] is False
    assert source.read_bytes() == source_before
    assert overlay.is_file()


def test_namespaced_overlay_suffix_is_distinct(tmp_path: Path) -> None:
    environment = object.__new__(IndependentV2FiveBySixEnvV1)
    environment._connection = None
    environment._high_density_overlay_root = tmp_path
    environment._high_density_vehicle_scale = 1.3
    environment.scenario = "left_turn"
    path = environment._overlay_path(Path("traffic_136.rou.xml"))
    assert path.name.endswith("_i5m6s100_v1.rou.xml")
    assert not path.name.endswith("_ss100_v2.rou.xml")


class _CaptureEnvironment:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _factory_args(split: str = "validation") -> argparse.Namespace:
    return argparse.Namespace(
        scenario="roundabout_medium",
        history_steps=10,
        neighbors=5,
        path_length=10,
        action_repeat=3,
        discount=0.99,
        ego_control_profile="direct",
        episode_limit_profile="source",
        gui=False,
        evaluation_split=split,
    )


def test_v4_factory_maps_validation_to_same_physical_holdout(tmp_path: Path) -> None:
    protocol = load_protocol()
    factory = trainer._make_environment_factory(
        adapter="v4_13",
        density=protocol["scenarios"]["roundabout_medium"],
        overlay_root=tmp_path,
        v4_environment_class=_CaptureEnvironment,
    )
    environment = factory(_factory_args(), evaluation=True)
    assert environment.kwargs["high_density_partition"] == "evaluation"
    assert environment.kwargs["high_density_contract_partition"] == "validation"
    with pytest.raises(ValueError, match="formal/test"):
        factory(_factory_args("test"), evaluation=True)


def test_v413_command_binds_lineage_and_never_requests_test(tmp_path: Path) -> None:
    protocol = load_protocol()
    args = argparse.Namespace(
        scenario="left_turn",
        seed=0,
        device="cpu",
        output_dir=tmp_path,
        model_name="fresh",
        check_only=True,
    )
    values = trainer._trainer_argv(
        args,
        adapter="v4_13",
        algorithm=METHOD_ADAPTERS["v4_13"][1],
        profile=protocol["profiles"]["smoke"],
        protocol=protocol,
    )
    assert values[values.index("--evaluation-split") + 1] == "validation"
    assert "test" not in values
    assert values[values.index("--experiment-contract-sha256") + 1] == (
        protocol["v4_13_lineage"]["architecture_contract_sha256"]
    )
    assert values[values.index("--implementation-freeze-sha256") + 1] == (
        protocol["v4_13_lineage"]["implementation_freeze_sha256"]
    )
    tensorboard_root = Path(
        values[values.index("--tensorboard-log-root") + 1]
    )
    assert tensorboard_root.name == "tb_v4"
    assert len(str(tensorboard_root / ("f" * 16) / "SAC_1")) < 200
    assert values[values.index("--checkpoint-prefix") + 1] == "ckpt"


def test_direct_trainer_refuses_parent_cell(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="already exists"):
        trainer.main(
            [
                "--method",
                "mst_slt",
                "--scenario",
                "cross",
                "--seed",
                "0",
                "--profile",
                "smoke",
                "--protocol",
                str(DEFAULT_PROTOCOL_PATH),
                "--output-dir",
                str(tmp_path),
                "--model-name",
                "must_not_run",
                "--device",
                "cpu",
                "--check-only",
            ]
        )
    assert not (tmp_path / "must_not_run").exists()


def test_parent_adoption_audits_exactly_twelve_real_cells() -> None:
    protocol = load_protocol()
    audit = runner._audit_parent(protocol)
    assert audit["complete"] is True
    assert audit["validated_cells"] == 12
    assert audit["training_or_evaluation_rerun"] is False
    assert all(row["evaluation_episodes"] == 100 for row in audit["rows"])


def test_protocol_file_is_plain_json() -> None:
    payload = json.loads(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert payload["matrix"]["method_scenario_cells"] == 30
