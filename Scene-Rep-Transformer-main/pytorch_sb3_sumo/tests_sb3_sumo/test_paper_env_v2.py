from __future__ import annotations

import pytest

from envs.sumo.paper_env_v2 import PaperSumoSceneEnvV2
from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
from tools.paper_evaluation_contract_v2 import expected_v2_partition


@pytest.mark.parametrize("scenario", tuple(PAPER_SCENARIOS))
def test_v2_traffic_split_is_deterministic_and_disjoint_when_possible(
    scenario: str,
) -> None:
    specification = PAPER_SCENARIOS[scenario]
    selections = {}
    for partition in ("train", "validation", "test"):
        env = PaperSumoSceneEnvV2(
            scenario=scenario,
            traffic_partition=partition,
        )
        selections[partition] = set(env._partitioned_traffic_paths(specification))
        env.close()
    all_paths = set(specification.traffic_paths)
    if len(all_paths) < 3:
        assert selections["train"] == selections["validation"] == selections["test"]
        assert selections["train"] == all_paths
    else:
        assert selections["train"].isdisjoint(selections["validation"])
        assert selections["train"].isdisjoint(selections["test"])
        assert selections["validation"].isdisjoint(selections["test"])
        assert set().union(*selections.values()) == all_paths
        paths = specification.traffic_paths
        assert len(selections["train"]) == sum(
            index % 5 in (2, 3, 4) for index in range(len(paths))
        )
        assert len(selections["validation"]) == sum(
            index % 5 == 0 for index in range(len(paths))
        )
        assert len(selections["test"]) == sum(
            index % 5 == 1 for index in range(len(paths))
        )


def test_v2_partition_contract_rejects_unregistered_combinations() -> None:
    assert expected_v2_partition("source_all", "validation") == "all"
    assert expected_v2_partition("frozen_60_20_20", "validation") == "validation"
    assert expected_v2_partition("frozen_60_20_20", "test") == "test"
    with pytest.raises(ValueError):
        expected_v2_partition("frozen_80_20", "validation")
    with pytest.raises(ValueError):
        expected_v2_partition("frozen_60_20_20", "development")
