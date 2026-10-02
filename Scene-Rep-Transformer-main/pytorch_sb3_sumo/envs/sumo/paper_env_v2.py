"""Three-way traffic split used only by the topology-temporal v2 study."""

from __future__ import annotations

from pathlib import Path

from .paper_env import PaperSumoSceneEnv
from .paper_scenario_registry import PaperScenarioSpec


V2_TRAFFIC_PARTITIONS = ("all", "train", "validation", "test")


class PaperSumoSceneEnvV2(PaperSumoSceneEnv):
    """Paper environment with a frozen 60/20/20 asset partition.

    Lexicographic traffic indices modulo five are assigned as follows:

    * ``0``: validation (the already exposed v1 evaluation subset),
    * ``1``: v2 formal test (not used by v2 training or selection),
    * ``2,3,4``: v2 training.

    CARLA has one reconstructed traffic file and therefore necessarily reuses
    it in all partitions; the inherited info payload exposes non-disjointness.
    """

    def __init__(
        self,
        *args,
        traffic_partition: str = "all",
        **kwargs,
    ) -> None:
        if traffic_partition not in V2_TRAFFIC_PARTITIONS:
            raise ValueError(
                f"traffic_partition must be one of {V2_TRAFFIC_PARTITIONS}, "
                f"got {traffic_partition!r}"
            )
        requested_partition = str(traffic_partition)
        # Parent validation knows only all/train/evaluation.  Construct with
        # all, then install the v2 partition consumed by our override.
        super().__init__(*args, traffic_partition="all", **kwargs)
        self._traffic_partition = requested_partition

    def _partitioned_traffic_paths(
        self, specification: PaperScenarioSpec
    ) -> tuple[Path, ...]:
        paths = specification.traffic_paths
        if self._traffic_partition == "all" or len(paths) < 3:
            return paths
        if self._traffic_partition == "validation":
            selected = tuple(path for index, path in enumerate(paths) if index % 5 == 0)
        elif self._traffic_partition == "test":
            selected = tuple(path for index, path in enumerate(paths) if index % 5 == 1)
        elif self._traffic_partition == "train":
            selected = tuple(path for index, path in enumerate(paths) if index % 5 in (2, 3, 4))
        else:
            raise ValueError(f"Unexpected v2 traffic partition {self._traffic_partition!r}")
        if not selected:
            raise RuntimeError(
                f"Traffic partition {self._traffic_partition!r} is empty for "
                f"scenario {specification.name!r}"
            )
        return selected


__all__ = ["PaperSumoSceneEnvV2", "V2_TRAFFIC_PARTITIONS"]
