from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
import torch

from algos.sb3_torch.contractfix_encoder import (
    ContractFixedIncrementalTopoEncoder,
    ContractFixedTemporalInteractionEncoder,
)
from algos.sb3_torch.features import _nonzero_mask
from algos.sb3_torch.incremental_topo_encoder import IncrementalTopoEncoder
from envs.sumo.topology_graph_v2 import PaddedTopologyGraphV2


def _empty_topology_graph() -> PaddedTopologyGraphV2:
    return PaddedTopologyGraphV2(
        lane_points=np.zeros((1, 10, 2), dtype=np.float32),
        lane_attrs=np.zeros((1, 8), dtype=np.float32),
        node_mask=np.zeros((1,), dtype=bool),
        edge_index=np.zeros((2, 1), dtype=np.int64),
        edge_type=np.zeros((1,), dtype=np.int64),
        edge_mask=np.zeros((1,), dtype=bool),
    )


def _observation_space(*, actors: int = 3, history: int = 5, paths_per_actor: int = 2):
    return gym.spaces.Dict(
        {
            "trajectory": gym.spaces.Box(
                -1000.0, 1000.0, shape=(actors, history, 5), dtype=np.float32
            ),
            "map": gym.spaces.Box(
                -1000.0,
                1000.0,
                shape=(actors * paths_per_actor, 3, 5),
                dtype=np.float32,
            ),
        }
    )


def _make_encoder(*, use_route: bool = False, actors: int = 3, history: int = 5):
    return ContractFixedIncrementalTopoEncoder(
        _observation_space(actors=actors, history=history),
        source_observation_contract="smarts",
        topology_graph=_empty_topology_graph(),
        use_route=use_route,
        use_topology=False,
        use_slots=False,
        use_incremental_slots=False,
        features_dim=128,
        hidden_dim=128,
        num_heads=2,
        random_augmentation=False,
        carla_contract=False,
    )


class LastValidSelectionTests(unittest.TestCase):
    def test_helper_supports_raw_and_batched_layouts_and_all_mask_patterns(self):
        # The history axis is penultimate in both raw [A,H,D] and batched
        # [B,A,H,D] inputs.  Include left/right padding, a gap, and empty rows.
        raw = torch.arange(3 * 6 * 2, dtype=torch.float32).reshape(3, 6, 2)
        raw_mask = torch.tensor(
            [
                [False, False, True, True, True, True],  # left padded, m=4
                [True, True, False, False, False, False],  # right padded, m=2
                [True, False, True, False, False, False],  # internal gap
            ]
        )
        selected = ContractFixedIncrementalTopoEncoder._last_valid(raw, raw_mask)
        torch.testing.assert_close(selected, torch.stack([raw[0, 5], raw[1, 1], raw[2, 2]]))

        batched = raw.unsqueeze(0).repeat(2, 1, 1, 1)
        batched_mask = raw_mask.unsqueeze(0).repeat(2, 1, 1)
        batched_mask[1, 1:] = False
        selected_batched = ContractFixedIncrementalTopoEncoder._last_valid(
            batched, batched_mask
        )
        torch.testing.assert_close(selected_batched[0], selected)
        torch.testing.assert_close(selected_batched[1, 0], raw[0, 5])
        torch.testing.assert_close(selected_batched[1, 1:], torch.zeros_like(selected_batched[1, 1:]))

    def test_ego_rotation_anchor_uses_true_last_and_empty_is_zero(self):
        encoder = _make_encoder()
        trajectories = torch.zeros(2, 3, 5, 5)
        trajectories[0, 0, 2] = torch.tensor([3.0, 4.0, 0.5, 2.0, 1.0])
        trajectories[0, 0, 4] = torch.tensor([9.0, 8.0, 1.25, 6.0, 2.0])
        trajectories[1, 0, :, 1] = 7.0  # x=0 remains invalid under legacy mask
        valid = _nonzero_mask(trajectories)
        frames = encoder._current_ego_frame(trajectories, valid)
        torch.testing.assert_close(frames[0], trajectories[0, 0, 4])
        torch.testing.assert_close(frames[1], torch.zeros(5))
        self.assertFalse(bool(valid[1, 0].any()))

    def test_topology_last_query_mask_uses_true_index_for_padding_gaps_and_empty(self):
        encoder = _make_encoder()
        key_mask = torch.zeros(1, 3, 5, 2, dtype=torch.bool)
        valid = torch.tensor(
            [
                [
                    [False, True, False, False, True],  # gap, actual index 4
                    [True, True, False, False, False],  # right pad, index 1
                    [False, False, False, False, False],  # empty
                ]
            ]
        )
        key_mask[0, 0, 1] = torch.tensor([True, False])  # count-1 would choose this
        key_mask[0, 0, 4] = torch.tensor([False, True])  # actual last slot
        key_mask[0, 1, 1] = torch.tensor([True, True])
        key_mask[0, 2, 0] = torch.tensor([True, True])  # must stay invalid
        selected = encoder._last_query_mask(key_mask, valid)
        torch.testing.assert_close(selected[0, 0], torch.tensor([False, True]))
        torch.testing.assert_close(selected[0, 1], torch.tensor([True, True]))
        torch.testing.assert_close(selected[0, 2], torch.tensor([False, False]))

    def test_actual_route_query_uses_last_true_frame_and_empty_actor_is_safe(self):
        torch.manual_seed(11)
        encoder = _make_encoder(use_route=True).eval()
        trajectories = torch.zeros(1, 3, 5, 5)
        trajectories[0, 0, 2] = torch.tensor([10.0, 2.0, 0.1, 3.0, 0.0])
        trajectories[0, 0, 4] = torch.tensor([20.0, 4.0, 0.2, 4.0, 0.0])
        trajectories[0, 1, 1] = torch.tensor([11.0, 3.0, 0.0, 2.0, 0.0])
        trajectories[0, 1, 4] = torch.tensor([15.0, 5.0, 0.1, 3.0, 0.0])
        trajectories[0, 2, :, 1] = 8.0  # x==0 => legacy presence mask says empty
        map_state = torch.ones(1, 6, 3, 5)
        observations = {
            "trajectory": trajectories,
            "map": map_state,
            # Optional Dict keys remain legal and are ignored by this D1 path.
            "state_lstm": torch.zeros(1, 5, 15),
            "state_lstm_mask": torch.ones(1, 5),
            "auxiliary_probe": torch.zeros(1, 3),
        }

        valid = _nonzero_mask(trajectories)
        rotated, _, _ = encoder._rotate_trajectories(trajectories, valid)
        state_tokens = encoder.state_encoder(rotated) * valid[..., None]
        expected_query = encoder._last_valid(state_tokens, valid).reshape(3, 1, 128)
        observed: dict[str, torch.Tensor] = {}

        def capture_query(module, inputs):
            observed["query"] = inputs[0].detach().clone()

        handle = encoder.route_attention.register_forward_pre_hook(capture_query)
        try:
            output = encoder(observations)
        finally:
            handle.remove()

        self.assertEqual(tuple(output.shape), (1, 128))
        self.assertTrue(bool(torch.isfinite(output).all()))
        torch.testing.assert_close(observed["query"], expected_query)
        torch.testing.assert_close(observed["query"][2], torch.zeros_like(observed["query"][2]))

    def test_policy_forward_accepts_batched_small_actor_history_layout(self):
        # D1's public env space is [A,H,F]; SB3 adds B at policy time.  The
        # encoder accepts another valid A/H pair and leaves extra Dict keys
        # alone, without guessing a reshape from a flattened state_lstm field.
        encoder = _make_encoder(use_route=False, actors=2, history=1).eval()
        trajectories = torch.zeros(2, 2, 1, 5)
        trajectories[:, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0])
        trajectories[:, 1, 0] = torch.tensor([3.0, 1.0, 0.0, 1.0, 0.0])
        observations = {
            "trajectory": trajectories,
            "map": torch.ones(2, 4, 3, 5),
            "state_lstm": torch.zeros(2, 1, 10),
            "state_lstm_mask": torch.ones(2, 1),
        }
        output = encoder(observations)
        self.assertEqual(tuple(output.shape), (2, 128))
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_temporal_pool_query_uses_max_valid_index_and_all_invalid_is_zero(self):
        torch.manual_seed(19)
        encoder = ContractFixedTemporalInteractionEncoder(
            feature_dim=8, num_heads=2, history_steps=6
        ).eval()
        features = torch.randn(1, 4, 6, 8)
        valid = torch.tensor(
            [
                [
                    [False, False, True, True, True, True],
                    [True, True, False, False, False, False],
                    [True, False, False, True, False, False],
                    [False, False, False, False, False, False],
                ]
            ]
        )
        observed: dict[str, torch.Tensor] = {}

        def capture_query(module, inputs):
            observed["query"] = inputs[0].detach().clone()

        handle = encoder.pool_attention.register_forward_pre_hook(capture_query)
        try:
            output = encoder(features, valid)
        finally:
            handle.remove()

        flat = features.reshape(4, 6, 8)
        flat_valid = valid.reshape(4, 6)
        actor_valid = flat_valid.any(dim=1)
        safe_valid = flat_valid.clone()
        safe_valid[~actor_valid, 0] = True
        embedded = (flat + encoder.lag_embedding[None, :6]) * flat_valid[..., None]
        attended, _ = encoder.self_attention(
            embedded, embedded, embedded, key_padding_mask=~safe_valid, need_weights=False
        )
        encoded = encoder.norm1(embedded + attended) * flat_valid[..., None]
        encoded = encoder.norm2(encoded + encoder.ffn(encoded)) * flat_valid[..., None]
        max_indices = torch.tensor([5, 1, 3, 0])
        expected = encoded[torch.arange(4), max_indices].unsqueeze(1)
        torch.testing.assert_close(observed["query"], expected)
        torch.testing.assert_close(output[0, 3], torch.zeros(8))
        self.assertTrue(bool(torch.isfinite(output).all()))


class GeometryEdgeTests(unittest.TestCase):
    @staticmethod
    def _pseudo_velocity(cartesian: tuple[float, float]) -> tuple[float, float]:
        vx, vy = cartesian
        return vy, -vx

    def _pair_edge(self, p0, v0, p1, v1, *, v2: bool = False):
        encoder = _make_encoder(actors=2, history=1)
        trajectories = torch.zeros(1, 2, 1, 5)
        trajectories[0, 0, 0, :2] = torch.tensor(p0)
        trajectories[0, 0, 0, 3:5] = torch.tensor(self._pseudo_velocity(v0))
        trajectories[0, 1, 0, :2] = torch.tensor(p1)
        trajectories[0, 1, 0, 3:5] = torch.tensor(self._pseudo_velocity(v1))
        valid = torch.ones(1, 2, 1, dtype=torch.bool)
        if v2:
            edge = encoder._vehicle_edge_features_v2(
                trajectories,
                valid,
                torch.ones(1, 2, 1, 1),
                collect_diagnostics=False,
            )[0]
        else:
            edge = encoder._vehicle_edge_features(
                trajectories, valid, None, include_topology_relations=False
            )[0]
        return edge[0, 0, 0, 1]

    def test_cartesian_velocity_conversion_for_cardinal_headings(self):
        headings = torch.tensor([0.0, np.pi / 2, np.pi, -np.pi / 2])
        speed = 7.0
        pseudo = torch.stack((speed * torch.cos(headings), speed * torch.sin(headings)), dim=-1)
        corrected = ContractFixedIncrementalTopoEncoder._correct_smarts_geometry_edges(
            torch.cat(
                [
                    torch.zeros(4, 1, 1, 1, 2),
                    pseudo[:, None, None, None, :],
                    torch.zeros(4, 1, 1, 1, 4),
                ],
                dim=-1,
            )
        )
        expected = torch.tensor([[0.0, 7.0], [-7.0, 0.0], [0.0, -7.0], [7.0, 0.0]])
        torch.testing.assert_close(corrected[:, 0, 0, 0, 2:4], expected)

    def test_following_head_on_and_crossing_cpa_times_use_raw_units(self):
        # Edge [0,0,0,1] stores actor-1 minus actor-0 for both r and v.
        following = self._pair_edge((0.0, 0.0), (5.0, 0.0), (20.0, 0.0), (0.0, 0.0))
        torch.testing.assert_close(following[2:4], torch.tensor([-5.0, 0.0]))
        torch.testing.assert_close(following[5], torch.tensor(0.2))  # 20 m / 5 m/s = 4 s

        head_on = self._pair_edge((0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (-5.0, 0.0))
        torch.testing.assert_close(head_on[2:4], torch.tensor([-10.0, 0.0]))
        torch.testing.assert_close(head_on[5], torch.tensor(0.05))  # 1 s / 20 s cap

        crossing = self._pair_edge((0.0, 0.0), (4.0, 0.0), (5.0, -5.0), (0.0, 4.0))
        torch.testing.assert_close(crossing[2:4], torch.tensor([-4.0, 4.0]))
        torch.testing.assert_close(crossing[5], torch.tensor(0.0625))  # 1.25 s / 20 s

    def test_parallel_and_stationary_relative_motion_use_finite_undefined_sentinel(self):
        parallel = self._pair_edge((0.0, 0.0), (5.0, 0.0), (0.0, 3.0), (5.0, 0.0))
        stationary = self._pair_edge((0.0, 0.0), (0.0, 0.0), (4.0, 0.0), (0.0, 0.0))
        torch.testing.assert_close(parallel[2:4], torch.zeros(2))
        torch.testing.assert_close(stationary[2:4], torch.zeros(2))
        torch.testing.assert_close(parallel[5], torch.tensor(1.0))
        torch.testing.assert_close(stationary[5], torch.tensor(1.0))
        self.assertTrue(bool(torch.isfinite(parallel).all()))
        self.assertTrue(bool(torch.isfinite(stationary).all()))

    def test_v2_relation_edge_path_also_corrects_geometry(self):
        edge = self._pair_edge(
            (0.0, 0.0), (5.0, 0.0), (20.0, 0.0), (0.0, 0.0), v2=True
        )
        torch.testing.assert_close(edge[2:4], torch.tensor([-5.0, 0.0]))
        torch.testing.assert_close(edge[5], torch.tensor(0.2))

    def test_rotation_commutes_with_velocity_basis_recovery(self):
        edge = torch.tensor([[[[[10.0, -5.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0]]]]])
        corrected = ContractFixedIncrementalTopoEncoder._correct_smarts_geometry_edges(edge)
        angle = torch.tensor(0.73)
        cosine, sine = torch.cos(angle), torch.sin(angle)

        def rotate(vector):
            x, y = vector.unbind(-1)
            return torch.stack((cosine * x + sine * y, -sine * x + cosine * y), dim=-1)

        rotated_edge = edge.clone()
        rotated_edge[..., :2] = rotate(edge[..., :2])
        rotated_edge[..., 2:4] = rotate(edge[..., 2:4])
        corrected_rotated = ContractFixedIncrementalTopoEncoder._correct_smarts_geometry_edges(
            rotated_edge
        )
        torch.testing.assert_close(corrected_rotated[..., :2], rotate(corrected[..., :2]))
        torch.testing.assert_close(corrected_rotated[..., 2:4], rotate(corrected[..., 2:4]))
        torch.testing.assert_close(corrected_rotated[..., 5], corrected[..., 5])


class CompatibilityTests(unittest.TestCase):
    def test_state_dict_parameter_count_and_rng_stream_match_legacy_class(self):
        kwargs = {
            "topology_graph": _empty_topology_graph(),
            "use_route": False,
            "use_topology": False,
            "use_slots": False,
            "use_incremental_slots": False,
            "features_dim": 128,
            "hidden_dim": 128,
            "num_heads": 2,
            "random_augmentation": False,
            "carla_contract": False,
        }
        torch.manual_seed(31)
        legacy = IncrementalTopoEncoder(_observation_space(), **kwargs)
        legacy_next_random = torch.rand(4)

        torch.manual_seed(31)
        fixed = ContractFixedIncrementalTopoEncoder(
            _observation_space(), source_observation_contract="smarts", **kwargs
        )
        fixed_next_random = torch.rand(4)

        self.assertEqual(set(legacy.state_dict()), set(fixed.state_dict()))
        self.assertEqual(
            sum(parameter.numel() for parameter in legacy.parameters()),
            sum(parameter.numel() for parameter in fixed.parameters()),
        )
        torch.testing.assert_close(legacy_next_random, fixed_next_random)
        for key, value in legacy.state_dict().items():
            torch.testing.assert_close(value, fixed.state_dict()[key])

    def test_non_smarts_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only supports the SMARTS"):
            ContractFixedIncrementalTopoEncoder(
                _observation_space(),
                source_observation_contract="carla",
                topology_graph=_empty_topology_graph(),
                use_topology=False,
                carla_contract=True,
            )

    def test_state_lstm_only_and_flattened_spaces_are_rejected_explicitly(self):
        state_lstm_only = gym.spaces.Dict(
            {
                "state_lstm": gym.spaces.Box(
                    -1.0, 1.0, shape=(5, 15), dtype=np.float32
                ),
                "state_lstm_mask": gym.spaces.Box(
                    0.0, 1.0, shape=(5,), dtype=np.float32
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "requires Dict keys"):
            ContractFixedIncrementalTopoEncoder(
                state_lstm_only,
                source_observation_contract="smarts",
                topology_graph=_empty_topology_graph(),
                use_topology=False,
            )

        flattened = gym.spaces.Dict(
            {
                "trajectory": gym.spaces.Box(
                    -1000.0, 1000.0, shape=(3 * 5 * 5,), dtype=np.float32
                ),
                "map": gym.spaces.Box(
                    -1000.0, 1000.0, shape=(6, 3, 5), dtype=np.float32
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "trajectory must have observation shape"):
            ContractFixedIncrementalTopoEncoder(
                flattened,
                source_observation_contract="smarts",
                topology_graph=_empty_topology_graph(),
                use_topology=False,
            )

        flat_box = gym.spaces.Box(-1.0, 1.0, shape=(120,), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "requires Dict keys"):
            ContractFixedIncrementalTopoEncoder(
                flat_box,
                source_observation_contract="smarts",
                topology_graph=_empty_topology_graph(),
                use_topology=False,
            )


if __name__ == "__main__":
    unittest.main()
