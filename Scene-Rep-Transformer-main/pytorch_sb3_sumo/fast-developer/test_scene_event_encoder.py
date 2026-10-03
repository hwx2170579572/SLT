"""Pure-torch tests for the independent M0/M1 scene encoders (no simulator)."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from scene_event.encoder import SceneEncoder
from scene_event.event_projection import events_off_observation


def make_public_map(lanes: int = 70, edges: int = 85, zones: int = 3) -> dict[str, np.ndarray]:
    points = np.zeros((lanes, 10, 2), dtype=np.float32)
    for lane in range(lanes):
        points[lane, :, 0] = float(lane % 10) * 3.5
        points[lane, :, 1] = np.linspace(float(lane // 10) * 12.0,
                                         float(lane // 10) * 12.0 + 30.0, 10)
    src = np.arange(edges, dtype=np.int64) % lanes
    dst = (src + 1) % lanes
    index = np.stack((src, dst), axis=0)
    return {
        "lane_points_xy": points,
        "lane_attrs": np.zeros((lanes, 8), dtype=np.float32),
        "lane_valid": np.ones((lanes,), dtype=np.bool_),
        "edge_index": index,
        "edge_type": (np.arange(edges, dtype=np.int64) % 6),
        "edge_valid": np.ones((edges,), dtype=np.bool_),
        "ego_route_lane_mask": (np.arange(lanes) < 12),
        "zone_polygons_xy": np.zeros((zones, 5, 2), dtype=np.float32),
        "zone_vertex_valid": np.ones((zones, 5), dtype=np.bool_),
        "lane_zone_s_m": np.zeros((lanes, zones, 2), dtype=np.float32),
        "lane_zone_valid": np.zeros((lanes, zones), dtype=np.bool_),
        "zone_type": (np.arange(zones, dtype=np.int64) % 2),
        "zone_valid": np.ones((zones,), dtype=np.bool_),
    }


def make_observation(batch: int = 2, actors: int = 6, zones: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    history = torch.zeros(batch, actors, 21, 6)
    base_xy = torch.zeros(batch, actors, 2)
    for actor in range(actors):
        base_xy[:, actor, 0] = float(actor % 3) * 7.0
        base_xy[:, actor, 1] = float(actor // 3) * 9.0 + (2.0 if actor else 0.0)
    history[..., 0:2] = base_xy[:, :, None, :]
    history[..., 1] += torch.linspace(-2.0, 0.0, 21)[None, None, :]
    history[..., 2] = 0.0
    history[..., 3] = 1.0
    history[..., 4] = 0.0
    history[..., 5] = 3.0
    history_valid = torch.ones(batch, actors, 21, dtype=torch.bool)
    actor_valid = torch.ones(batch, actors, dtype=torch.bool)
    routes, path_points = 4, 16
    ptr = torch.full((batch, actors, routes, path_points), -1, dtype=torch.int32)
    lane_mask = torch.zeros_like(ptr, dtype=torch.bool)
    for b in range(batch):
        for a in range(actors):
            for r in range(2):
                lane_mask[b, a, r, :3] = True
                ptr[b, a, r, :3] = torch.tensor([(a + r + j + b) % 70 for j in range(3)], dtype=torch.int32)
    candidate_valid = torch.zeros(batch, actors, routes, dtype=torch.bool)
    candidate_valid[:, :, :2] = True
    candidate_status = torch.zeros(batch, actors, routes, dtype=torch.uint8)
    candidate_status[:, :, :2] = 1
    event_s = torch.zeros(batch, actors, routes, zones, 2)
    event_mask = torch.zeros(batch, actors, routes, zones, dtype=torch.bool)
    event_status = torch.ones(batch, actors, routes, zones, dtype=torch.uint8)
    path_s = torch.zeros(batch, actors, routes, zones, 2)
    path_mask = torch.zeros(batch, actors, routes, zones, dtype=torch.bool)
    for b in range(batch):
        for a in range(1, actors):
            r = a % 2
            z = a % zones
            event_mask[b, a, r, z] = True
            event_status[b, a, r, z] = 2
            event_s[b, a, r, z] = torch.tensor([0.25 + 0.1 * a, 0.8 + 0.1 * a])
            path_mask[b, a, r, z] = True
            path_s[b, a, r, z] = torch.tensor([15.0 + 3.0 * a, 20.0 + 3.0 * a])
    obs = {
        "actor_history": history,
        "history_valid": history_valid,
        "actor_valid": actor_valid,
        "actor_size_lw": torch.full((batch, actors, 2), 4.5),
        "actor_age_s": torch.full((batch, actors), 2.0),
        "actor_lane_ptr": torch.tensor([[i % 70 for i in range(actors)] for _ in range(batch)], dtype=torch.int32),
        "actor_lane_s_m": torch.tensor([[12.0 + i for i in range(actors)] for _ in range(batch)]),
        "lane_match_conf": torch.ones(batch, actors),
        "remaining_time_s": torch.full((batch, 1), 45.0),
        "candidate_lane_ptr": ptr,
        "candidate_lane_mask": lane_mask,
        "candidate_valid": candidate_valid,
        "candidate_status": candidate_status,
        "candidate_progress_m": torch.full((batch, actors, routes), 5.0),
        "candidate_unknown_count": torch.zeros(batch, actors, dtype=torch.int32),
        "candidate_pair_topology_status": torch.zeros(
            batch, actors, routes, actors, routes, dtype=torch.uint8),
        "candidate_zone_event_s": event_s,
        "candidate_zone_event_mask": event_mask,
        "candidate_zone_status": event_status,
        "candidate_zone_s_m": path_s,
        "candidate_zone_path_mask": path_mask,
    }
    return obs


def permute_actors(obs: dict[str, torch.Tensor], order: torch.Tensor) -> dict[str, torch.Tensor]:
    result = dict(obs)
    actor_axis_keys = {
        "actor_history", "history_valid", "actor_valid", "actor_size_lw", "actor_age_s",
        "actor_lane_ptr", "actor_lane_s_m", "lane_match_conf", "candidate_lane_ptr",
        "candidate_lane_mask", "candidate_valid", "candidate_status", "candidate_progress_m",
        "candidate_unknown_count", "candidate_zone_event_s", "candidate_zone_event_mask",
        "candidate_zone_status", "candidate_zone_s_m", "candidate_zone_path_mask",
    }
    for key in actor_axis_keys:
        result[key] = obs[key][:, order]
    if "candidate_pair_topology_status" in obs:
        relation = obs["candidate_pair_topology_status"][:, order]
        result["candidate_pair_topology_status"] = relation[:, :, :, order, :]
    return result


def permute_routes(obs: dict[str, torch.Tensor], order: torch.Tensor) -> dict[str, torch.Tensor]:
    result = dict(obs)
    route_axis_keys = {"candidate_lane_ptr", "candidate_lane_mask", "candidate_valid", "candidate_status",
                       "candidate_progress_m", "candidate_zone_event_s", "candidate_zone_event_mask",
                       "candidate_zone_status", "candidate_zone_s_m", "candidate_zone_path_mask"}
    for key in route_axis_keys:
        result[key] = obs[key][:, :, order]
    if "candidate_pair_topology_status" in obs:
        relation = obs["candidate_pair_topology_status"][:, :, order]
        result["candidate_pair_topology_status"] = relation[:, :, :, :, order]
    return result


class SceneEventEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.map = make_public_map()
        self.obs = make_observation()

    def test_m0_and_m1_shapes_and_actual_map_size(self) -> None:
        for method in ("m0", "m1"):
            encoder = SceneEncoder(method, width=32, z_dim=24, public_map=self.map).eval()
            with torch.no_grad():
                z = encoder(self.obs)
            self.assertEqual(tuple(z.shape), (2, 24))
            self.assertTrue(torch.isfinite(z).all())
            self.assertEqual(encoder.public_map.num_lanes, 70)
            self.assertEqual(encoder.public_map.num_edges, 85)
            self.assertTrue(all(not buffer.requires_grad for buffer in encoder.public_map.buffers()))

    def test_common_initialization_matches_between_m0_and_m1(self) -> None:
        torch.manual_seed(0)
        m0 = SceneEncoder("m0", width=32, z_dim=24, public_map=self.map)
        torch.manual_seed(0)
        m1 = SceneEncoder("m1", width=32, z_dim=24, public_map=self.map)
        state0, state1 = m0.state_dict(), m1.state_dict()
        common = [name for name in state0 if not name.startswith("event_graph.")]
        self.assertEqual(set(common), {name for name in state1 if not name.startswith("event_graph.")})
        for name in common:
            torch.testing.assert_close(state0[name], state1[name], rtol=0.0, atol=0.0)

    def test_m0_does_not_consume_event_keys(self) -> None:
        encoder = SceneEncoder("m0", width=32, z_dim=24, public_map=self.map).eval()
        without = {k: v for k, v in self.obs.items() if not k.startswith("candidate_zone_event") and k != "candidate_zone_status"}
        changed = dict(self.obs)
        changed["candidate_zone_event_s"] = torch.randn_like(self.obs["candidate_zone_event_s"]) * 500.0
        changed["candidate_zone_event_mask"] = ~self.obs["candidate_zone_event_mask"]
        changed["candidate_zone_status"] = torch.full_like(self.obs["candidate_zone_status"], 4)
        with torch.no_grad():
            reference = encoder(without)
            actual = encoder(changed)
        torch.testing.assert_close(reference, actual, rtol=0.0, atol=0.0)

    def test_unknown_candidate_coverage_is_distinct_from_zero_omitted(self) -> None:
        encoder = SceneEncoder("m0", width=32, z_dim=24, public_map=self.map).eval()
        known_zero = dict(self.obs)
        known_zero["candidate_unknown_count"] = torch.zeros_like(self.obs["candidate_unknown_count"])
        unknown = dict(self.obs)
        unknown["candidate_unknown_count"] = torch.full_like(self.obs["candidate_unknown_count"], -1)
        with torch.no_grad():
            z_zero = encoder(known_zero)
            z_unknown = encoder(unknown)
        self.assertFalse(torch.allclose(z_zero, z_unknown, rtol=1e-7, atol=1e-9))

    def test_unknown_candidate_coverage_survives_zero_active_routes(self) -> None:
        encoder = SceneEncoder("m0", width=32, z_dim=24, public_map=self.map).eval()
        known_zero = dict(self.obs)
        unknown = dict(self.obs)
        for obs in (known_zero, unknown):
            obs["candidate_valid"] = torch.zeros_like(self.obs["candidate_valid"])
            obs["candidate_lane_mask"] = torch.zeros_like(self.obs["candidate_lane_mask"])
            obs["candidate_lane_ptr"] = torch.full_like(self.obs["candidate_lane_ptr"], -1)
        known_zero["candidate_unknown_count"] = torch.zeros_like(self.obs["candidate_unknown_count"])
        unknown["candidate_unknown_count"] = torch.full_like(self.obs["candidate_unknown_count"], -1)
        with torch.no_grad():
            z_zero = encoder(known_zero)
            z_unknown = encoder(unknown)
        self.assertFalse(torch.allclose(z_zero, z_unknown, rtol=1e-7, atol=1e-9))

    def test_topology_only_relation_reaches_m0_and_m1_readout(self) -> None:
        status = torch.zeros_like(self.obs["candidate_pair_topology_status"])
        # A directed actor-route relation: actor 1 route 0 has foe-only
        # evidence with actor 2 route 1, but no time-indexed event zone.
        status[:, 1, 0, 2, 1] = 1
        related = dict(self.obs)
        related["candidate_pair_topology_status"] = status
        with torch.no_grad():
            for method in ("m0", "m1"):
                encoder = SceneEncoder(method, width=32, z_dim=24, public_map=self.map).eval()
                base = encoder(self.obs)
                changed = encoder(related)
                self.assertFalse(torch.allclose(base, changed, rtol=1e-7, atol=1e-9))
                metrics = encoder.diagnostics()
                self.assertEqual(metrics["topology_relation_pairs"], 2.0)
                self.assertEqual(metrics["topology_relation_foe_pairs"], 2.0)
                self.assertEqual(metrics["topology_relation_merge_pairs"], 0.0)
                self.assertGreater(metrics["topology_relation_token_norm"], 0.0)
                self.assertGreater(metrics["topology_relation_readout_delta_norm"], 0.0)

    def test_topology_relation_is_masked_by_actor_and_route_activity(self) -> None:
        encoder = SceneEncoder("m0", width=32, z_dim=24, public_map=self.map).eval()
        none = dict(self.obs)
        inactive = dict(self.obs)
        status = torch.zeros_like(self.obs["candidate_pair_topology_status"])
        status[:, 1, 3, 2, 1] = 2  # actor 1 route 3 is inactive
        inactive["candidate_pair_topology_status"] = status
        with torch.no_grad():
            reference = encoder(none)
            actual = encoder(inactive)
        torch.testing.assert_close(reference, actual, rtol=0.0, atol=0.0)
        self.assertEqual(encoder.diagnostics()["topology_relation_pairs"], 0.0)

        actor_invalid = dict(self.obs)
        actor_invalid["actor_valid"] = self.obs["actor_valid"].clone()
        actor_invalid["actor_valid"][:, 1] = False
        actor_invalid["candidate_pair_topology_status"] = torch.zeros_like(
            self.obs["candidate_pair_topology_status"])
        actor_invalid["candidate_pair_topology_status"][:, 1, 0, 2, 1] = 1
        actor_invalid["candidate_pair_topology_status"][:, 2, 1, 1, 0] = 1
        actor_invalid_no_relation = dict(actor_invalid)
        actor_invalid_no_relation["candidate_pair_topology_status"] = torch.zeros_like(
            actor_invalid["candidate_pair_topology_status"])
        with torch.no_grad():
            expected = encoder(actor_invalid_no_relation)
            observed = encoder(actor_invalid)
        torch.testing.assert_close(expected, observed, rtol=0.0, atol=0.0)
        self.assertEqual(encoder.diagnostics()["topology_relation_pairs"], 0.0)

    def test_topology_relation_parameters_receive_gradient(self) -> None:
        encoder = SceneEncoder("m0", width=32, z_dim=24, public_map=self.map).train()
        related = dict(self.obs)
        related["candidate_pair_topology_status"] = torch.zeros_like(
            self.obs["candidate_pair_topology_status"])
        related["candidate_pair_topology_status"][:, 1, 0, 2, 1] = 3
        encoder(related).square().sum().backward()
        grads = [p.grad for p in encoder.topology_relation.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)

    def test_duplicate_sender_route_does_not_increase_relation_weight(self) -> None:
        encoder = SceneEncoder("m0", width=32, z_dim=24, public_map=self.map).eval()
        one = dict(self.obs)
        one["candidate_valid"] = torch.zeros_like(self.obs["candidate_valid"])
        one["candidate_valid"][:, 0, 0] = True
        one["candidate_valid"][:, 1, 0] = True
        one["candidate_valid"][:, 2, 1] = True
        one["candidate_pair_topology_status"] = torch.zeros_like(
            self.obs["candidate_pair_topology_status"])
        one["candidate_pair_topology_status"][:, 1, 0, 2, 1] = 1
        one["candidate_pair_topology_status"][:, 2, 1, 1, 0] = 1
        duplicate = dict(one)
        duplicate["candidate_valid"] = one["candidate_valid"].clone()
        duplicate["candidate_valid"][:, 2, 2] = True
        for key in ("candidate_lane_ptr", "candidate_lane_mask", "candidate_status",
                    "candidate_progress_m"):
            duplicate[key] = one[key].clone()
            duplicate[key][:, 2, 2] = one[key][:, 2, 1]
        duplicate["candidate_pair_topology_status"] = one["candidate_pair_topology_status"].clone()
        duplicate["candidate_pair_topology_status"][:, 1, 0, 2, 2] = 1
        duplicate["candidate_pair_topology_status"][:, 2, 2, 1, 0] = 1
        with torch.no_grad():
            z_one = encoder(one)
            z_duplicate = encoder(duplicate)
        torch.testing.assert_close(z_one, z_duplicate, rtol=1e-5, atol=2e-6)

    def test_absent_topology_status_is_backward_compatible(self) -> None:
        encoder = SceneEncoder("m0", width=32, z_dim=24, public_map=self.map).eval()
        without_key = dict(self.obs)
        without_key.pop("candidate_pair_topology_status")
        zero_status = dict(self.obs)
        with torch.no_grad():
            first = encoder(without_key)
            second = encoder(zero_status)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_event_off_matches_disabled_branch_and_has_zero_event_gradient(self) -> None:
        encoder = SceneEncoder("m1", width=32, z_dim=24, public_map=self.map).eval()
        without_events = events_off_observation(self.obs)
        with torch.no_grad():
            off = encoder(without_events)
            disabled = encoder(self.obs, disable_events=True)
        torch.testing.assert_close(off, disabled, rtol=0.0, atol=0.0)
        encoder.zero_grad(set_to_none=True)
        encoder(self.obs, disable_events=True).square().sum().backward()
        grads = [p.grad for p in encoder.event_graph.parameters()]
        self.assertTrue(any(g is not None for g in grads))
        self.assertTrue(all(g is None or torch.count_nonzero(g) == 0 for g in grads))

    def test_valid_events_reach_event_parameters_and_report_coverage(self) -> None:
        encoder = SceneEncoder("m1", width=32, z_dim=24, public_map=self.map).train()
        z = encoder(self.obs)
        z.square().sum().backward()
        grads = [p.grad for p in encoder.event_graph.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)
        diagnostics = encoder.diagnostics()
        self.assertGreater(diagnostics["event_valid_intervals"], 0.0)
        self.assertGreater(diagnostics["event_interval_routes"], 0.0)

    def test_actor_and_candidate_route_permutation_invariance(self) -> None:
        encoder = SceneEncoder("m1", width=32, z_dim=24, public_map=self.map).eval()
        social_order = torch.tensor([0, 4, 1, 5, 2, 3])
        route_order = torch.tensor([2, 0, 3, 1])
        related = dict(self.obs)
        related["candidate_pair_topology_status"] = self.obs["candidate_pair_topology_status"].clone()
        related["candidate_pair_topology_status"][:, 1, 0, 2, 1] = 3
        with torch.no_grad():
            reference = encoder(related)
            actor_permuted = encoder(permute_actors(related, social_order))
            route_permuted = encoder(permute_routes(related, route_order))
        torch.testing.assert_close(reference, actor_permuted, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(reference, route_permuted, rtol=1e-5, atol=1e-6)

    def test_duplicate_route_candidate_does_not_double_sender_actor_weight(self) -> None:
        encoder = SceneEncoder("m1", width=32, z_dim=24, public_map=self.map).eval()
        one = dict(self.obs)
        one["candidate_valid"] = torch.zeros_like(self.obs["candidate_valid"])
        one["candidate_valid"][:, :, 0] = True
        # Give actor 1 a valid, interacting route at slot zero.
        for key in ("candidate_lane_ptr", "candidate_lane_mask", "candidate_status", "candidate_progress_m",
                    "candidate_zone_event_s", "candidate_zone_event_mask", "candidate_zone_status",
                    "candidate_zone_s_m", "candidate_zone_path_mask"):
            one[key] = self.obs[key].clone()
            one[key][:, 1, 0] = self.obs[key][:, 1, 1]
        duplicate = dict(one)
        duplicate["candidate_valid"] = one["candidate_valid"].clone()
        duplicate["candidate_valid"][:, 1, 1] = True
        for key in ("candidate_lane_ptr", "candidate_lane_mask", "candidate_status", "candidate_progress_m",
                    "candidate_zone_event_s", "candidate_zone_event_mask", "candidate_zone_status",
                    "candidate_zone_s_m", "candidate_zone_path_mask"):
            duplicate[key] = one[key].clone()
            duplicate[key][:, 1, 1] = one[key][:, 1, 0]
        with torch.no_grad():
            z_one = encoder(one)
            z_duplicate = encoder(duplicate)
        torch.testing.assert_close(z_one, z_duplicate, rtol=1e-5, atol=2e-6)

    def test_stationary_zone_sequence_uses_route_distance_order(self) -> None:
        encoder = SceneEncoder("m1", width=32, z_dim=24, public_map=self.map).eval()
        before = dict(self.obs)
        before["candidate_zone_event_mask"] = torch.zeros_like(self.obs["candidate_zone_event_mask"])
        before["candidate_zone_status"] = self.obs["candidate_zone_status"].clone()
        before["candidate_zone_status"][:, 1, 0, :2] = 3  # stationary; no CV arrival time
        before["candidate_zone_path_mask"] = self.obs["candidate_zone_path_mask"].clone()
        before["candidate_zone_path_mask"][:, 1, 0, :2] = True
        before["candidate_zone_s_m"] = self.obs["candidate_zone_s_m"].clone()
        before["candidate_zone_s_m"][:, 1, 0, 0] = torch.tensor([10.0, 15.0])
        before["candidate_zone_s_m"][:, 1, 0, 1] = torch.tensor([15.0, 20.0])
        reversed_order = dict(before)
        reversed_order["candidate_zone_s_m"] = before["candidate_zone_s_m"].clone()
        reversed_order["candidate_zone_s_m"][:, 1, 0, 0] = torch.tensor([15.0, 10.0])
        reversed_order["candidate_zone_s_m"][:, 1, 0, 1] = torch.tensor([20.0, 15.0])
        with torch.no_grad():
            z_before = encoder(before)
            z_reversed = encoder(reversed_order)
        self.assertFalse(torch.allclose(z_before, z_reversed, rtol=1e-6, atol=1e-7))

    def test_padding_contents_and_all_empty_histories_are_safe(self) -> None:
        encoder = SceneEncoder("m1", width=32, z_dim=24, public_map=self.map).eval()
        obs = dict(self.obs)
        obs["history_valid"] = self.obs["history_valid"].clone()
        obs["history_valid"][:, 2, 4:7] = False
        changed_padding = dict(obs)
        changed_padding["actor_history"] = obs["actor_history"].clone()
        changed_padding["actor_history"][:, 2, 4:7] = 1e6
        with torch.no_grad():
            first = encoder(obs)
            second = encoder(changed_padding)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        padded_paths = dict(obs)
        padded_paths["candidate_lane_ptr"] = obs["candidate_lane_ptr"].clone()
        padded_paths["candidate_lane_ptr"][~obs["candidate_lane_mask"]] = 999
        padded_events = dict(obs)
        padded_events["candidate_zone_event_s"] = obs["candidate_zone_event_s"].clone()
        padded_events["candidate_zone_event_s"][~obs["candidate_zone_event_mask"]] = 1e6
        padded_path = dict(obs)
        padded_path["candidate_zone_s_m"] = obs["candidate_zone_s_m"].clone()
        padded_path["candidate_zone_s_m"][~obs["candidate_zone_path_mask"]] = 1e6
        with torch.no_grad():
            route_padding = encoder(padded_paths)
            event_padding = encoder(padded_events)
            path_padding = encoder(padded_path)
        torch.testing.assert_close(first, route_padding, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first, event_padding, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first, path_padding, rtol=0.0, atol=0.0)
        empty = dict(obs)
        empty["history_valid"] = torch.zeros_like(obs["history_valid"])
        empty["actor_valid"] = torch.zeros_like(obs["actor_valid"])
        with torch.no_grad():
            z = encoder(empty)
        self.assertTrue(torch.isfinite(z).all())

    def test_unbatched_contract(self) -> None:
        encoder = SceneEncoder("m0", width=32, z_dim=24, public_map=self.map).eval()
        obs = {key: value[0] for key, value in self.obs.items()}
        with torch.no_grad():
            z = encoder(obs)
        self.assertEqual(tuple(z.shape), (24,))

    def test_fixed_policy_width_and_24_actor_axis(self) -> None:
        obs = make_observation(batch=1, actors=24)
        encoder = SceneEncoder("m1", width=128, z_dim=128, public_map=self.map).eval()
        with torch.no_grad():
            z = encoder(obs)
        self.assertEqual(tuple(z.shape), (1, 128))
        self.assertTrue(torch.isfinite(z).all())


if __name__ == "__main__":
    unittest.main()
