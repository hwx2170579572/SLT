import unittest

import torch

from scene_event_prediction.progress_process import (
    EventDistributionProjector, ProgressGrid, ProgressProcess, ProgressTransitionHead,
)


class ProgressProcessTests(unittest.TestCase):
    def setUp(self):
        self.grid = ProgressGrid(spacing_m=1, states=5, steps=3, max_advance_bins=1)
        self.logits = torch.zeros(3, 5, 2, dtype=torch.float64, requires_grad=True)
        self.process = ProgressProcess(self.logits, self.grid)

    def test_analytic_binomial_and_mass_conservation(self):
        p = self.process.marginals()
        torch.testing.assert_close(p.sum(-1), torch.ones(4, dtype=torch.float64))
        torch.testing.assert_close(p[-1], torch.tensor([1, 3, 3, 1, 0], dtype=torch.float64) / 8)

    def test_missing_labels_integrate_not_negative(self):
        emission = torch.zeros(3, 5, dtype=torch.float64)
        emission[-1, 2] = 1
        observed = torch.tensor([False, False, True])
        out = self.process.filtered_log_likelihood(emission, observed)
        self.assertAlmostEqual(out["log_likelihood"].exp().item(), 3 / 8)
        self.assertEqual(out["observed_count"].item(), 1)
        all_missing = self.process.filtered_log_likelihood(emission, torch.zeros(3, dtype=torch.bool))
        self.assertAlmostEqual(all_missing["log_likelihood"].item(), 0)
        self.assertEqual(all_missing["observed_count"].item(), 0)

    def test_impossible_observation_is_explicit(self):
        emission = torch.zeros(3, 5, dtype=torch.float64)
        emission[0, 4] = 1  # One step can only advance by one bin.
        out = self.process.filtered_log_likelihood(emission, torch.tensor([True, False, False]))
        self.assertTrue(out["impossible"].item())
        self.assertTrue(torch.isneginf(out["log_likelihood"]).item())

    def test_single_process_entry_clear_and_censor(self):
        project = EventDistributionProjector()
        out = project(self.process, torch.tensor([1., 2., -1.]),
                      torch.tensor([2., 7., 0.]), torch.ones(3, dtype=torch.bool))
        # At t=3, P(progress=1) = 3/8; entry first passage tail = P(progress=0)=1/8.
        self.assertAlmostEqual(out["occupancy"][0, -1].item(), 3 / 8)
        self.assertAlmostEqual(out["entry_horizon_survival"][0].item(), 1 / 8)
        self.assertTrue(out["entry_valid"][1].item())
        self.assertFalse(out["clear_valid"][1].item())
        self.assertFalse(out["occupancy_valid"][1].item())
        self.assertEqual(out["initially_entered"][2].item(), 1)
        self.assertEqual(out["first_entry_after_now"][2].sum().item(), 0)
        self.assertEqual(out["occupancy"][2].sum().item(), 0)

    def test_cross_zone_joint_cannot_reverse_progress(self):
        positions = torch.arange(5)
        joint = self.process.two_time_region_probability(1, positions >= 1, 3, positions < 1)
        self.assertEqual(joint.item(), 0)
        p = self.process.marginals()
        independent = p[1, 1:].sum() * p[3, 0]
        self.assertGreater(independent.item(), 0)

    def test_known_right_censor_uses_survival_region(self):
        # This label means observed NOT to have reached progress 1 by t=3.
        # It is different from an actor disappearing before that observation.
        emission = torch.zeros(3, 5, dtype=torch.float64)
        emission[-1, 0] = 1
        observed = torch.tensor([False, False, True])
        out = self.process.filtered_log_likelihood(emission, observed)
        self.assertAlmostEqual(out["log_likelihood"].exp().item(), 1 / 8)
        (-out["log_likelihood"]).backward()
        self.assertTrue(torch.isfinite(self.logits.grad).all().item())
        self.assertGreater(self.logits.grad.abs().sum().item(), 0)

    def test_batched_projector_shape_and_gradient(self):
        logits = torch.zeros(2, 3, 4, 3, 5, 2, dtype=torch.float64, requires_grad=True)
        shape = (2, 3, 4, 2)
        entry = torch.ones(shape, dtype=torch.float64)
        clear = torch.full(shape, 2., dtype=torch.float64)
        out = EventDistributionProjector()(ProgressProcess(logits, self.grid), entry, clear,
                                            torch.ones(shape, dtype=torch.bool))
        self.assertEqual(tuple(out["occupancy"].shape), (2, 3, 4, 2, 4))
        out["occupancy"][..., -1].mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_overflow_is_absorbing(self):
        grid = ProgressGrid(spacing_m=1, states=2, steps=3, max_advance_bins=1)
        p = ProgressProcess(torch.zeros(3, 2, 2), grid).marginals()
        torch.testing.assert_close(p[-1], torch.tensor([1 / 8, 7 / 8]))

    def test_impossible_branch_does_not_poison_compatible_mixture_gradient(self):
        logits = torch.zeros(2, 3, 5, 2, dtype=torch.float64, requires_grad=True)
        emission = torch.zeros(2, 3, 5, dtype=torch.float64)
        emission[0, 0, 4] = 1  # Unsupported route prefix: impossible at t=1.
        emission[1, 0, 1] = 1
        observed = torch.tensor([[True, False, False], [True, False, False]])
        out = ProgressProcess(logits, self.grid).filtered_log_likelihood(emission, observed)
        mixture_loss = -torch.logsumexp(out["log_likelihood"] - torch.log(torch.tensor(2.)), 0)
        mixture_loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertEqual(logits.grad[0].abs().sum().item(), 0)
        self.assertGreater(logits.grad[1].abs().sum().item(), 0)

    def test_auxiliary_gradient_and_branch_permutation(self):
        torch.manual_seed(11)
        head = ProgressTransitionHead(4, self.grid, hidden=8)
        features = torch.randn(2, 3, 4, requires_grad=True)
        logits = head(features)
        self.assertEqual(tuple(logits.shape), (2, 3, 3, 5, 2))
        permutation = torch.tensor([2, 0, 1])
        torch.testing.assert_close(head(features[:, permutation]), logits[:, permutation])
        process = ProgressProcess(logits, self.grid)
        emission = torch.zeros(2, 3, 3, 5)
        emission[..., 2, 1] = 1
        observed = torch.zeros(2, 3, 3, dtype=torch.bool)
        observed[..., 2] = True
        loss = -process.filtered_log_likelihood(emission, observed)["log_likelihood"].mean()
        loss.backward()
        self.assertGreater(features.grad.abs().sum().item(), 0)
        self.assertGreater(head.context.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
