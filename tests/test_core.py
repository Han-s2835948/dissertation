import unittest

import torch

from mnist_cdg.constraints import (
    GeometricSevenConstraint,
    SevenGeometryThresholds,
)
from mnist_cdg.models import HNet, ScoreUNet
from mnist_cdg.sde import VPSDE, reverse_sde_sample


class CoreTests(unittest.TestCase):
    def test_vp_marginal_shapes_and_bounds(self):
        sde = VPSDE()
        x = torch.zeros(4, 1, 28, 28)
        t = torch.tensor([0.001, 0.2, 0.5, 1.0])
        xt, noise, std = sde.marginal(x, t)
        self.assertEqual(xt.shape, x.shape)
        self.assertEqual(noise.shape, x.shape)
        self.assertTrue(torch.all(std > 0))
        self.assertTrue(torch.all(std <= 1))

    def test_network_shapes(self):
        x = torch.randn(2, 1, 28, 28)
        t = torch.rand(2)
        self.assertEqual(ScoreUNet(base_channels=8, time_dim=16)(x, t).shape, x.shape)
        h = HNet(base_channels=8, time_dim=16)(x, t)
        self.assertEqual(h.shape, (2,))
        self.assertTrue(torch.all((h >= 0) & (h <= 1)))

    def test_short_reverse_sample(self):
        model = ScoreUNet(base_channels=8, time_dim=16).eval()
        sample = reverse_sde_sample(
            model, VPSDE(), (2, 1, 28, 28), 2, torch.device("cpu")
        )
        self.assertEqual(sample.shape, (2, 1, 28, 28))
        self.assertTrue(torch.isfinite(sample).all())

    def test_empty_image_is_not_a_geometric_seven(self):
        rule = GeometricSevenConstraint(SevenGeometryThresholds())
        accepted = rule.indicator(torch.zeros(2, 1, 28, 28))
        self.assertFalse(accepted.any())


if __name__ == "__main__":
    unittest.main()

