import unittest

from rnr.core.distribution import (
    AdhesionDistributionBuilder,
    SizeDistributionBuilder,
)


class TestDistribution(unittest.TestCase):
    def setUp(self):
        # Parameters
        self.size_nbins = 100
        self.adh_nbins = 1000
        self.tol = 1e-3

        # Create size distributions
        # Unimodal distribution, no spread
        self.size_distrib_unimodal_disc = SizeDistributionBuilder(
            nmodes=1,
            width=0,
            nbins=1,
            modes=[5.0],
            spreads=[0.0],
            weights=[1.0],
        ).generate()

        # Multimodal, no spread
        self.size_distrib_multimodal_disc = SizeDistributionBuilder(
            nmodes=2,
            width=0,
            nbins=1,
            modes=[5.0, 10.0],
            spreads=[0.0, 0.0],
            weights=[0.5, 0.5],
        ).generate()

        # Unimodal with spread
        self.size_distrib_unimodal_cont = SizeDistributionBuilder(
            nmodes=1,
            width=3,
            nbins=self.size_nbins,
            modes=[5.0],
            spreads=[0.5],
            weights=[1.0],
        ).generate()

        # Multimodal with spread
        self.size_distrib_multimodal_cont = SizeDistributionBuilder(
            nmodes=2,
            width=3,
            nbins=self.size_nbins,
            modes=[5.0, 10.0],
            spreads=[0.5, 1.0],
            weights=[0.5, 0.5],
        ).generate()

        # Create adhesion distributions
        # Unimodal
        self.adh_distrib_unimodal = AdhesionDistributionBuilder(
            size_distrib=self.size_distrib_unimodal_disc,
            nbins=self.adh_nbins,
            fmax=30.0,
            dist_params="biasi",
            adhesion_model="JKR",
            surface_energy=0.1,
        ).generate()

        # Multimodal
        self.adh_distrib_multimodal = AdhesionDistributionBuilder(
            size_distrib=self.size_distrib_multimodal_cont,
            nbins=self.adh_nbins,
            fmax=30.0,
            dist_params="biasi",
            adhesion_model="JKR",
            surface_energy=0.1,
        ).generate()


class TestSizeDistribution(TestDistribution):
    def test_array_shapes(self):
        """Basic tests. Mostly makes sure the arrays created are of the right shape."""
        assert self.size_distrib_unimodal_disc.radii.shape[0] == 1
        assert self.size_distrib_unimodal_cont.radii.shape[0] == self.size_nbins
        assert self.size_distrib_multimodal_disc.radii.shape[0] == 2
        assert self.size_distrib_multimodal_cont.radii.shape[0] == self.size_nbins

    def test_sum(self):
        """Checks that the weights of the distribution sum up to 1."""
        assert abs(self.size_distrib_unimodal_disc.weights.sum() - 1.0) < self.tol
        assert abs(self.size_distrib_unimodal_cont.weights.sum() - 1.0) < self.tol
        assert abs(self.size_distrib_multimodal_disc.weights.sum() - 1.0) < self.tol
        assert abs(self.size_distrib_multimodal_cont.weights.sum() - 1.0) < self.tol


class TestAdhesionDistribution(TestDistribution):
    def test_array_shapes(self):
        """Basic tests. Mostly makes sure the arrays created are of the right shape."""
        assert self.adh_distrib_unimodal.weights.shape == (1, self.adh_nbins)
        assert self.adh_distrib_unimodal.fadh_norm.shape == (1, self.adh_nbins)
        assert self.adh_distrib_multimodal.weights.shape == (self.size_nbins, self.adh_nbins)
        assert self.adh_distrib_multimodal.fadh_norm.shape == (self.size_nbins, self.adh_nbins)

    def test_sum(self):
        """Checks that the weights for each size bins sum up to 1."""
        assert abs(self.adh_distrib_unimodal.weights.sum() - 1.0) < self.tol
        assert abs(self.adh_distrib_multimodal.weights.sum() - self.size_nbins) < self.tol


if __name__ == "__main__":
    unittest.main()
