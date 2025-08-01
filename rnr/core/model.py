import logging

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from rnr.core.distribution import AdhesionDistribution, SizeDistribution
from rnr.core.flow import Flow

logger = logging.getLogger(__name__)


class ResuspensionModel:
    def __init__(
        self,
        size_distrib: SizeDistribution,
        adh_distrib: AdhesionDistribution,
    ) -> None:
        self.size_distrib = size_distrib
        self.adh_distrib = adh_distrib

    def rate(
        self,
        flow: Flow,
        t: int | None = None,
    ) -> NDArray:
        pass

    def rate_vectorized(
        self,
    ) -> NDArray:
        pass


class RocknRollModel(ResuspensionModel):
    @staticmethod
    def compute_rate(
        burst: NDArray,
        fluct: NDArray,
        fluct_var: NDArray,
    ) -> NDArray:
        # Compute the resuspension rate
        rate = burst * np.exp(-(fluct**2) / (2 * fluct_var)) / (0.5 * (1 + erf(fluct / np.sqrt(2 * fluct_var))))

        # Makes sure the rate is never superior to the burst frequency
        rate = np.minimum(rate, burst)

        return rate

    def rate(
        self,
        flow: Flow,
        t: int | None = None,
    ) -> NDArray:
        if t is None:  # Vectorized case
            fadh = np.tile(self.adh_distrib.fadh, (flow.nsteps, 1, 1))
            faero = np.tile(flow.faero, (self.adh_distrib.nbins, 1, 1)).transpose(
                1,
                2,
                0,
            )
            fluct_var = np.tile(
                flow.fluct_var,
                (self.adh_distrib.nbins, 1, 1),
            ).transpose(1, 2, 0)
            burst = np.tile(
                flow.burst[:, np.newaxis, np.newaxis],
                (1, self.size_distrib.nbins, self.adh_distrib.nbins),
            )
        else:  # Sequential case
            fadh = self.adh_distrib.fadh
            faero = np.tile(
                flow.faero[t, :].reshape(-1, 1),
                (1, self.adh_distrib.nbins),
            )
            fluct_var = np.tile(
                flow.fluct_var[t, :].reshape(-1, 1),
                (1, self.adh_distrib.nbins),
            )
            burst = flow.burst[t] * np.ones(
                [self.size_distrib.nbins, self.adh_distrib.nbins],
            )

        # Compute the aerodynamic fluctuation at detachment, and the variance of force fluctations
        fluct = fadh - faero

        return self.compute_rate(burst, fluct, fluct_var)


class StaticMomentBalance(RocknRollModel):
    @staticmethod
    def compute_rate(
        burst: NDArray,
        fluct: NDArray,
        fluct_var: NDArray,
    ) -> NDArray:
        return np.where(fluct > 0, 0.0, 0.5 * burst / np.pi)


class NonGaussianRocknRollModel(RocknRollModel):
    @staticmethod
    def compute_rate(
        burst: NDArray,
        fluct: NDArray,
        fluct_var: NDArray,
    ) -> NDArray:
        # Non-Gaussian distrib parameters
        a1 = 1.812562
        a2 = 1.463790
        bf = 0.343658

        # Compute the resuspension rate
        zdh = fluct / np.sqrt(fluct_var)
        rate = (
            (bf * burst)
            * ((zdh + a1) / (a2**2))
            * np.exp(-0.5 * ((zdh + a1) / a2) ** 2)
            / (1 - np.exp(-0.5 * ((zdh + a1) / a2) ** 2))
        )

        # Makes sure the rate is never superior to the burst frequency
        rate = np.minimum(rate, (0.5 * burst / np.pi))

        return rate
