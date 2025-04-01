import numpy as np
from scipy.special import erf

from numpy.typing import NDArray

from rnr.utils.config import setup_logging
from rnr.core.distribution import AdhesionDistribution, SizeDistribution
from rnr.core.flow import Flow


# Configure module logger from utils file
logger = setup_logging(__name__, 'logs/log.log')


class ResuspensionModel:
    def __init__(self,
                 size_distrib: SizeDistribution,
                 adh_distrib: AdhesionDistribution,
                 flow: Flow,
                 ) -> None:
        self.size_distrib = size_distrib
        self.adh_distrib = adh_distrib
        self.flow = flow

    def rate(self, t: int) -> NDArray[np.floating]:
        pass

    def rate_vectorized(self,) -> NDArray[np.floating]:
        pass


class RocknRollModel(ResuspensionModel):

    def rate(self, t: int = None,):
        """
        Computes the resuspension rate using the quasi-static formulation of the Rock'n'Roll model.

        Args:
            t (int, optional): The time-step at which to compute the rate. Defaults to None.

        Returns:
            rate (NDArray): If a timestep is provided, the rate at time t is returned. Otherwise, the whole rate array
                            is computed an returned.
        """
        if t is None: # Sequential case
            adh_tiled = np.tile(self.adh_distrib.fadh, (self.flow.nsteps, 1, 1))
            aero_tiled = np.tile(self.flow.faero, (self.adh_distrib.nbins, 1, 1)).transpose(1, 2, 0)
            burst_tiled = np.tile(self.flow.burst[:, np.newaxis, np.newaxis],
                                  (1, self.size_distrib.nbins, self.adh_distrib.nbins))
        else: # Vectorized case
            adh_tiled = self.adh_distrib.fadh
            aero_tiled = np.tile(self.flow.faero[t, :].reshape(-1, 1), (1, self.adh_distrib.nbins))
            burst_tiled = self.flow.burst[t] * np.ones([self.size_distrib.nbins, self.adh_distrib.nbins])

        # Compute the aerodynamic fluctuation at detachment, and the variance of force fluctations
        fluct = adh_tiled - aero_tiled
        fluct_var = 0.04 * (aero_tiled ** 2)

        # Compute the resuspension rate
        rate = burst_tiled * np.exp(- (fluct ** 2) / (2 * fluct_var)) / (
                    0.5 * (1 + erf(fluct / np.sqrt(2 * fluct_var))))

        # Makes sure the rate is never superior to the burst frequency
        rate = np.minimum(rate, burst_tiled)

        return rate
