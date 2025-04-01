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
        if t is None:
            adh_tiled = np.tile(self.adh_distrib.fadh, (self.flow.nsteps, 1, 1))
            aero_tiled = np.tile(self.flow.faero, (self.adh_distrib.nbins, 1, 1)).transpose(1, 2, 0)
            burst_tiled = np.tile(self.flow.burst[:, np.newaxis, np.newaxis],
                                  (1, self.size_distrib.nbins, self.adh_distrib.nbins))
        else:
            adh_tiled = self.adh_distrib.fadh
            aero_tiled = np.tile(self.flow.faero[t, :].reshape(-1, 1), (1, self.adh_distrib.nbins))
            burst_tiled = self.flow.burst[t] * np.ones([self.size_distrib.nbins, self.adh_distrib.nbins])

        diff = adh_tiled - aero_tiled
        fluct_tiled = 0.04 * (aero_tiled ** 2)

        rate = burst_tiled * np.exp(- (diff ** 2) / (2 * fluct_tiled)) / (
                    0.5 * (1 + erf(diff / np.sqrt(2 * fluct_tiled))))

        return rate

    # def rate(self, t: int) -> NDArray[np.floating]:
    #     # Construct the Fadh - Faero array
    #     adh_tiled = self.adh_distrib.fadh
    #     aero_tiled = np.tile(self.flow.faero[t,:].reshape(-1,1), (1, self.adh_distrib.nbins))
    #     burst_tiled = self.flow.burst[t] * np.ones([self.size_distrib.nbins, self.adh_distrib.nbins])
    #
    #     diff = adh_tiled - aero_tiled
    #     fluct_tiled = 0.04 * (aero_tiled ** 2)
    #
    #     rate = burst_tiled * np.exp(- (diff ** 2) / (2 * fluct_tiled)) / (
    #                 0.5 * (1 + erf(diff / np.sqrt(2 * fluct_tiled))))
    #
    #     return rate
    #
    # def rate_vectorized(self,) -> NDArray[np.floating]:
    #     """
    #     Computes the resuspension rate for all time steps and stores it in a single array.
    #     This leads to faster simulation times, but can request a LOT of memory.
    #     """
    #     # Construct the Fadh - Faero array
    #     adh_tiled = np.tile(self.adh_distrib.fadh, (self.flow.nsteps, 1, 1))
    #     aero_tiled = np.tile(self.flow.faero, (self.adh_distrib.nbins, 1, 1)).transpose(1,2,0)
    #     burst_tiled = np.tile(self.flow.burst[:, np.newaxis, np.newaxis], (1, self.size_distrib.nbins, self.adh_distrib.nbins))
    #
    #     diff = adh_tiled - aero_tiled
    #     fluct_tiled = 0.04 * (aero_tiled ** 2)
    #
    #     rate = burst_tiled * np.exp(- (diff ** 2) / (2 * fluct_tiled)) / (0.5 * (1 + erf(diff / np.sqrt( 2 * fluct_tiled))))
    #
    #     return rate
