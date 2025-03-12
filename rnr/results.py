import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray

from .config import setup_logging
from .distribution import AdhesionDistribution, SizeDistribution
from .flow import Flow


# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')


class Results:
    def __init__(self,
                 adh_distrib: AdhesionDistribution,
                 size_distrib: SizeDistribution,
                 flow: Flow,
                 counts: NDArray[np.floating],
                 time: NDArray[np.floating],
                 ) -> None:

        self.adh_distrib = adh_distrib
        self.size_distrib = size_distrib
        self.flow = flow
        self.counts = counts
        self.time = time

    @property
    def remaining_fraction(self,) -> NDArray[np.floating]:
        return np.sum(self.counts, axis=(1,2))

    @property
    def instant_rate(self,) -> NDArray[np.floating]:
        rate = np.zeros(self.time.shape[0] - 1)
        for t in range(self.time.shape[0] - 1):
            rate[t] = self.remaining_fraction[t] - self.remaining_fraction[t+1]
        return rate

    def plot_distribution(self, t: int,) -> None:
        plt.clf()

        plt.matshow(self.counts[t,:,:], norm=matplotlib.colors.LogNorm(vmin=self.counts[-1,:,:].min(), vmax=self.counts[0,:,:].max()))
        plt.colorbar()

        plt.savefig(f'figs/distribution_tstep={t}.png', dpi=300)

    def plot_remaining_fraction(self,) -> None:
        plt.clf()

        plt.plot(self.time, self.remaining_fraction)

        plt.savefig(f'figs/remaining_fraction.png', dpi=300)

    def plot_instant_rate(self,) -> None:
        plt.clf()

        plt.plot(self.time[:-1], self.instant_rate)

        plt.savefig(f'figs/instant_rate.png', dpi=300)