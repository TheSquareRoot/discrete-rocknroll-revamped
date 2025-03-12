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
    def resuspended_fraction(self,) -> NDArray[np.floating]:
        return 1 - np.sum(self.counts, axis=(1,2))

    @property
    def instant_rate(self) -> NDArray[np.floating]:
        return self.remaining_fraction[:-1] - self.remaining_fraction[1:]

    def plot_distribution(self, t: int,) -> None:
        plt.clf()

        # plt.matshow(self.counts[t,:,:], norm=matplotlib.colors.LogNorm(vmin=self.counts[-1,:,:].min(), vmax=self.counts[0,:,:].max()))
        plt.matshow(self.counts[t,:,:])
        plt.colorbar()

        plt.savefig(f'figs/distribution_tstep={t}.png', dpi=300)

    def plot_remaining_fraction(self, scale: str='log',) -> None:
        plt.clf()

        plt.plot(self.time, self.remaining_fraction)

        plt.xscale(scale)

        plt.savefig(f'figs/remaining_fraction.png', dpi=300)

    def plot_resuspended_fraction(self, scale: str='log',) -> None:
        plt.clf()

        plt.plot(self.time, self.resuspended_fraction)

        plt.xscale(scale)
        plt.ylim(0.0, 1.1)

        plt.grid(axis='x', which='both')
        plt.grid(axis='y', which='major')

        plt.savefig(f'figs/resuspended_fraction.png', dpi=300)

    def plot_instant_rate(self, scale: str = 'log',) -> None:
        plt.clf()

        plt.plot(self.time[:-1], self.instant_rate)

        plt.xscale(scale)
        plt.yscale('log')

        plt.grid(axis='x', which='both')
        plt.grid(axis='y', which='major')

        plt.savefig(f'figs/instant_rate.png', dpi=300)