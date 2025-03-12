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

    def plot_distribution(self, t: int,) -> None:
        plt.clf()

        plt.matshow(self.counts[t,:,:], norm=matplotlib.colors.LogNorm())
        plt.colorbar()

        plt.savefig(f'figs/distribution_tstep={t}.png', dpi=300)