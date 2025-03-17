import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray

from rnr.utils.config import setup_logging
from rnr.core.distribution import AdhesionDistribution, SizeDistribution
from rnr.core.flow import Flow


# Configure module logger from utils file
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

    def time_to_fraction(self, fraction: float) -> float:
        # Normalize resuspended fraction by the final value
        normalized_resuspended = self.resuspended_fraction / self.resuspended_fraction[-1]
        print(self.resuspended_fraction[-1])

        # Find the index where the fraction first exceeds the target percentage
        idx = np.searchsorted(normalized_resuspended, fraction)

        if idx == 0:
            return self.time[0]  # If the target is reached immediately

        if idx >= len(self.time):
            return self.time[-1]  # If the target is never reached

        # Linear interpolation for better accuracy
        t1, t2 = self.time[idx - 1], self.time[idx]
        f1, f2 = normalized_resuspended[idx - 1], normalized_resuspended[idx]
        t_target = t1 + (fraction - f1) * (t2 - t1) / (f2 - f1)

        return t_target

    def plot_distribution(self, t: int,) -> None:
        plt.clf()

        # plt.matshow(self.counts[t,:,:], norm=matplotlib.colors.LogNorm(vmin=self.counts[-1,:,:].min(), vmax=self.counts[0,:,:].max()))
        plt.matshow(self.counts[t,:,:])
        plt.colorbar()

        plt.savefig(f'figs/distribution_tstep={t}.png', dpi=300)

    def plot_remaining_fraction(self, scale: str='log',) -> None:
        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(self.time, self.remaining_fraction)

        ax.set_xscale(scale)

        fig.tight_layout()

        fig.savefig(f'figs/remaining_fraction.png', dpi=300)
        plt.close(fig)

    def plot_resuspended_fraction(self, scale: str='log',) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(self.time, self.resuspended_fraction, color='r',)

        ax.axvline(x=self.time_to_fraction(0.5), ymax=0.5*self.resuspended_fraction[-1]/1.1, color='r', linestyle='-',)
        ax.axvline(x=self.time_to_fraction(0.9), ymax=0.9*self.resuspended_fraction[-1]/1.1, color='r', linestyle='--',)
        ax.axvline(x=self.time_to_fraction(0.99), ymax=0.99*self.resuspended_fraction[-1]/1.1, color='r',linestyle=':',)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Resuspended fraction')

        ax.set_xscale(scale)
        ax.set_xlim(left=self.time[1],)
        ax.set_ylim(0.0, 1.1)

        ax.grid(axis='x', which='both')
        ax.grid(axis='y', which='major')

        fig.tight_layout()

        fig.savefig(f'figs/resuspended_fraction.png', dpi=300)
        plt.close(fig)

    def plot_instant_rate(self, scale: str = 'log',) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(self.time[:-1], self.instant_rate, color='r',)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Resuspension rate')

        ax.set_xscale(scale)
        ax.set_yscale('log')
        ax.set_xlim(self.time[1], self.time[-1])

        ax.grid(axis='x', which='both')
        ax.grid(axis='y', which='major')

        fig.tight_layout()

        fig.savefig(f'figs/instant_rate.png', dpi=300)
        plt.close(fig)
