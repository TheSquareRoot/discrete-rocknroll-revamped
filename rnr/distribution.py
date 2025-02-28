import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray
from scipy.integrate import quad

from .config import setup_logging
from .utils import biasi_params, normal, log_norm


# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')


class SizeDistribution:
    def __init__(self,
                 modes: NDArray[np.float64],
                 spreads: NDArray[np.float64],
                 radii: NDArray[np.float64],
                 counts: NDArray[np.float64]):
        self.modes = modes
        self.spreads = spreads
        self.radii = radii
        self.weights = counts

    def plot(self, scale: str = 'log', **kwargs) -> None:
        """Basic bar plot of the size distribution."""
        plt.bar(self.radii, self.weights, **kwargs)

        plt.xscale(scale)
        plt.ylim([0.0, 1.1*np.max(self.weights)])

        plt.grid()

        plt.savefig('figs/size_distrib.png', dpi=300)


class SizeDistributionBuilder:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                # Recursively convert dicts into DistributionBuilder instances
                setattr(self, key, SizeDistributionBuilder(**value))
            else:
                setattr(self, key, value)

    def _radius_domain(self,) -> NDArray[np.float64]:
        """Computes the radius domain of the size distribution."""
        lower_bounds, upper_bounds = [], []

        for i in range(self.sizedistrib.nmodes):
            lower_bounds.append(self.sizedistrib.modes[i] - self.sizedistrib.width * self.sizedistrib.spreads[i])
            upper_bounds.append(self.sizedistrib.modes[i] + self.sizedistrib.width * self.sizedistrib.spreads[i])

        return np.logspace(np.log10(min(lower_bounds)), np.log10(max(upper_bounds)), num=self.sizedistrib.nbins)

    def generate(self,) -> SizeDistribution:
        """
        Generate a particle size distribution from the parameters loaded from the config file.
        The result is a composite of normal distributions.
        """
        # Get the domain over which the size distribution will be generated
        rad_domain = self._radius_domain()
        logger.debug(f'Size domain is [{np.min(rad_domain):.2f}, {np.max(rad_domain):.2f}]')

        # Compute the distribution function
        weights = np.zeros_like(rad_domain)

        for i in range(self.sizedistrib.nbins):
            for j in range(self.sizedistrib.nmodes):
                weights[i] += self.sizedistrib.weights[j] * normal(rad_domain[i], self.sizedistrib.modes[j], self.sizedistrib.spreads[j])

        # Instantiate the size distribution
        size_distrib = SizeDistribution(np.array(self.sizedistrib.modes),
                                        np.array(self.sizedistrib.spreads),
                                        rad_domain,
                                        weights)

        return size_distrib