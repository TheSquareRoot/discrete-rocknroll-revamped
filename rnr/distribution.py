import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray
from scipy.integrate import quad

from .config import setup_logging
from .utils import biasi_params, normal, log_norm


# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')


class DistributionBuilder:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                # Recursively convert dicts into DistributionBuilder instances
                setattr(self, key, DistributionBuilder(**value))
            else:
                setattr(self, key, value)

    def generate(self):
        pass


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
        plt.clf()

        plt.bar(self.radii, self.weights, **kwargs)

        plt.xscale(scale)
        plt.ylim([0.0, 1.1*np.max(self.weights)])

        plt.grid()

        plt.savefig('figs/size_distrib.png', dpi=300)


class SizeDistributionBuilder(DistributionBuilder):
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


class AdhesionDistribution:
    def __init__(self,
                 weights: NDArray[np.float64],
                 centers: NDArray[np.float64],
                 edges: NDArray[np.float64],
                 widths: NDArray[np.float64],
                 ) -> None:
        self.weights = weights
        self.centers = centers
        self.edges = edges
        self.widths = widths

    def plot(self, i: int, scale: str = 'log', **kwargs) -> None:
        plt.clf()

        plt.bar(self.centers[i], self.weights[i], width=self.widths[i], **kwargs)
        plt.xscale(scale)

        plt.savefig("figs/adh_distrib.png", dpi=300)


class AdhesionDistributionBuilder(DistributionBuilder):
    def __init__(self, size_distrib: SizeDistribution, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size_distrib = size_distrib

    def generate(self,) -> AdhesionDistribution:
        """
        Generate an adhesion force distribution from the parameters loaded from the config file.
        A log-normal distribution is assumed for each size bin.
        """
        # Get the log-normal median and spread parameters
        medians, spreads = biasi_params(*self.size_distrib.radii)

        edges = np.empty([self.sizedistrib.nbins, self.adhdistrib.nbins + 1])

        for i in range(self.sizedistrib.nbins):
            edges[i,:] = np.linspace(0.0, medians[i]*self.adhdistrib.fmax, self.adhdistrib.nbins + 1)

        widths = edges[:,1:] - edges[:,:-1]
        centers =  (edges[:,1:] + edges[:,:-1])/2

        # Compute the bin probabilities
        weights = np.zeros_like(centers)

        print(np.shape(weights))

        for i in range(self.sizedistrib.nbins):
            for j in range(self.adhdistrib.nbins):
                weights[i, j] = quad(log_norm, edges[i, j], edges[i, j + 1], args=(medians[i], spreads[i],))[0]

        # Instantiate the adhesion force distribution
        adh_distrib = AdhesionDistribution(weights, centers, edges, widths)

        return adh_distrib
