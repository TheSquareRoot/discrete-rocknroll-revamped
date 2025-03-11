import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray
from scipy.integrate import quad

from .config import setup_logging
from .utils import biasi_params, force_jkr, log_norm, normal


# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')


class SizeDistribution:
    def __init__(self,
                 modes: NDArray[np.floating],
                 spreads: NDArray[np.floating],
                 radii: NDArray[np.floating],
                 counts: NDArray[np.floating]):
        self.modes = modes
        self.spreads = spreads
        self.radii = radii
        self.weights = counts

    def __str__(self) -> str:
        return (
            f"AdhesionDistribution(\n"
            f"  modes: {np.shape(self.modes)},\n"
            f"  radii: {np.shape(self.radii)}   - [{self.radii[0]:.2e} ... {self.radii[-1]:.2e}],\n"
            f"  weights: {np.shape(self.weights)} - [{self.weights[0]:.2e} ... {self.weights[-1]:.2e}]\n"
            f")"
        )

    @property
    def nbins(self,) -> int:
        return len(self.radii)

    @property
    def radii_meter(self,) -> NDArray[np.floating]:
        return self.radii * 1e-6

    def plot(self, scale: str = 'log', **kwargs) -> None:
        """Basic bar plot of the size distribution."""
        plt.clf()

        plt.bar(self.radii, self.weights, **kwargs)

        plt.xscale(scale)
        plt.ylim([0.0, 1.1*np.max(self.weights)])

        plt.grid()

        plt.savefig('figs/size_distrib.png', dpi=300)


class SizeDistributionBuilder:
    def __init__(self,
                 nmodes: int,
                 width: float,
                 nbins: int,
                 modes: list[float],
                 spreads: list[float],
                 weights: list[float],
                 **kwargs,
                 ) -> None:

        self.nmodes = nmodes
        self.width = width
        self.nbins = nbins
        self.modes = modes
        self.spreads = spreads
        self.weights = weights

    def _radius_domain(self,) -> NDArray[np.float64]:
        """Computes the radius domain of the size distribution."""
        lower_bounds, upper_bounds = [], []

        for i in range(self.nmodes):
            lower_bounds.append(self.modes[i] - self.width * self.spreads[i])
            upper_bounds.append(self.modes[i] + self.width * self.spreads[i])

        return np.logspace(np.log10(min(lower_bounds)), np.log10(max(upper_bounds)), num=self.nbins)

    def _generate_no_spread(self,) -> SizeDistribution:
        size_distrib = SizeDistribution(np.array(self.modes),
                                       np.zeros_like(self.modes),
                                       np.array(self.modes),
                                       np.ones_like(self.modes)
                                       )

        return size_distrib

    def _generate_with_spread(self,) -> SizeDistribution:
        """
        Generate a particle size distribution from the parameters loaded from the config file.
        The result is a composite of normal distributions.
        """
        # TODO: integrate the normal distribution to compute the weights

        # Get the domain over which the size distribution will be generated
        rad_domain = self._radius_domain()
        logger.debug(f'Size domain is [{np.min(rad_domain):.2f}, {np.max(rad_domain):.2f}]')

        # Compute the distribution function
        weights = np.zeros_like(rad_domain)

        for i in range(self.nbins):
            for j in range(self.nmodes):
                weights[i] += self.weights[j] * normal(rad_domain[i], self.modes[j], self.spreads[j])

        # Instantiate the size distribution
        size_distrib = SizeDistribution(np.array(self.modes),
                                        np.array(self.spreads),
                                        rad_domain,
                                        weights)

        return size_distrib

    def generate(self,) -> SizeDistribution:
        if hasattr(self, 'spreads'):
            return self._generate_with_spread()
        else:
            return self._generate_no_spread()


class AdhesionDistribution:
    def __init__(self,
                 weights: NDArray[np.floating],
                 fadh_norm: NDArray[np.floating],
                 norm_factors: NDArray[np.floating],
                 ) -> None:
        self.weights = weights
        self.fadh_norm = fadh_norm
        self.norm_factors = norm_factors

    def __str__(self) -> str:
        return (
            f"AdhesionDistribution(\n"
            f"  weights: {np.shape(self.weights)}   - [{self.weights[0,0]:.2e} ... {self.weights[-1,-1]:.2e}],\n"
            f"  fadh_norm: {np.shape(self.fadh_norm)} - [{self.fadh_norm[0,0]:.2e} ... {self.fadh_norm[-1,-1]:.2e}],\n"
            f"  norm_factors: {np.shape(self.norm_factors)} - [{self.norm_factors[0,0]:.2e} ... {self.norm_factors[-1,-1]:.2e}]\n"
            f")"
        )

    @property
    def fadh(self,) -> NDArray[np.floating]:
        """Return a denormalized adhesion force array"""
        return self.fadh_norm * self.norm_factors

    def plot(self, i: int, norm: bool = True, scale: str = 'log', **kwargs) -> None:
        plt.clf()

        if norm:
            plt.plot(self.fadh_norm[i], self.weights[i], **kwargs)
            plt.xlabel('Normalized adhesion force')
        else:
            plt.plot(self.fadh[i], self.weights[i], **kwargs)
            plt.xlabel('Adhesion force [N]')

        # Set scale
        plt.xscale(scale)

        plt.savefig("figs/adh_distrib.png", dpi=300)


class AdhesionDistributionBuilder:
    def __init__(self,
                 size_distrib: SizeDistribution,
                 nbins: int,
                 fmax: float,
                 surface_energy: float,
                 **kwargs,
                 ) -> None:

        self.size_distrib = size_distrib
        self.nbins = nbins
        self.fmax = fmax
        self.surface_energy = surface_energy

    def generate(self,) -> AdhesionDistribution:
        """
        Generate an adhesion force distribution from the parameters loaded from the config file.
        A log-normal distribution is assumed for each size bin.
        """
        # Initialize arrays
        edges = np.empty([self.size_distrib.nbins, self.nbins + 1])
        fadh_norm = np.empty([self.size_distrib.nbins, self.nbins])
        weights = np.empty_like(fadh_norm)

        # Get the log-normal median and spread parameters
        medians, spreads = biasi_params(self.size_distrib.radii)

        # Compute the normalization factor for each size bin
        norm_factors = np.array([[force_jkr(self.surface_energy, r*1e-6),] for r in self.size_distrib.radii])

        for i in range(self.size_distrib.nbins):
            edges[i,:] = np.linspace(0.0, medians[i]*self.fmax, self.nbins + 1)

            fadh_norm[i,:] =  (edges[i,1:] + edges[i,:-1])/2

            for j in range(self.nbins):
                weights[i, j] = quad(log_norm, edges[i, j], edges[i, j + 1], args=(medians[i], spreads[i],))[0]

        # Instantiate the adhesion force distribution
        adh_distrib = AdhesionDistribution(weights,
                                           fadh_norm,
                                           norm_factors,
                                           )
        return adh_distrib
