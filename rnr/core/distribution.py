import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from rnr.utils.config import setup_logging
from rnr.utils.misc import (
    biasi_params,
    force_jkr,
    force_rabinovich,
    log_norm,
    median,
    normal,
)

# Configure module logger from utils file
logger = setup_logging(__name__, "logs/log.log")


class SizeDistribution:
    def __init__(
        self,
        modes: NDArray[np.floating],
        spreads: NDArray[np.floating],
        radii: NDArray[np.floating],
        counts: NDArray[np.floating],
    ) -> None:
        self.modes = modes
        self.spreads = spreads
        self.radii = radii
        self.weights = counts

    def __str__(self) -> str:
        return (
            f"SizeDistribution(\n"
            f"  modes: {np.shape(self.modes)},\n"
            f"  radii: {np.shape(self.radii)}   - [{self.radii[0]:.2e} ... {self.radii[-1]:.2e}],\n"
            f"  weights: {np.shape(self.weights)} - [{self.weights[0]:.2e} ... {self.weights[-1]:.2e}]\n"
            f")"
        )

    @property
    def nbins(
        self,
    ) -> int:
        return len(self.radii)

    @property
    def nmodes(
        self,
    ) -> int:
        return len(self.modes)

    @property
    def radii_meter(
        self,
    ) -> NDArray[np.floating]:
        return self.radii * 1e-6


class SizeDistributionBuilder:
    """
    A builder class for generating particle size distributions based on a set of user defined parameters.

    This class supports constructing both discrete and continuous multimodal particle size distributions
    using composite normal distributions.

    Attributes
    ----------
    nmodes : int
        Number of particle size modes.
    width : float
        Span of the modes support as multiples of the standard deviation.
    nbins : int
        Number of bins used to discretize the radius domain.
    modes : list of float
        Modal radii of the size distribution.
    spreads : list of float
        Spread (standard deviation) of each mode. A spread of zero indicates a delta function.
    weights : list of float
        Relative weights of each mode. Should sum to 1 for proper normalization.
    **kwargs : dict
        Additional keyword arguments (currently unused, included for extensibility).

    Methods
    -------
    generate() -> SizeDistribution
        Generate the particle size distribution using the provided parameters.
    """

    def __init__(
        self,
        nmodes: int,
        width: float,
        nbins: int,
        modes: list[float],
        spreads: list[float],
        weights: list[float],
        **kwargs: dict,
    ) -> None:
        self.nmodes = nmodes
        self.width = width
        self.nbins = nbins
        self.modes = modes
        self.spreads = spreads
        self.weights = weights

    def _radius_domain(
        self,
    ) -> NDArray[np.float64]:
        """Computes the radius domain of the size distribution."""
        lower_bounds, upper_bounds = [], []

        for i in range(self.nmodes):
            lower_bounds.append(self.modes[i] - self.width * self.spreads[i])
            upper_bounds.append(self.modes[i] + self.width * self.spreads[i])

        return np.logspace(
            np.log10(min(lower_bounds)),
            np.log10(max(upper_bounds)),
            num=self.nbins,
        )

    def _generate_no_spread(self) -> SizeDistribution:
        """Generate a discrete particle size distribution."""
        size_distrib = SizeDistribution(
            modes=np.array(self.modes),
            spreads=np.zeros_like(self.modes),
            radii=np.array(self.modes),
            counts=np.array(self.weights),
        )

        return size_distrib

    def _generate_with_spread(self) -> SizeDistribution:
        """Generate a continuous particle size distribution.The result is a composite of normal distributions."""
        # TODO: integrate the normal distribution to compute the weights

        # Get the domain over which the size distribution will be generated
        rad_domain = self._radius_domain()
        logger.debug(
            f"Size domain is [{np.min(rad_domain):.2f}, {np.max(rad_domain):.2f}]",
        )

        # Compute the distribution function
        weights = np.zeros_like(rad_domain)

        for i in range(self.nbins):
            for j in range(self.nmodes):
                weights[i] += self.weights[j] * normal(
                    rad_domain[i],
                    self.modes[j],
                    self.spreads[j],
                )

        # Normalize weights
        weights /= np.sum(weights)

        # Instantiate the size distribution
        size_distrib = SizeDistribution(
            modes=np.array(self.modes),
            spreads=np.array(self.spreads),
            radii=rad_domain,
            counts=weights,
        )

        return size_distrib

    def generate(
        self,
    ) -> SizeDistribution:
        if any(x == 0 for x in self.spreads):
            return self._generate_no_spread()
        return self._generate_with_spread()


class AdhesionDistribution:
    """
    Represents the distribution of adhesion forces for each size bin.

    nr: number of size bins
    na: number of adhesion bins

    Attributes:
        weights (NDArray[nr x na]): The probability associated to each adhesion force bin. Each line adds up to 1.
        fadh_norm (NDArray[nr x na]): The normalized adhesion force value for each bin.
        norm_factors (NDArray[nr]): Normalization factor for each size bin (unit: Newton) .
    """

    def __init__(
        self,
        weights: NDArray[np.floating],
        fadh_norm: NDArray[np.floating],
        norm_factors: NDArray[np.floating],
    ) -> None:
        self.weights = weights
        self.fadh_norm = fadh_norm
        self.norm_factors = norm_factors

    def __str__(self) -> str:
        """String representation of the adhesion distribution."""
        return (
            f"AdhesionDistribution(\n"
            f"  weights: {np.shape(self.weights)}   - [{self.weights[0, 0]:.2e} ... {self.weights[-1, -1]:.2e}],\n"
            f"  fadh_norm: {np.shape(self.fadh_norm)} - [{self.fadh_norm[0, 0]:.2e} ... {self.fadh_norm[-1, -1]:.2e}],\n"
            f"  norm_factors: {np.shape(self.norm_factors)} - [{self.norm_factors[0, 0]:.2e} ... {self.norm_factors[-1, -1]:.2e}]\n"
            f")"
        )

    @property
    def nbins(
        self,
    ) -> int:
        """Returns the number of adhesion bins."""
        return self.weights.shape[1]

    @property
    def fadh(
        self,
    ) -> NDArray[np.floating]:
        """Returns a denormalized adhesion force array"""
        return self.fadh_norm * self.norm_factors

    # Some statistical quantities --------------------------------------------------------------------------------------
    def median(self, i: int, *, norm: bool = True) -> float:
        """
        Computes the median adhesion force for a given size bin.

        Args:
            i (int): The size bin index.
            norm (bool, optional): Whether to use the normalized adhesion force.
        Returns:
            float: The median adhesion force.
        """
        if norm:
            return median(self.fadh_norm[i], self.weights[i])

        return median(self.fadh[i], self.weights[i])

    def mean(self, i: int, *, norm: bool = True) -> float:
        """
        Computes the median adhesion force for a given size bin.

        Args:
            i (int): The size bin index.
            norm (bool, optional): Whether to use the normalized adhesion force.
        Returns:
            float: The median adhesion force.
        """
        if norm:
            return float(np.average(self.fadh_norm[i], weights=self.weights[i]))
        return float(np.average(self.fadh[i], weights=self.weights[i]))

    def geo_spread(self, i: int, *, norm: bool = True) -> float:
        mean, med = self.mean(i, norm=norm), self.median(i, norm=norm)

        return np.exp(np.sqrt(2 * np.log(mean / med)))


class AdhesionDistributionBuilder:
    def __init__(
        self,
        size_distrib: SizeDistribution,
        nbins: int,
        fmax: float,
        dist_params: str,
        adhesion_model: str,
        medians: list[float] | None = None,
        spreads: list[float] | None = None,
        surface_energy: float | None = None,
        asperity_radius: float | None = None,
        peaktopeak: float | None = None,
        **kwargs: dict,
    ) -> None:
        self.size_distrib = size_distrib
        self.nbins = nbins
        self.fmax = fmax
        self.dist_params = dist_params
        self.medians = medians
        self.spreads = spreads

        self.adhesion_model = adhesion_model
        self.surface_energy = surface_energy
        self.asperity_radius = asperity_radius
        self.peaktopeak = peaktopeak

    def _compute_distribution_params(
        self,
        radii: NDArray[np.floating],
    ) -> tuple:
        """Wrapper to set the correct distribution medians and spreads."""
        if self.dist_params == "biasi":
            return biasi_params(
                radii,
            )
        if self.dist_params == "custom":
            # Distinguish between the spread and no spread case.
            # If there is a radius spread, it is unlikely the user will provide medians and spreads for each size bin
            # So instead, values have to be derived from the user inputs.
            # If only set of parameters is used, it is applied to all size bins
            if len(self.medians) == 1:
                return (
                    np.ones(self.size_distrib.nbins) * self.medians[0],
                    np.ones(self.size_distrib.nbins) * self.spreads[0],
                )
            # If there as many sets of params are modes, one set is assigned to each mode
            if len(self.medians) == self.size_distrib.nmodes:
                # TODO: implement what happens when a set of parameters are provided for each mode
                raise NotImplementedError(
                    "Custom per-mode distribution parameters are not yet implemented.",
                )
            # Otherwise wtf are they doing anyway
            raise ValueError(
                f"Incorrect number of parameters provided: {len(self.medians)}",
            )
        raise ValueError(f"Unknown distribution parameter {self.dist_params}")

    def _compute_norm_factor(
        self,
        radius: float,
    ) -> float:
        """Wrapper to call the correct adhesion force function."""
        if self.adhesion_model == "JKR":
            return force_jkr(
                radius,
                self.surface_energy,
            )
        if self.adhesion_model == "Rabinovich":
            return force_rabinovich(
                radius,
                self.asperity_radius,
                self.peaktopeak,
            )
        raise ValueError(f"Unknown adhesion model: {self.adhesion_model}")

    def generate(
        self,
    ) -> AdhesionDistribution:
        """
        Generate an adhesion force distribution from the parameters loaded from the utils file.
        A log-normal distribution is assumed for each size bin.
        """
        # Initialize arrays
        edges = np.empty([self.size_distrib.nbins, self.nbins + 1])
        fadh_norm = np.empty([self.size_distrib.nbins, self.nbins])
        weights = np.empty_like(fadh_norm)

        # Get the log-normal median and spread parameters
        medians, spreads = self._compute_distribution_params(self.size_distrib.radii)

        # Compute the normalization factor for each size bin
        norm_factors = np.array(
            [
                [
                    self._compute_norm_factor(r * 1e-6),
                ]
                for r in self.size_distrib.radii
            ],
        )

        for i in range(self.size_distrib.nbins):
            edges[i, :] = np.linspace(0.0, medians[i] * self.fmax, self.nbins + 1)

            fadh_norm[i, :] = (edges[i, 1:] + edges[i, :-1]) / 2

            for j in range(self.nbins):
                weights[i, j] = quad(
                    log_norm,
                    edges[i, j],
                    edges[i, j + 1],
                    args=(
                        medians[i],
                        spreads[i],
                    ),
                )[0]

            # Normalize weights
            weights[i, :] /= weights[i, :].sum()

        # Instantiate the adhesion force distribution
        adh_distrib = AdhesionDistribution(
            weights,
            fadh_norm,
            norm_factors,
        )
        return adh_distrib
