import numpy as np
from numpy.typing import NDArray
from scipy import stats

from rnr.utils.config import setup_logging
from rnr.utils.misc import (
    biasi_params,
    force_jkr,
    force_rabinovich,
    median,
)

# Configure module logger from utils file
logger = setup_logging(__name__, "logs/log.log")


class SizeDistribution:
    """
    Represents a particle size distribution, including radii, modal parameters, and associated weights.

    This class stores the radii bins and corresponding statistical parameters of a discrete or continuous
    size distribution. It also provides convenience properties for unit conversion and metadata.

    Parameters
    ----------
    modes : NDArray
        The modal radius (in micrometers) of each size mode of the distribution.
    spreads : NDArray
        The standard deviation associated with each mode.
    radii : NDArray
        The center value of each size bin, in micrometers.
    weights : NDArray
        The probability weight associated with each radius bin. Must sum to 1.

    Properties
    ----------
    nbins : int
        Number of radius bins in the distribution (i.e., the number of values in `radii`).
    nmodes : int
        Number of modes in the distribution.
    radii_meter : NDArray
        Radii converted to meters.
    """

    def __init__(
        self,
        modes: NDArray,
        spreads: NDArray,
        radii: NDArray,
        weights: NDArray,
    ) -> None:
        self.modes = modes
        self.spreads = spreads
        self.radii = radii
        self.weights = weights

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
    ) -> NDArray:
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
        coeffs: list[float],
        **kwargs: dict,  # noqa: ARG002
    ) -> None:
        self.nmodes = nmodes
        self.width = width
        self.nbins = nbins
        self.modes = modes
        self.spreads = spreads
        self.coeffs = coeffs

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
            weights=np.array(self.coeffs),
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
                weights[i] = self.coeffs[j] * stats.norm.pdf(rad_domain[i], loc=self.modes[j], scale=self.spreads[j])

        # Normalize weights
        weights /= np.sum(weights)

        # Instantiate the size distribution
        size_distrib = SizeDistribution(
            modes=np.array(self.modes),
            spreads=np.array(self.spreads),
            radii=rad_domain,
            weights=weights,
        )

        return size_distrib

    def generate(
        self,
    ) -> SizeDistribution:
        """Wrapper for generator methods."""
        if any(x == 0 for x in self.spreads):
            return self._generate_no_spread()
        return self._generate_with_spread()


class AdhesionDistribution:
    """
    Represents the distribution of adhesion forces across different particle size bins.

    This class holds both the normalized adhesion force values and the associated probability weights
    for each particle size bin. It also provides utilities for statistical analysis such as median,
    mean, and geometric spread of the adhesion forces.

    Notation
    --------
    nr : int
        Number of particle size bins.
    na : int
        Number of adhesion force bins per size bin.

    Parameters
    ----------
    weights : NDArray[nr x na]
        The probability associated with each adhesion force bin for each size bin.
        Each row should sum to 1.
    fadh_norm : NDArray[nr x na]
        The normalized adhesion force values. These are scaled to [0, fmax] and are dimensionless.
    norm_factors : NDArray[nr]
        Normalization factors (in Newtons) used to scale `fadh_norm` to actual force values per size bin.

    Properties
    ----------
    nbins : int
        Number of adhesion bins (na).
    fadh : NDArray[nr x na]
        Denormalized adhesion forces obtained by multiplying `fadh_norm` by `norm_factors`.

    Methods
    -------
    median(i: int, norm: bool = True) -> float
        Returns the median adhesion force for the i-th size bin, optionally normalized.
    mean(i: int, norm: bool = True) -> float
        Returns the mean adhesion force for the i-th size bin, optionally normalized.
    geo_spread(i: int, norm: bool = True) -> float
        Returns the geometric spread of the adhesion force distribution for the i-th size bin.
    """

    def __init__(
        self,
        weights: NDArray,
        fadh_norm: NDArray,
        norm_factors: NDArray,
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
        return self.weights.shape[1]

    @property
    def fadh(
        self,
    ) -> NDArray:
        return self.fadh_norm * self.norm_factors

    # Some statistical quantities --------------------------------------------------------------------------------------
    def median(self, i: int, *, norm: bool = True) -> float:
        """Computes the median for a given size bin."""
        if norm:
            return median(self.fadh_norm[i], self.weights[i])

        return median(self.fadh[i], self.weights[i])

    def mean(self, i: int, *, norm: bool = True) -> float:
        """Computes the mean for a given size bin."""
        if norm:
            return float(np.average(self.fadh_norm[i], weights=self.weights[i]))
        return float(np.average(self.fadh[i], weights=self.weights[i]))

    def geo_spread(self, i: int, *, norm: bool = True) -> float:
        mean, med = self.mean(i, norm=norm), self.median(i, norm=norm)

        return np.exp(np.sqrt(2 * np.log(mean / med)))


class AdhesionDistributionBuilder:
    """
    Builder class for generating adhesion force distributions based on a given particle size distribution.

    This class supports two main modes of adhesion force generation:
    1. Biasi-based mode: generates a lognormal adhesion force distribution using Biasi parameters.
    2. Custom mode: constructs a user-defined distribution using a combination of known continuous
       probability distributions from `scipy.stats`.

    The generated adhesion distribution accounts for a specific adhesion force model, such as "JKR"
    or "Rabinovich", and is discretized over a fixed number of bins up to a maximum adhesion force.

    Parameters
    ----------
    size_distrib : SizeDistribution
        A size distribution object providing radii and weights.
    nbins : int
        Number of bins used to discretize the adhesion force domain.
    fmax : float
        Maximum normalized adhesion force (upper bound of the support for the distribution).
    adhesion_model : str
        Adhesion model to use for normalization. Supported values: "JKR", "Rabinovich". Only JKR implemented for now!
    biasi : bool
        If True, use Biasi parameters to generate lognormal adhesion force distributions.
        If False, use custom user-defined distributions via `scipy.stats`.
    distnames : list of str, optional
        List of distribution names from `scipy.stats` to be used in custom mode.
    distshapes : list of float or list of list of float, optional
        Shape parameters for each distribution in `distnames`.
    loc : list of float, optional
        Location parameters for each distribution.
    scale : list of float, optional
        Scale parameters for each distribution.
    surface_energy : float, optional
        Surface energy used in the JKR adhesion force model.
    asperity_radius : float, optional
        Asperity radius used in the Rabinovich model.
    peaktopeak : float, optional
        Surface roughness amplitude used in the Rabinovich model.
    **kwargs : dict
        Additional keyword arguments for future extensibility (currently unused).

    Methods
    -------
    generate() -> AdhesionDistribution
        Generates the adhesion force distribution using the configured parameters.
    """

    def __init__(
        self,
        size_distrib: SizeDistribution,
        nbins: int,
        fmax: float,
        adhesion_model: str,
        biasi: bool,  # noqa: FBT001
        distnames: list[str] | None = None,
        loc: list[float] | None = None,
        scale: list[float] | None = None,
        distshapes: list[float] | None = None,
        surface_energy: float | None = None,
        asperity_radius: float | None = None,
        peaktopeak: float | None = None,
        **kwargs: dict,  # noqa: ARG002
    ) -> None:
        self.size_distrib = size_distrib
        self.nbins = nbins
        self.fmax = fmax

        self.biasi = biasi
        self.distnames = distnames
        self.distshapes = distshapes
        self.loc = loc
        self.scale = scale

        self.adhesion_model = adhesion_model
        self.surface_energy = surface_energy
        self.asperity_radius = asperity_radius
        self.peaktopeak = peaktopeak

    def _get_distribution(self) -> list[stats.rv_continuous]:
        """Get distribution objects from `scipy.stats`"""
        distlist = []
        for name in self.distnames:
            try:
                distlist.append(getattr(stats, name))
            except AttributeError as e:
                logger.exception(f"Unknown distribution: {name}")
                raise AttributeError from e
        return distlist

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

    def _compute_biasi_weights(self) -> tuple[NDArray, NDArray]:
        """
        Generate an adhesion force lognormal distribution using Biasi parameters.
        """
        # Initialize arrays
        fadh_norm = np.empty([self.size_distrib.nbins, self.nbins])
        weights = np.zeros_like(fadh_norm)

        # Set Biasi parameters
        medians, spreads = biasi_params(self.size_distrib.radii)

        # Compute weights from a lognormal distribution
        for i in range(self.size_distrib.nbins):
            fadh_norm[i, :] = np.linspace(0, self.fmax, num=self.nbins + 1)[1:]
            weights[i, :] = stats.lognorm.pdf(fadh_norm[i, :], *[np.log(spreads[i])], loc=0, scale=medians[i])

            # Normalize weights
            weights[i, :] /= weights[i, :].sum()

        return fadh_norm, weights

    def _compute_custom_weights(self) -> tuple[NDArray, NDArray]:
        """
        Generate a custom adhesion force distribution as a sum of distributions available in scipy.stats
        Restricted to unimodal discrete particle size distributions.
        """
        # Initialize arrays
        fadh_norm = np.empty([1, self.nbins])
        weights = np.zeros_like(fadh_norm)

        # Load distributions from scipy.stats
        distlist = self._get_distribution()

        # Compute the sum of user given distributions
        for i in range(len(distlist)):
            fadh_norm[0, :] = np.linspace(0, self.fmax, num=self.nbins + 1)[1:]
            proba = distlist[i].pdf(fadh_norm[0, :], *self.distshapes[i], loc=self.loc[i], scale=self.scale[i])
            weights[0, :] += proba / proba.sum()

        # Normalize weights
        weights /= weights.sum()

        return fadh_norm, weights

    def generate(
        self,
    ) -> AdhesionDistribution:
        """
        Generate an adhesion force distribution from the parameters loaded from the utils file.
        A log-normal distribution is assumed for each size bin.
        """
        # Compute the normalization factor for each size bin
        norm_factors = np.array(
            [
                [
                    self._compute_norm_factor(r * 1e-6),
                ]
                for r in self.size_distrib.radii
            ],
        )

        # Compute the weights
        if self.biasi:
            fadh_norm, weights = self._compute_biasi_weights()
        else:
            fadh_norm, weights = self._compute_custom_weights()

        # Instantiate the adhesion force distribution
        adh_distrib = AdhesionDistribution(
            weights=weights,
            fadh_norm=fadh_norm,
            norm_factors=norm_factors,
        )
        return adh_distrib
