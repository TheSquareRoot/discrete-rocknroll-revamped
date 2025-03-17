import numpy as np

from numpy.typing import NDArray

from .config import setup_logging

# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')


def biasi_params(radii) -> tuple:
    """
    Return the log-normal median and spread parameters for an arbitrary number of radii.
    Uses the fit from Biasi (2001).

    NOTE: radii are in microns.
    """
    medians = np.array([0.016 - 0.0023 * (r ** 0.545) for r in radii])
    spreads = np.array([1.8 + 0.136 * (r ** 1.4) for r in radii])

    return medians, spreads


def force_jkr(radius: float, surface_energy: float,) -> float:
    """Adhesion force of a spherical particle on a flat surface according to JKR theory."""
    return 1.5 * np.pi * surface_energy * radius

def force_rabinovich(radius: float, asperity_radius: float, peaktopeak: float) -> float:
    """Adhesion force of a spherical particle on a rough surface according to the Rabinovich model."""
    pass

def log_norm(x: float, mean: float, stdv: float) -> float:
    """Log normal PDF. Geometric parameters are used."""
    proba_density = (1 / np.sqrt(2 * np.pi)) * (1 / (x * np.log(stdv))) * np.exp(
        -0.5 * (np.log(x / mean) / np.log(stdv)) ** 2)

    return proba_density


def normal(x: float, mean: float, stdv: float) -> float:
    """Normal PDF"""
    proba_density = np.exp(-(x - mean) ** 2 / (2 * (stdv ** 2))) / np.sqrt(2 * np.pi * (stdv ** 2))

    return proba_density

def median(values: NDArray, freqs: NDArray) -> float:
    # Compute total count and cumulative sum of frequencies
    total_count = np.sum(freqs)
    cum_freq = np.cumsum(freqs)

    # Find the bin index where cumulative frequency exceeds half the total count
    median_bin_idx = np.searchsorted(cum_freq, total_count / 2)

    # Interpolate the median within that bin
    if median_bin_idx == 0:
        med = values[0]  # If median is in the first bin, return its value
    else:
        bin_start = values[median_bin_idx - 1]  # Lower bin value
        bin_end = values[median_bin_idx]  # Upper bin value
        freq_below = cum_freq[median_bin_idx - 1]  # Cumulative freq below bin
        freq_in_bin = freqs[median_bin_idx]  # Frequency of the median bin

        # Linear interpolation formula
        med = bin_start + (bin_end - bin_start) * ((total_count / 2 - freq_below) / freq_in_bin)

    return med
