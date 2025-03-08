import toml
import sys

import numpy as np

from .config import setup_logging

# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')


def biasi_params(*radii) -> tuple:
    """
    Return the log-normal median and spread parameters for an arbitrary number of radii.
    Uses the fit from Biasi (2001).

    NOTE: radii are in microns.
    """
    medians = [0.016 - 0.0023 * (r ** 0.545) for r in radii]
    spreads = [1.8 + 0.136 * (r ** 1.4) for r in radii]

    return medians, spreads


def force_jkr(surface_energy: float, radius: float) -> float:
    """Adhesion force of a spherical particle on a flat surface according to JKR theory."""
    return 1.5 * np.pi * surface_energy * radius


def load_config(file_path: str) -> dict:
    try:
        with open(file_path, "r") as file:
            logger.info(f"Loading configuration from {file_path}")
            return toml.load(file)
    except FileNotFoundError:
        logger.error(f"Error: Configuration file '{file_path}' not found.")
        sys.exit(1)
    except toml.TomlDecodeError as e:
        logger.error(f"Error: Failed to parse '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)


def log_norm(x: float, mean: float, stdv: float) -> float:
    """Log normal PDF. Geometric parameters are used."""
    proba_density = (1 / np.sqrt(2 * np.pi)) * (1 / (x * np.log(stdv))) * np.exp(
        -0.5 * (np.log(x / mean) / np.log(stdv)) ** 2)

    return proba_density


def normal(x: float, mean: float, stdv: float) -> float:
    """Normal PDF"""
    proba_density = np.exp(-(x - mean) ** 2 / (2 * (stdv ** 2))) / np.sqrt(2 * np.pi * (stdv ** 2))

    return proba_density
