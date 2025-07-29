import sys
from pathlib import Path

import toml

from rnr.postproc.plotting import plot_validity_domain
from rnr.utils.misc import rplus

from .config import setup_logging

# Configure module logger from utils file
logger = setup_logging(__name__, "logs/log.log")


def load_config(config_file: Path) -> dict:
    try:
        with config_file.open("r") as file:
            logger.info("Loading configuration file...")
            return toml.load(file)
    except FileNotFoundError as e:
        logger.exception("Configuration file not found!")
        raise FileNotFoundError from e
    except toml.TomlDecodeError as e:
        logger.exception("Failed to parse configuration file!")
        raise ValueError from e


def check_model_validity(config: dict) -> None:
    """
    Checks that each mode is within the allowed r+ range for the given flow conditions.
    The main thing to check is that r+ is less than 2.5, i.e. the particle is fully submerged in the
    viscous sublayer.
    """
    for i, mode in enumerate(config["sizedistrib"]["modes"]):
        rp = rplus(
            mode * 1e-6,
            config["simulation"]["target_vel"],
            config["physics"]["viscosity"],
        )
        logger.info(f"Mode {i + 1}: r+={rp:.2f}")
        if rp > 2.5:
            logger.warning(f"Mode {i} is outside the viscous sublayer!")


def check_config(config: dict) -> None:
    """
    Check the conformity of the parameters provided by the user.
    """
    # Creation of derived parameters
    # i.e. params indirectly defined by the user, and that are practical to store as parameters
    config["sizedistrib"]["nmodes"] = len(config["sizedistrib"]["modes"])

    # ADHESION PARAMETERS
    # Biasi parametrization can only be used with a lognormal distribution. If custom params are given, they will be ignored.
    if config["adhdistrib"]["biasi"]:
        if "distnames" in config["adhdistrib"]:
            logger.warning("Biasi parametrization uses a lognormal distribution, custom distributions will be ignored.")

        if "distshapes" in config["adhdistrib"]:
            logger.warning("Biasi parametrization used, custom parameters will be ignored.")

        if config["physics"]["adhesion_model"] == "Rabinovich":
            logger.error("Biasi parameters should only be used with the JKR model.")
            raise ValueError

    else:
        if any(x != 0 for x in config["sizedistrib"]["spreads"]):
            logger.error("Custom parameters can only be used with discrete distributions!")
            raise ValueError

        if not (
            len(config["adhdistrib"]["loc"])
            == len(config["adhdistrib"]["scale"])
            == len(config["adhdistrib"]["distshapes"])
        ):
            logger.error("disthapes, loc and scale should have the same length!")
            raise ValueError

        if config["sizedistrib"]["nbins"] > 1:
            logger.error("For now multimodal distributions are only available with a single particle size bin!")
            raise ValueError

    # SIMULATION PARAMETERS
    if config["simulation"]["duration"] < config["simulation"]["dt"]:
        logger.error(f"dt must be smaller than total duration. dt set to {config['simulation']['duration']:.2f}s")
        config["simulation"]["dt"] = config["simulation"]["duration"]

    if config["simulation"]["duration"] < config["simulation"]["acc_time"]:
        logger.warning("Spin-up time is longer than simulation time.")

    # Check whether the hypothesis of the RnR model are respected
    check_model_validity(config)
    plot_validity_domain(
        config["sizedistrib"]["modes"],
        config["simulation"]["target_vel"],
        config["physics"]["viscosity"],
    )
