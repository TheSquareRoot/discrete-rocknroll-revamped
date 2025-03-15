import sys
import toml

from .config import setup_logging


# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')


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

def check_config(config):
    """
    Check the conformity of the parameters provided by the user.
    """
    # Creation of derived parameters
    # i.e. params indirectly defined by the user, and that are practical to store as parameters
    config["sizedistrib"]["nmodes"] = len(config["sizedistrib"]["modes"])

    # If biasi params are selected, user should not define custom median and scatter values
    if config["adhdistrib"]["dist_params"] == "biasi":
        if (config["adhdistrib"]["medians"] is not None) or (config["distribution"]["spreads"] is not None):
            logger.warning("Biasi parameters selected. Custom median and scatter parameters will be ignored.")

        if config["physics"]["adhesion_model"] == 'Rabinovich':
            logger.error("Biasi parameters should only be used with the JKR model.")
