from .config import setup_logging


# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')

def check_config(config):
    """
    Check the conformity of the parameters provided by the user.
    """
    # Creation of derived parameters
    # i.e. params indirectly defined by the user, and that are practical to store as parameters
    config["sizedistrib"]["nmodes"] = len(config["sizedistrib"]["modes"])

    # Check there are enough particles per bin (below 1000 values get wonky)
    parts_per_bin = config["adhdistrib"]["nparts"]/config["adhdistrib"]["nbins"]
    if parts_per_bin < 1000:
        logger.warning(f"Too few particles per bin: {parts_per_bin}")

    # If biasi params are selected, user should not define custom median and scatter values
    if config["adhdistrib"]["params"] == "biasi":
        if (config["adhdistrib"]["medians"] is not None) or (config["distribution"]["spreads"] is not None):
            logger.warning("Biasi parameters selected. Custom median and scatter parameters will be ignored.")

        if config["physics"]["adhesion_model"] == 'Rabinovich':
            logger.error("Biasi parameters should only be used with the JKR model.")