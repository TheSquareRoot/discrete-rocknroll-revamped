from .config import setup_logging


# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')

def check_config(config):
    # Check there are enough particles per bin (below 1000 values get wonky)
    parts_per_bin = config["distribution"]["nparts"]/config["distribution"]["nparts"]
    if parts_per_bin < 1000:
        logger.warning(f"Too few particles per bin: {parts_per_bin}")

    # If biasi params are selected, user should not define custom median and scatter values
    if config["distribution"]["params"] == "biasi":
        if (config["distribution"]["adh_median"] is not None) or (config["distribution"]["adh_scatter"] is not None):
            logger.warning("Biasi parameters selected. Custom median and scatter parameters will be ignored.")

        if config["physics"]["adhesion_model"] == 'Rabinovich':
            logger.error("Biasi parameters should only be used with the JKR model.")

