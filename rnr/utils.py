import toml
import sys

from .config import setup_logging

# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')

def load_config(file_path):
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
