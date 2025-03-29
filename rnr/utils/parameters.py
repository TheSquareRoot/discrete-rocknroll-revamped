import sys
import toml

from .config import setup_logging
from ..postproc.plotting import plot_validity_domain
from ..utils.misc import rplus


# Configure module logger from utils file
logger = setup_logging(__name__, 'logs/log.log')


def load_config(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            logger.info(f'Loading configuration from {file_path}')
            return toml.load(file)
    except FileNotFoundError:
        logger.error(f'Error: Configuration file "{file_path}" not found.')
        sys.exit(1)
    except toml.TomlDecodeError as e:
        logger.error(f'Error: Failed to parse "{file_path}": {e}', file=sys.stderr)
        sys.exit(1)

def check_model_validity(config):
    """
    Checks that each mode is within the allowed r+ range for the given flow conditions.
    The main thing to check is that r+ is less than 2.5, i.e. the particle is fully submerged in the
    viscous sublayer.
    """
    for i, mode in enumerate(config['sizedistrib']['modes']):
        rp = rplus(mode*1e-6, config['simulation']['target_vel'], config['physics']['viscosity'])
        logger.info(f'Mode {i+1}: r+={rp:.2f}')
        if rp > 2.5:
            logger.warning(f'Mode {i} is outside the viscous sublayer!')

def check_config(config):
    """
    Check the conformity of the parameters provided by the user.
    """
    # Creation of derived parameters
    # i.e. params indirectly defined by the user, and that are practical to store as parameters
    config['sizedistrib']['nmodes'] = len(config['sizedistrib']['modes'])

    # ADHESION PARAMETERS
    # If biasi params are selected, user should not define custom median and scatter values
    if config['adhdistrib']['dist_params'] == 'biasi':
        if (config['adhdistrib']['medians'] is not None) or (config['distribution']['spreads'] is not None):
            logger.warning('Biasi parameters selected. Custom median and scatter parameters will be ignored.')

        if config['physics']['adhesion_model'] == 'Rabinovich':
            logger.error('Biasi parameters should only be used with the JKR model.')

    # SIMULATION PARAMETERS
    if config['simulation']['duration'] < config['simulation']['dt']:
        logger.error(f'dt must be smaller than total duration. dt set to {config["simulation"]["duration"]:.2f}s')
        config['simulation']['dt'] = config['simulation']['duration']

    if config['simulation']['duration'] < config['simulation']['acc_time']:
        logger.warning('Spin-up time is longer than simulation time.')

    # Check whether the hypothesis of the RnR model are respected
    check_model_validity(config)
    plot_validity_domain(config['sizedistrib']['modes'], config['simulation']['target_vel'], config['physics']['viscosity'])
