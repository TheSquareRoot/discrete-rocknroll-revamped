from .check_config import check_config
from .config import setup_logging
from .distribution import SizeDistributionBuilder, AdhesionDistributionBuilder
from .flow import FlowBuilder
from .utils import load_config


logger = setup_logging(__name__, 'logs/log.log')


def run(config_file: str) -> None:
    # Load config file
    config = load_config(f"configs/{config_file}.toml")

    # Check the values from the config file
    logger.info('Checking parameters...')
    check_config(config)

    # Build the particle size distribution
    logger.info('Generating size distribution...')
    sizedistrib_builder = SizeDistributionBuilder(**config['sizedistrib'])
    size_distrib = sizedistrib_builder.generate()
    logger.debug(f'Size distribution generated: {size_distrib}')

    size_distrib.plot(scale='linear')

    # Build the adhesion force distribution
    logger.info('Generating adhesion distribution...')
    adhesion_builder = AdhesionDistributionBuilder(size_distrib=size_distrib,
                                                   surface_energy=config['physics']['surf_energy'],
                                                   **config['adhdistrib'])
    adh_distrib = adhesion_builder.generate()
    logger.debug(f'Adhesion distribution generated: {adh_distrib}')

    adh_distrib.plot(0, norm=False, scale='linear')

    # Build the flow
    logger.info('Generating friction velocity time history...')
    flow_params = {**config['simulation'], **config['physics']}
    flow_builder = FlowBuilder(size_distrib=size_distrib,
                               **flow_params)
    flow = flow_builder.generate()

    flow.plot(scale='linear')
