import sys

from rnr.check_config import check_config
from rnr.config import setup_logging
from rnr.distribution import SizeDistributionBuilder, AdhesionDistributionBuilder
from rnr.flow import FlowBuilder
from rnr.utils import load_config

logger = setup_logging(__name__, 'logs/log.log')

def main():
    # Load config file
    config = load_config(f"configs/{sys.argv[1]}.toml")

    # Check the values from the config file
    logger.info('Checking parameters...')
    check_config(config)

    # Build the particle size distribution
    logger.info('Generating size distribution...')
    sizedistrib_builder = SizeDistributionBuilder(**config)
    size_distrib = sizedistrib_builder.generate()
    logger.debug(f'Size distribution generated: {size_distrib}')

    size_distrib.plot(scale='linear')

    # Build the adhesion force distribution
    logger.info('Generating adhesion distribution...')
    adhesion_builder = AdhesionDistributionBuilder(size_distrib, **config)
    adh_distrib = adhesion_builder.generate()
    logger.debug(f'Adhesion distribution generated: {adh_distrib}')

    adh_distrib.plot(0, norm=False, scale='linear')

    # Build the flow
    logger.info('Generating friction velocity time history...')
    flow_builder = FlowBuilder(**config)
    flow = flow_builder.generate()

    flow.plot(scale='linear')

if __name__ == "__main__":
    main()