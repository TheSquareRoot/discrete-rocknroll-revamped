import sys

from rnr.check_config import check_config
from rnr.config import setup_logging
from rnr.distribution import SizeDistributionBuilder, AdhesionDistributionBuilder
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

    size_distrib.plot(scale='linear')

    print(size_distrib.radii)

    # Build the adhesion force distribution
    logger.info('Generating adhesion distribution...')
    adhesion_builder = AdhesionDistributionBuilder(size_distrib, **config)
    adh_distrib = adhesion_builder.generate()

    adh_distrib.plot(1, scale='linear')


if __name__ == "__main__":
    main()