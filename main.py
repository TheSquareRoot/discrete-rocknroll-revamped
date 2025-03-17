import argparse
import sys

from rnr.config import setup_logging
from rnr.core import run, fraction_velocity_curve

logger = setup_logging(__name__, 'logs/log.log')

def main():
    # Define parser
    parser = argparse.ArgumentParser()

    # Add CLi arguments to the parser
    parser.add_argument('-c','--config',
                        help='name of the configuration file',
                        type=str,)
    parser.add_argument('-r','--single-run',
                        help='run a single simulation from the config file',
                        action='store_true',)
    parser.add_argument('-f','--fraction-velocity',
                        help='plot the fraction-velocity curve',
                        action='store_true',)

    args = parser.parse_args()

    if args.single_run:
        run(args.config)
    elif args.fraction_velocity:
        fraction_velocity_curve(args.config)

if __name__ == "__main__":
    main()