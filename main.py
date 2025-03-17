import sys

from rnr.config import setup_logging
from rnr.core import run, fraction_velocity_curve

logger = setup_logging(__name__, 'logs/log.log')

def main():
    # run(sys.argv[1])
    fraction_velocity_curve(sys.argv[1])

if __name__ == "__main__":
    main()