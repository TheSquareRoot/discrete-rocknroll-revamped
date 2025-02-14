import sys

from rnr.check_config import check_config
from rnr.distribution import DistributionBuilder
from rnr.utils import load_config


def main():
    # Load config file
    config = load_config(f"configs/{sys.argv[1]}.toml")

    # Check the values from the config file
    check_config(config)



if __name__ == "__main__":
    main()