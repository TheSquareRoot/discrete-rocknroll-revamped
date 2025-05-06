from rnr.utils.config import setup_logging, setup_parsing
from rnr.simulation.scripts import (single_run,
                                    multiple_runs,
                                    fraction_velocity_curve,
                                    multiple_fraction_velocity_curves
                                    )

logger = setup_logging(__name__, 'logs/log.log')

def main():

    # Create parser
    parser = setup_parsing()

    # Parse CLI arghuments
    args = parser.parse_args()

    # Run simulations
    if args.single_run:
        if args.multiple_runs:
            multiple_runs(args.config_dir)
        else:
            single_run(args.config_file)
    elif args.fraction_velocity:
        if args.multiple_runs:
            multiple_fraction_velocity_curves(args.config_dir)
        else:
            fraction_velocity_curve(args.config_file)

if __name__ == "__main__":
    main()