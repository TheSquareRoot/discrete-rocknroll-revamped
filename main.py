from rnr.simulation.scripts import (
    fraction_velocity_curve,
    multiple_fraction_velocity_curves,
    multiple_runs,
    single_run,
)
from rnr.utils.config import setup_logging, setup_parsing


def main() -> None:
    # Create parser
    parser = setup_parsing()

    # Parse CLI arghuments
    args = parser.parse_args()

    # Set up logging configuration
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(testing=False, log_level=log_level)

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
