from rnr.utils.config import setup_logging, setup_parsing
from rnr.simulation.scripts import run, fraction_velocity_curve

logger = setup_logging(__name__, 'logs/log.log')

def main():

    # Create parser
    parser = setup_parsing()

    # Parse CLI arghuments
    args = parser.parse_args()

    # Run simulations
    if args.single_run:
        run(args.config)
    elif args.fraction_velocity:
        fraction_velocity_curve(args.config)

if __name__ == "__main__":
    main()