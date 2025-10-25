import argparse
import logging
import logging.config

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


def setup_logging(*, testing: bool = False, log_level: str = "INFO") -> None:
    """
    Set up application-wide logging configuration.

    Parameters
    ----------
    testing : bool, optional
        Toggles simplified logging configuration for test environments.
    log_level : str
        Logging level for the console output.
    """
    if testing:  # No console output while testing
        config = {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "file": {"format": "%(asctime)s [%(levelname)s] %(name)s: [TEST] - %(message)s"},
            },
            "handlers": {
                "file": {
                    "level": "DEBUG",
                    "formatter": "file",
                    "class": "logging.FileHandler",
                    "filename": "logs/log.log",
                },
            },
            "loggers": {
                "": {  # root logger
                    "handlers": ["file"],
                    "level": "WARNING",
                    "propagate": False,
                },
                "rnr": {
                    "handlers": ["file"],
                    "level": "DEBUG",
                    "propagate": False,
                },
                "__main__": {  # if __name__ == '__main__'
                    "handlers": ["file"],
                    "level": "DEBUG",
                    "propagate": False,
                },
            },
        }
    else:
        config = {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "console": {"format": "[%(levelname)s]: %(message)s"},
                "file": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
            },
            "handlers": {
                "console": {
                    "()": "rich.logging.RichHandler",
                    "level": log_level,
                    "formatter": "console",
                    "show_time": False,
                    "markup": False,
                    "rich_tracebacks": True,
                },
                "file": {
                    "level": "DEBUG",
                    "formatter": "file",
                    "class": "logging.FileHandler",
                    "filename": "logs/log.log",
                },
            },
            "loggers": {
                "": {  # root logger
                    "handlers": ["console"],
                    "level": "WARNING",
                    "propagate": False,
                },
                "rnr": {
                    "handlers": ["console", "file"],
                    "level": "DEBUG",
                    "propagate": False,
                },
                "__main__": {  # if __name__ == '__main__'
                    "handlers": ["console", "file"],
                    "level": "DEBUG",
                    "propagate": False,
                },
            },
        }

    logging.config.dictConfig(config)


def setup_parsing() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""

    # Define parser
    parser = argparse.ArgumentParser()

    # Add CLi arguments to the parser
    parser.add_argument(
        "-c",
        "--config-file",
        help="name of the configuration file",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--config-dir",
        help="name of the configuration directory",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--single-run",
        help="run a single simulation from the utils file",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--fraction-velocity",
        help="plot the fraction-velocity curve",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--multiple-runs",
        help="run multiple simulations from different config files.",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging output.",
    )

    return parser


def setup_progress_bar() -> Progress:
    """Create and return a Rich progress bar instance with default columns."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )

    return progress
