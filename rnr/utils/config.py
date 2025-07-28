import argparse
import logging

from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


def setup_logging(name: str, log_file: str) -> logging.Logger:
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    file_handler = logging.FileHandler(log_file, mode="w")
    console_handler = RichHandler()

    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Formatter
    file_formatter = logging.Formatter(
        "{asctime} - {name} - {levelname} - {message}",
        style="{",
        datefmt="%H:%M",
    )

    console_formatter = logging.Formatter(
        "{levelname} - {message}",
        style="{",
        datefmt="%H:%M",
    )

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    return logger


def setup_parsing() -> argparse.ArgumentParser:
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

    return parser


def setup_progress_bar() -> Progress:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )

    return progress
