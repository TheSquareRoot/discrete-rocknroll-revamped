import argparse
import logging
from logging import StreamHandler


def setup_logging(name, log_file):
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    file_handler = logging.FileHandler(log_file, mode='w')
    console_handler = StreamHandler()

    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Formatter
    file_formatter = logging.Formatter("{asctime} - {name} - {levelname} - {message}",
                                       style="{",
                                       datefmt="%H:%M",
                                       )

    console_formatter = logging.Formatter("{levelname} - {message}",
                                          style="{",
                                          datefmt="%H:%M",
                                          )

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    return logger

def setup_parsing():
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

    return parser