import logging
from logging import Formatter
from typing import Literal

LoggingLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(level: LoggingLevelType = None):
    logging.basicConfig(
        level=level,
        format="{asctime} - {name} - {levelname} - {message}",
        style="{",
        handlers=[logging.StreamHandler()],
    )


def get_logger(name: str, level: LoggingLevelType = None, formatter: Formatter = None):
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    if formatter is not None:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
