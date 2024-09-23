import logging
from logging import Logger, Formatter
from typing import Literal

LoggingLevelType = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']


def setup_logging(level: LoggingLevelType, name: str, formatter: Formatter = None) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    if formatter is None:
        formatter = logging.Formatter('{asctime} - {name} - {levelname} - {message}', style="{")
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger
