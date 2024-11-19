"""Logger."""

import logging
from pprint import pformat


def get_logger() -> logging.Logger:
    """
    Get a logger with a pretty-printing formatter.

    :return: A logger object.
    """
    logger = logging.getLogger(__name__)

    formatter = PPrintFormatter()
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


class PPrintFormatter(logging.Formatter):
    """A logging formatter that pretty-prints dictionaries."""

    def __init__(self) -> None:
        """Initialize the formatter."""
        super().__init__(fmt="{asctime} - {name} - {levelname} - {message}", style="{")

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record.

        :param record: The log record.
        :return:
        """
        if isinstance(record.msg, dict):
            format_msg = "module scoring results:\n"
            dct_to_str = pformat(record.msg)
            record.msg = format_msg + dct_to_str
        return super().format(record)
