import importlib.resources as ires
import logging.config
import logging.handlers
from pathlib import Path

import yaml

from autointent.custom_types import LogLevel


def setup_logging(level: LogLevel | str, log_filename: Path | str | None = None) -> None:
    """Set stdout and file handlers for logging autointent internal actions.

    The first parameter affects the logs to the standard output stream. The second parameter is optional.
    If it is specified, then the "DEBUG" messages are logged to the file,
    regardless of what is specified by the first parameter.

    Args:
        level: one of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        log_filename: specify location of logfile, omit extension as suffix ``.log.jsonl`` will be appended.
    """
    config_file = ires.files("autointent._logging").joinpath("config.yaml")
    with config_file.open(encoding="utf-8") as f_in:
        config = yaml.safe_load(f_in)

    level = LogLevel(level)
    config["handlers"]["stdout"]["level"] = level.value

    if log_filename is not None:
        config["loggers"]["root"]["handlers"].append("file")

        filename = str(log_filename) + ".log.jsonl"
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": filename,
        }
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)
