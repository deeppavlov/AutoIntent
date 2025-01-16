import importlib.resources as ires
import logging.config
import logging.handlers
from pathlib import Path

import yaml

from autointent.custom_types import LogLevel


def setup_logging(level: LogLevel | str, log_to_filepath: Path | str | None = None) -> None:
    config_file = ires.files("autointent._logging").joinpath("config.yaml")
    with config_file.open() as f_in:
        config = yaml.safe_load(f_in)

    level = LogLevel(level)
    config["handlers"]["stdout"]["level"] = level.value

    if log_to_filepath is not None:
        config["loggers"]["root"]["handlers"].append("file")
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": str(log_to_filepath),
        }


    logging.config.dictConfig(config)
