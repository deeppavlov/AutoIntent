import importlib.resources as ires
import logging.config
import logging.handlers

import yaml

from autointent.custom_types import LogLevel


def setup_logging(level: LogLevel | str) -> None:
    config_file = ires.files("autointent._logging").joinpath("config.yaml")
    with config_file.open() as f_in:
        config = yaml.safe_load(f_in)

    level = LogLevel(level)
    config["loggers"]["root"]["level"] = level.value

    logging.config.dictConfig(config)
