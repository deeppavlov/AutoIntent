import importlib.resources as ires
import json
import logging.config
import logging.handlers

from autointent.custom_types import LogLevel


def setup_logging(level: LogLevel) -> None:
    config_file = ires.files("autointent._logging").joinpath("config.json")
    with config_file.open() as f_in:
        config = json.load(f_in)

    logging.config.dictConfig(config)
    logging.basicConfig(level=level.value)
