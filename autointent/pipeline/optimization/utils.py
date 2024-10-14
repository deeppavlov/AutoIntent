import importlib.resources as ires
import json
import logging
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any

import yaml

from autointent.pipeline.utils import generate_name


def load_data(data_path: str, multilabel: bool) -> list[dict[str, Any]]:
    """load data from the given path or load sample data which is distributed along with the autointent package"""
    if data_path == "default":
        data_name = "dstc3-20shot.json" if multilabel else "banking77.json"
        with ires.files("autointent.datafiles").joinpath(data_name).open() as file:
            return json.load(file)
    elif data_path != "":
        with Path(data_path).open() as file:
            return json.load(file)
    return []


def get_run_name(run_name: str) -> str:
    if run_name == "":
        run_name = generate_name()
    return f"{run_name}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}"  # noqa: DTZ005


def setup_logging(level: str | None = None) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="{asctime} - {name} - {levelname} - {message}",
        style="{",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_config(config_path: str, multilabel: bool, logger: Logger | None = None) -> dict[str, Any]:
    """load config from the given path or load default config which is distributed along with the autointent package"""
    if config_path != "":
        if logger is not None:
            logger.debug("loading optimization search space config from %s...)", config_path)
        with Path(config_path).open() as file:
            file_content = file.read()
    else:
        if logger is not None:
            logger.debug("loading default optimization search space config...")
        config_name = "default-multilabel-config.yaml" if multilabel else "default-multiclass-config.yaml"
        with ires.files("autointent.datafiles").joinpath(config_name).open() as file:
            file_content = file.read()
    return yaml.safe_load(file_content)
