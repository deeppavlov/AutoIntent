import importlib.resources as ires
import json
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any

import yaml

from autointent.context.data_handler import Dataset

from .name import generate_name


def load_data(data_path: str) -> Dataset | None:
    """load data from the given path or load sample data which is distributed along with the autointent package"""
    if data_path == "default-multiclass":
        with ires.files("autointent.datafiles").joinpath("banking77.json").open() as file:
            res = json.load(file)
    elif data_path == "default-multilabel":
        with ires.files("autointent.datafiles").joinpath("dstc3-20shot.json").open() as file:
            res = json.load(file)
    elif data_path != "":
        with Path(data_path).open() as file:
            res = json.load(file)
    else:
        return None
    return Dataset.model_validate(res)


def get_run_name(run_name: str) -> str:
    if run_name == "":
        run_name = generate_name()
    return f"{run_name}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}"  # noqa: DTZ005


def get_logs_dir(logs_dir: str, run_name: str) -> Path:
    logs_dir_ = Path.cwd() if logs_dir == "" else Path(logs_dir)
    res = logs_dir_ / run_name
    res.mkdir(parents=True)
    return res


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
    return yaml.safe_load(file_content)  # type: ignore[no-any-return]
