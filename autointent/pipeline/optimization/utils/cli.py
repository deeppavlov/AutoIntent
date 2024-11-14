import importlib.resources as ires
import json
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any

import yaml

from autointent.context.data_handler import Dataset

from .name import generate_name


def load_data(filepath: str | Path) -> Dataset:
    """load data from the given path or load sample data which is distributed along with the autointent package"""
    if filepath == "default-multiclass":
        return Dataset.load(ires.files("autointent.datafiles").joinpath("banking77.json"))
    if filepath == "default-multilabel":
        return Dataset.load(ires.files("autointent.datafiles").joinpath("dstc3-20shot.json"))
    return Dataset.load(filepath)

def get_run_name(run_name: str | None = None) -> str:
    if run_name is None:
        run_name = generate_name()
    return f"{run_name}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}"  # noqa: DTZ005


def get_logs_dir(run_name: str, logs_dir: Path | None = None) -> Path:
    if logs_dir is None:
        logs_dir = Path.cwd()
    res = logs_dir / run_name
    res.mkdir(parents=True)
    return res


def load_config(config_path: str | Path | None, multilabel: bool, logger: Logger | None = None) -> dict[str, Any]:
    """load config from the given path or load default config which is distributed along with the autointent package"""
    if config_path is not None:
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
