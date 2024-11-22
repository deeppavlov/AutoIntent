"""Cli utilities for optimization."""

import importlib.resources as ires
from logging import Logger
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path | None, multilabel: bool, logger: Logger | None = None) -> dict[str, Any]:
    """
    Load configuration from the given path or load default configuration.

    :param config_path: Path to the configuration file
    :param multilabel: Whether to use multilabel or not
    :param logger: Logger
    :return:
    """
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
