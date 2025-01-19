"""AutoIntent utilities."""

import importlib.resources as ires
from pathlib import Path
from typing import Any

import yaml


def load_default_search_space(multilabel: bool) -> list[dict[str, Any]]:
    """
    Load configuration from the given path or load default configuration.

    :param multilabel: Whether to use multilabel or not
    :return:
    """
    config_name = "default-multilabel-config.yaml" if multilabel else "default-multiclass-config.yaml"
    path = ires.files("autointent._datafiles").joinpath(config_name)
    return load_search_space(path)  # type: ignore[arg-type]


def load_search_space(path: Path | str) -> list[dict[str, Any]]:
    """
    Load hyperparameters search space from file.

    :param path: path to yaml file
    :return:
    """
    with Path(path).open() as file:
        return yaml.safe_load(file)  # type: ignore[no-any-return]
