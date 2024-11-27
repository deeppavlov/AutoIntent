"""AutoIntent utilities."""

import importlib.resources as ires
from typing import Any

import yaml


def load_default_search_space(multilabel: bool) -> dict[str, Any]:
    """
    Load configuration from the given path or load default configuration.

    :param multilabel: Whether to use multilabel or not
    :return:
    """
    config_name = "default-multilabel-config.yaml" if multilabel else "default-multiclass-config.yaml"
    with ires.files("autointent.datafiles").joinpath(config_name).open() as file:
        file_content = file.read()
    return yaml.safe_load(file_content)  # type: ignore[no-any-return]
