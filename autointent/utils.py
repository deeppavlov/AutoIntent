"""AutoIntent utilities."""

import importlib.resources as ires
from pathlib import Path
from typing import Any

import yaml

from autointent.custom_types import SearchSpacePresets


def load_search_space(path: Path | str) -> list[dict[str, Any]]:
    """
    Load hyperparameters search space from file.

    :param path: path to yaml file
    :return:
    """
    with Path(path).open() as file:
        return yaml.safe_load(file)  # type: ignore[no-any-return]


def load_preset(name: SearchSpacePresets) -> dict[str, Any]:
    """
    Load one of preset search spaces.

    :param name: name of a presets.
    """
    path = ires.files("autointent._presets").joinpath(name + ".yaml")
    with path.open() as file:
        return yaml.safe_load(file)  # type: ignore[no-any-return]
