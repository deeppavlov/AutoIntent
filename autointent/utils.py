"""AutoIntent utilities."""

import importlib.resources as ires
from pathlib import Path
from typing import Any

import yaml

from autointent.custom_types import SearchSpacePreset


def load_search_space(path: Path | str) -> list[dict[str, Any]]:
    """Load hyperparameters search space from file.

    Args:
        path: Path to the search space file.

    Returns:
        List of dictionaries representing the search space.
    """
    with Path(path).open() as file:
        return yaml.safe_load(file)  # type: ignore[no-any-return]


def load_preset(name: SearchSpacePreset) -> dict[str, Any]:
    """Load one of preset search spaces.

    Args:
        name: Name of the preset search space.

    Returns:
        Dictionary representing the preset search space.
    """
    path = ires.files("autointent._presets").joinpath(name + ".yaml")
    with path.open() as file:
        return yaml.safe_load(file)  # type: ignore[no-any-return]
