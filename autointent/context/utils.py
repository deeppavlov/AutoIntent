"""Module for loading datasets and handling JSON serialization with numpy compatibility.

This module provides utilities for loading datasets and serializing objects
that include numpy data types.
"""

import importlib.resources as ires
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import ListConfig

from .data_handler import Dataset


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy data types and OmegaConf ListConfig.

    This encoder extends the default `json.JSONEncoder` to serialize numpy
    arrays, numpy data types, and OmegaConf ListConfig objects.
    """

    def default(self, obj: Any) -> str | int | float | list[Any] | Any:  # noqa: ANN401
        """
        Serialize objects with special handling for numpy and OmegaConf types.

        :param obj: Object to serialize.
        :return: JSON-serializable representation of the object.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, ListConfig):
            return list(obj)
        return super().default(obj)


def load_data(filepath: str | Path) -> Dataset:
    """
    Load data from a specified path or use default sample data.

    This function loads a dataset from a JSON file or retrieves sample data
    included with the `autointent` package for default multiclass or multilabel
    datasets.

    :param data_path: Path to the dataset file, or a predefined key:
                      - "default-multiclass": Loads sample multiclass dataset.
                      - "default-multilabel": Loads sample multilabel dataset.
    :return: A `Dataset` object containing the loaded data.
    """
    if filepath == "default-multiclass":
        return Dataset.from_json(
            ires.files("autointent.datafiles").joinpath("banking77.json"), # type: ignore[arg-type]
        )
    if filepath == "default-multilabel":
        return Dataset.from_json(
            ires.files("autointent.datafiles").joinpath("dstc3-20shot.json"), # type: ignore[arg-type]
        )
    return Dataset.from_json(filepath)
