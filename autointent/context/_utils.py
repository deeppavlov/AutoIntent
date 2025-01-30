"""Module for loading datasets and handling JSON serialization with numpy compatibility.

This module provides utilities for loading datasets and serializing objects
that include numpy data types.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from autointent import Dataset


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy data types.

    This encoder extends the default `json.JSONEncoder` to serialize numpy
    arrays, numpy data types.
    """

    def default(self, obj: Any) -> str | int | float | list[Any] | Any:  # noqa: ANN401
        """
        Serialize objects with special handling for numpy.

        :param obj: Object to serialize.
        :return: JSON-serializable representation of the object.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_dataset(path: str | Path) -> Dataset:
    """
    Load data from a specified path or use default sample data or load from hugging face hub.

    This function loads a dataset from a JSON file or retrieves sample data
    included with the `autointent` package for default multiclass or multilabel
    datasets.

    :param data_path: Path to the dataset file, or a predefined key:
                      - "default-multiclass": Loads sample multiclass dataset.
                      - "default-multilabel": Loads sample multilabel dataset.
    :return: A `Dataset` object containing the loaded data.
    """
    if path == "default-multiclass":
        return Dataset.from_hub("AutoIntent/clinc150_subset")
    if path == "default-multilabel":
        return Dataset.from_hub("AutoIntent/clinc150_subset").to_multilabel()
    if not Path(path).exists():
        return Dataset.from_hub(str(path))
    return Dataset.from_json(path)
