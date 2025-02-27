"""Module for loading datasets and handling JSON serialization with numpy compatibility."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from autointent import Dataset


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy data types.

    This encoder extends the default `json.JSONEncoder` to serialize numpy
    arrays and numpy data types.

    Attributes:
        Inherits all attributes from json.JSONEncoder.
    """

    def default(self, obj: Any) -> str | int | float | list[Any] | Any:  # noqa: ANN401
        """Serialize objects with special handling for numpy.

        Args:
            obj: Object to serialize.

        Returns:
            JSON-serializable representation of the object.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_dataset(path: str | Path) -> Dataset:
    """Load data from a specified path or use default sample data.

    This function loads a dataset from a JSON file or retrieves sample data
    included with the `autointent` package for default multiclass or multilabel
    datasets. If the path doesn't exist, it attempts to load from the Hugging Face hub.

    Args:
        path: Path to the dataset file, or a predefined key:
            - "default-multiclass": Loads sample multiclass dataset.
            - "default-multilabel": Loads sample multilabel dataset.

    Returns:
        A Dataset object containing the loaded data.
    """
    if path == "default-multiclass":
        return Dataset.from_hub("AutoIntent/clinc150_subset")
    if path == "default-multilabel":
        return Dataset.from_hub("AutoIntent/clinc150_subset").to_multilabel()
    if not Path(path).exists():
        return Dataset.from_hub(str(path))
    return Dataset.from_json(path)
