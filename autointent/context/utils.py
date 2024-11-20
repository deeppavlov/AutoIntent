import importlib.resources as ires
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import ListConfig

from .data_handler import Dataset


class NumpyEncoder(json.JSONEncoder):
    """Helper for dumping logs. Problem explained: https://stackoverflow.com/q/50916422"""

    def default(self, obj: Any) -> str | int | float | list[Any] | Any:  # noqa: ANN401
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
    """load data from the given path or load sample data which is distributed along with the autointent package"""
    if filepath == "default-multiclass":
        return Dataset.from_json(
            ires.files("autointent.datafiles").joinpath("banking77.json"), # type: ignore[arg-type]
        )
    if filepath == "default-multilabel":
        return Dataset.from_json(
            ires.files("autointent.datafiles").joinpath("dstc3-20shot.json"), # type: ignore[arg-type]
        )
    return Dataset.from_json(filepath)
