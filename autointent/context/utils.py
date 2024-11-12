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


def load_data(data_path: str | Path) -> Dataset:
    """load data from the given path or load sample data which is distributed along with the autointent package"""
    if data_path == "default-multiclass":
        with ires.files("autointent.datafiles").joinpath("banking77.json").open() as file:
            res = json.load(file)
    elif data_path == "default-multilabel":
        with ires.files("autointent.datafiles").joinpath("dstc3-20shot.json").open() as file:
            res = json.load(file)
    else:
        with Path(data_path).open() as file:
            res = json.load(file)

    return Dataset.model_validate(res)
