import json
from typing import Any

import numpy as np
from omegaconf import ListConfig


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
