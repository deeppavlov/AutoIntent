import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from autointent import Context
from autointent.context.data_handler import Tag
from autointent.custom_types import BaseMetadataDict, LabelType

from .base import PredictionModule
from .utils import InvalidNumClassesError, WrongClassificationError


class ArgmaxPredictorDumpMetadata(BaseMetadataDict):
    n_classes: int


class ArgmaxPredictor(PredictionModule):
    name = "argmax"
    n_classes: int

    def __init__(self) -> None:
        pass

    @classmethod
    def from_context(cls, context: Context) -> Self:
        return cls()

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: list[LabelType],
        tags: list[Tag] | None = None,
    ) -> None:
        multilabel = isinstance(labels[0], list)
        if multilabel:
            msg = "ArgmaxPredictor is compatible with single-label classifiction only"
            raise WrongClassificationError(msg)
        self.n_classes = (
            len(labels[0]) if multilabel and isinstance(labels[0], list) else len(set(labels).difference([-1]))
        )

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if scores.shape[1] != self.n_classes:
            msg = "Provided scores number don't match with number of classes which predictor was trained on."
            raise InvalidNumClassesError(msg)
        return np.argmax(scores, axis=1)  # type: ignore[no-any-return]

    def dump(self, path: str) -> None:
        self.metadata = ArgmaxPredictorDumpMetadata(n_classes=self.n_classes)

        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            metadata: ArgmaxPredictorDumpMetadata = json.load(file)

        self.n_classes = metadata["n_classes"]
        self.metadata = metadata
