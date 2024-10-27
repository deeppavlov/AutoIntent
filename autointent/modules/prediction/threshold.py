import json
import logging
from pathlib import Path
from typing import Any, TypedDict
from omegaconf.listconfig import ListConfig
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from autointent import Context
from autointent.context.data_handler.tags import Tag
from autointent.custom_types import LABEL_TYPE

from .base import PredictionModule, apply_tags
from ..base import BaseMetadataDict

logger = logging.getLogger(__name__)


class ThresholdPredictorDumpMetadata(BaseMetadataDict):
    multilabel: bool
    # tags: list[Tag] | None
    thresh: float | npt.NDArray[Any]
    multilabel: bool
    n_classes: int | None


class ThresholdPredictor(PredictionModule):
    metadata_dict_name: str = "metadata.json"
    multilabel: bool
    tags: list[Tag] | None

    def __init__(self, thresh: float | npt.NDArray[Any], multilabel: bool = False, n_classes: int | None = None, tags: list[Tag] | None = None) -> None:
        self.thresh = thresh
        self.multilabel = multilabel
        self.n_classes = n_classes
        # self.tags = tags

    @classmethod
    def from_context(cls, context: Context, thresh: float | npt.NDArray[Any] = 0.5, **kwargs: dict[str, Any]) -> Self:
        return cls(
            thresh=thresh,
            multilabel=context.multilabel,
            n_classes=context.n_classes,
        )

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: list[LABEL_TYPE],
        tags: list[Tag] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        self.tags = tags


        if not isinstance(self.thresh, float):
            if len(self.thresh) != self.n_classes:
                msg = "Wrong number of thresholds provided doesn't match with number of classes"
                logger.error(msg)
                raise ValueError(msg)
            self.thresh = np.array(self.thresh)

        self.metadata = ThresholdPredictorDumpMetadata(
            multilabel=self.multilabel,
            # tags=self.tags,
            thresh=self.thresh if isinstance(self.thresh, float) else self.thresh.tolist(),
            n_classes=self.n_classes)

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.multilabel:
            return multilabel_predict(scores, self.thresh, self.tags)
        return multiclass_predict(scores, self.thresh)

    def dump(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            metadata: ThresholdPredictorDumpMetadata = json.load(file)

        self.multilabel = metadata["multilabel"]
        # self.tags = metadata["tags"]
        self.thresh = metadata["thresh"]
        self.n_classes = metadata["n_classes"]
        self.metadata = metadata


def multiclass_predict(scores: npt.NDArray[Any], thresh: float | npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Return
    ---
    array of int labels, shape (n_samples,)
    """
    pred_classes: npt.NDArray[Any] = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(len(scores)), pred_classes]

    if isinstance(thresh, float):
        pred_classes[best_scores < thresh] = -1  # out of scope
    else:
        thresh_selected = thresh[pred_classes]
        pred_classes[best_scores < thresh_selected] = -1  # out of scope

    return pred_classes


def multilabel_predict(
    scores: npt.NDArray[Any], thresh: float | npt.NDArray[Any], tags: list[Tag] | None
) -> npt.NDArray[Any]:
    """
    Return
    ---
    array of binary labels, shape (n_samples, n_classes)
    """
    res = (scores >= thresh).astype(int) if isinstance(thresh, float) else (scores >= thresh[None, :]).astype(int)
    if tags:
        res = apply_tags(res, scores, tags)
    return res
