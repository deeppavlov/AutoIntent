"""Threshold."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from autointent import Context
from autointent.context.data_handler import Tag
from autointent.custom_types import BaseMetadataDict, LabelType

from ._base import PredictionModule
from ._utils import InvalidNumClassesError, apply_tags

logger = logging.getLogger(__name__)


class ThresholdPredictorDumpMetadata(BaseMetadataDict):
    """Threshold predictor metadata."""

    multilabel: bool
    tags: list[Tag] | None
    thresh: float | npt.NDArray[Any]
    n_classes: int


class ThresholdPredictor(PredictionModule):
    """
    Threshold predictor module.

    ThresholdPredictor uses a predefined threshold (or array of thresholds) to predict
    labels for single-label or multi-label classification tasks.

    :ivar metadata_dict_name: Filename for saving metadata to disk.
    :ivar multilabel: If True, the model supports multi-label classification.
    :ivar n_classes: Number of classes in the dataset.
    :ivar tags: Tags for predictions (if any).
    :ivar name: Name of the predictor, defaults to "adaptive".

    Examples
    --------
    Single-label classification example:
    >>> from autointent.modules import ThresholdPredictor
    >>> import numpy as np
    >>> scores = np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
    >>> labels = [1, 0, 1]
    >>> threshold = 0.5
    >>> predictor = ThresholdPredictor(thresh=threshold)
    >>> predictor.fit(scores, labels)
    >>> test_scores = np.array([[0.3, 0.7], [0.5, 0.5]])
    >>> predictions = predictor.predict(test_scores)
    >>> print(predictions)
    [1 0]

    Multi-label classification example:
    >>> labels = [[1, 0], [0, 1], [1, 1]]
    >>> predictor = ThresholdPredictor(thresh=[0.5, 0.5])
    >>> predictor.fit(scores, labels)
    >>> test_scores = np.array([[0.3, 0.7], [0.6, 0.4]])
    >>> predictions = predictor.predict(test_scores)
    >>> print(predictions)
    [[0 1] [1 0]]

    Save and load the model:
    >>> predictor.dump("outputs/")
    >>> loaded_predictor = ThresholdPredictor(thresh=0.5)
    >>> loaded_predictor.load("outputs/")
    >>> print(loaded_predictor.thresh)
    0.5
    """

    metadata: ThresholdPredictorDumpMetadata
    multilabel: bool
    n_classes: int
    tags: list[Tag] | None
    name = "threshold"

    def __init__(
        self,
        thresh: float | npt.NDArray[Any],
    ) -> None:
        """
        Initialize threshold predictor.

        :param thresh: Threshold for the scores, shape (n_classes,) or float
        """
        self.thresh = thresh

    @classmethod
    def from_context(cls, context: Context, thresh: float | npt.NDArray[Any] = 0.5) -> Self:
        """
        Initialize from context.

        :param context: Context
        :param thresh: Threshold
        """
        return cls(
            thresh=thresh,
        )

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: list[LabelType],
        tags: list[Tag] | None = None,
    ) -> None:
        """
        Fit the model.

        :param scores: Scores to fit
        :param labels: Labels to fit
        :param tags: Tags to fit
        """
        self.tags = tags
        self.multilabel = isinstance(labels[0], list)
        self.n_classes = (
            len(labels[0]) if self.multilabel and isinstance(labels[0], list) else len(set(labels).difference([-1]))
        )

        if not isinstance(self.thresh, float):
            if len(self.thresh) != self.n_classes:
                msg = (
                    f"Number of thresholds provided doesn't match with number of classes."
                    f" {len(self.thresh)} != {self.n_classes}"
                )
                logger.error(msg)
                raise InvalidNumClassesError(msg)
            self.thresh = np.array(self.thresh)

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Predict the best score.

        :param scores: Scores to predict
        """
        if self.multilabel:
            return multilabel_predict(scores, self.thresh, self.tags)
        if scores.shape[1] != self.n_classes:
            msg = "Provided scores number don't match with number of classes which predictor was trained on."
            raise InvalidNumClassesError(msg)
        return multiclass_predict(scores, self.thresh)

    def dump(self, path: str) -> None:
        """
        Dump the metadata.

        :param path: Path to dump
        """
        self.metadata = ThresholdPredictorDumpMetadata(
            multilabel=self.multilabel,
            tags=self.tags,
            thresh=self.thresh if isinstance(self.thresh, float) else self.thresh.tolist(),
            n_classes=self.n_classes,
        )

        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

    def load(self, path: str) -> None:
        """
        Load the metadata.

        :param path: Path to load
        """
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            metadata: ThresholdPredictorDumpMetadata = json.load(file)

        self.multilabel = metadata["multilabel"]
        self.tags = [Tag(**tag) for tag in metadata["tags"] if metadata["tags"] and isinstance(metadata["tags"], list)]  # type: ignore[arg-type, union-attr]
        self.thresh = metadata["thresh"]
        self.n_classes = metadata["n_classes"]
        self.metadata = metadata


def multiclass_predict(scores: npt.NDArray[Any], thresh: float | npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Make predictions for multiclass classification task.

    :param scores: Scores from the model, shape (n_samples, n_classes)
    :param thresh: Threshold for the scores, shape (n_classes,) or float
    :return: Predicted classes, shape (n_samples,)
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
    scores: npt.NDArray[Any],
    thresh: float | npt.NDArray[Any],
    tags: list[Tag] | None,
) -> npt.NDArray[Any]:
    """
    Make predictions for multilabel classification task.

    :param scores: Scores from the model, shape (n_samples, n_classes)
    :param thresh: Threshold for the scores, shape (n_classes,) or float
    :param tags: Tags for predictions
    :return: Multilabel prediction
    """
    res = (scores >= thresh).astype(int) if isinstance(thresh, float) else (scores >= thresh[None, :]).astype(int)
    if tags:
        res = apply_tags(res, scores, tags)
    return res
