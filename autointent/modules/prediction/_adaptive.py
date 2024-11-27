"""AdaptivePredictor module for multi-label classification with adaptive thresholds."""

import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
from sklearn.metrics import f1_score
from typing_extensions import Self

from autointent import Context
from autointent.context.data_handler import Tag
from autointent.custom_types import LabelType

from ._base import PredictionModule
from ._utils import InvalidNumClassesError, WrongClassificationError, apply_tags

default_search_space = np.linspace(0, 1, num=10)


class AdaptivePredictorDumpMetadata(TypedDict):
    """
    Metadata structure for saving the state of an AdaptivePredictor.

    :ivar r: The selected threshold scaling factor.
    :ivar tags: List of Tag objects for mutually exclusive classes.
    :ivar n_classes: Number of classes used during training.
    """

    r: float
    tags: list[Tag] | None
    n_classes: int


class AdaptivePredictor(PredictionModule):
    """
    Predictor for multi-label classification using adaptive thresholds.

    The AdaptivePredictor calculates optimal thresholds based on the given
    scores and labels, ensuring the best performance on multi-label data.

    :ivar metadata_dict_name: Filename for saving metadata to disk.
    :ivar n_classes: Number of classes in the dataset.
    :ivar _r: Scaling factor for thresholds.
    :ivar tags: List of Tag objects for mutually exclusive classes.
    :ivar name: Name of the predictor, defaults to "adaptive".

    Parameters
    ----------
    search_space : list[float], optional
        List of threshold scaling factors to search for optimal performance.
        Defaults to a range between 0 and 1.

    Examples
    --------
    >>> from autointent.modules import AdaptivePredictor
    >>> import numpy as np
    >>> scores = np.array([[0.8, 0.1, 0.4], [0.2, 0.9, 0.5]])
    >>> labels = [[1, 0, 0], [0, 1, 0]]
    >>> search_space = [0.1, 0.2, 0.3, 0.5, 0.7]
    >>> predictor = AdaptivePredictor(search_space=search_space)
    >>> predictor.fit(scores, labels)
    >>> predictions = predictor.predict(scores)
    >>> print(predictions)
    [[1 0 0]
     [0 1 0]]

    Save and load the predictor:
    >>> predictor.dump("outputs/")
    >>> predictor_loaded = AdaptivePredictor()
    >>> predictor_loaded.load("outputs/")
    >>> predictions = predictor_loaded.predict(scores)
    >>> print(predictions)
    [[1 0 0]
     [0 1 0]]
    """

    metadata_dict_name = "metadata.json"
    n_classes: int
    _r: float
    tags: list[Tag] | None
    name = "adaptive"

    def __init__(self, search_space: list[float] | None = None) -> None:
        """
        Initialize the AdaptivePredictor.

        :param search_space: List of threshold scaling factors to search for optimal performance.
                             Defaults to a range between 0 and 1.
        """
        self.search_space = search_space if search_space is not None else default_search_space

    @classmethod
    def from_context(cls, context: Context, search_space: list[float] | None = None) -> Self:
        """
        Create an AdaptivePredictor instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param search_space: List of threshold scaling factors, or None for default.
        :return: Initialized AdaptivePredictor instance.
        """
        return cls(
            search_space=search_space,
        )

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: list[LabelType],
        tags: list[Tag] | None = None,
    ) -> None:
        """
        Fit the predictor by optimizing the threshold scaling factor.

        :param scores: Array of shape (n_samples, n_classes) with predicted scores.
        :param labels: List of true multi-label targets.
        :param tags: List of Tag objects for mutually exclusive classes, or None.
        :raises WrongClassificationError: If used on non-multi-label data.
        """
        self.tags = tags
        multilabel = isinstance(labels[0], list)
        if not multilabel:
            msg = (
                "AdaptivePredictor is not designed to perform multiclass classification. "
                "Consider using other predictor algorithms."
            )
            raise WrongClassificationError(msg)
        self.n_classes = len(labels[0]) if multilabel else len(set(labels).difference([-1]))  # type: ignore[arg-type]

        metrics_list = []
        for r in self.search_space:
            y_pred = multilabel_predict(scores, r, self.tags)
            metric_value = multilabel_score(labels, y_pred)
            metrics_list.append(metric_value)

        self._r = float(self.search_space[np.argmax(metrics_list)])

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Predict labels for the given scores.

        :param scores: Array of shape (n_samples, n_classes) with predicted scores.
        :return: Array of shape (n_samples, n_classes) with predicted binary labels.
        :raises InvalidNumClassesError: If the number of classes does not match the trained predictor.
        """
        if scores.shape[1] != self.n_classes:
            msg = "Provided scores number doesn't match with number of classes which predictor was trained on."
            raise InvalidNumClassesError(msg)
        return multilabel_predict(scores, self._r, self.tags)

    def dump(self, path: str) -> None:
        """
        Save the predictor's metadata to disk.

        :param path: Path to the directory where metadata will be saved.
        """
        dump_dir = Path(path)

        metadata = AdaptivePredictorDumpMetadata(r=self._r, tags=self.tags, n_classes=self.n_classes)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(metadata, file, indent=4)

    def load(self, path: str) -> None:
        """
        Load the predictor's metadata from disk.

        :param path: Path to the directory containing saved metadata.
        """
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            metadata: AdaptivePredictorDumpMetadata = json.load(file)

        self._r = metadata["r"]
        self.n_classes = metadata["n_classes"]
        self.tags = [Tag(**tag) for tag in metadata["tags"] if metadata["tags"] and isinstance(metadata["tags"], list)]  # type: ignore[arg-type, union-attr]
        self.metadata = metadata


def get_adapted_threshes(r: float, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Compute adaptive thresholds based on scaling factor and scores.

    :param r: Scaling factor for thresholds.
    :param scores: Array of shape (n_samples, n_classes) with predicted scores.
    :return: Array of thresholds for each class and sample.
    """
    return r * np.max(scores, axis=1) + (1 - r) * np.min(scores, axis=1)  # type: ignore[no-any-return]


def multilabel_predict(scores: npt.NDArray[Any], r: float, tags: list[Tag] | None) -> npt.NDArray[Any]:
    """
    Predict binary labels for multi-label classification.

    :param scores: Array of shape (n_samples, n_classes) with predicted scores.
    :param r: Scaling factor for thresholds.
    :param tags: List of Tag objects for mutually exclusive classes, or None.
    :return: Array of shape (n_samples, n_classes) with predicted binary labels.
    """
    thresh = get_adapted_threshes(r, scores)
    res = (scores >= thresh[:, None]).astype(int)
    if tags:
        res = apply_tags(res, scores, tags)
    return res


def multilabel_score(y_true: list[LabelType], y_pred: npt.NDArray[Any]) -> float:
    """
    Calculate the weighted F1 score for multi-label classification.

    :param y_true: List of true multi-label targets.
    :param y_pred: Array of shape (n_samples, n_classes) with predicted labels.
    :return: Weighted F1 score.
    """
    return f1_score(y_pred, y_true, average="weighted")  # type: ignore[no-any-return]
