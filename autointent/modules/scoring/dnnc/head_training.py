"""CrossEncoderWithLogreg class for cross-encoder-based binary classification with logistic regression."""

import itertools as it
import logging
from pathlib import Path
from random import shuffle
from typing import Any, TypeVar

import joblib
import numpy as np
import numpy.typing as npt
import torch
from sentence_transformers import CrossEncoder
from sklearn.linear_model import LogisticRegressionCV
from typing_extensions import Self

from autointent.custom_types import LabelType

logger = logging.getLogger(__name__)


def construct_samples(
    texts: list[str], labels: list[Any], balancing_factor: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Construct balanced samples of text pairs for training.

    :param texts: List of texts to create pairs from.
    :param labels: List of labels corresponding to the texts.
    :param balancing_factor: Factor for balancing the positive and negative samples. If None, no balancing is applied.
    :return: Tuple containing lists of text pairs and their corresponding binary labels.
    """
    samples = [[], []]  # type: ignore[var-annotated]

    for (i, text1), (j, text2) in it.combinations(enumerate(texts), 2):
        pair = [text1, text2]
        label = int(labels[i] == labels[j])
        sample = {"texts": pair, "label": label}
        samples[label].append(sample)
    shuffle(samples[0])
    shuffle(samples[1])

    if balancing_factor is not None:
        i_min = min([0, 1], key=lambda i: len(samples[i]))
        i_max = 1 - i_min
        min_length = len(samples[i_min])
        samples = samples[i_min][:min_length] + samples[i_max][: min_length * balancing_factor]
    else:
        samples = samples[0] + samples[1]

    pairs = [dct["texts"] for dct in samples]  # type: ignore[call-overload]
    labels = [dct["label"] for dct in samples]  # type: ignore[call-overload]
    return pairs, labels


CrossEncoderType = TypeVar("CrossEncoderType", bound="CrossEncoderWithLogreg")


class CrossEncoderWithLogreg:
    """
    Cross-encoder with logistic regression for binary classification.

    This class uses a SentenceTransformers CrossEncoder model to extract features
    and LogisticRegressionCV for classification.
    """

    def __init__(self, model: CrossEncoder, batch_size: int = 326) -> None:
        """
        Initialize the CrossEncoderWithLogreg.

        :param model: The CrossEncoder model to use.
        :param batch_size: Batch size for processing text pairs, defaults to 326.
        """
        self.cross_encoder = model
        self.batch_size = batch_size

    @torch.no_grad()
    def get_features(self, pairs: list[list[str]]) -> npt.NDArray[Any]:
        """
        Extract features from text pairs using the CrossEncoder model.

        :param pairs: List of text pairs.
        :return: Numpy array of extracted features.
        """
        logits_list: list[npt.NDArray[Any]] = []

        def hook_function(module, input_tensor, output_tensor) -> None:  # type: ignore[no-untyped-def] # noqa: ARG001, ANN001
            logits_list.append(input_tensor[0].cpu().numpy())

        handler = self.cross_encoder.model.classifier.register_forward_hook(hook_function)

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            self.cross_encoder.predict(batch)

        handler.remove()

        return np.concatenate(logits_list, axis=0)

    def _fit(self, pairs: list[list[str]], labels: list[LabelType]) -> None:
        """
        Train the logistic regression model on cross-encoder features.

        :param pairs: List of text pairs.
        :param labels: Binary labels (1 = same class, 0 = different classes).
        :raises ValueError: If the number of pairs and labels do not match.
        """
        n_samples = len(pairs)
        if n_samples != len(labels):
            msg = "Number of pairs and labels do not match."
            logger.error(msg)
            raise ValueError(msg)

        features = self.get_features(pairs)

        clf = LogisticRegressionCV()
        clf.fit(features, labels)

        self._clf = clf

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        """
        Construct training samples and train the logistic regression classifier.

        :param utterances: List of utterances (texts).
        :param labels: Intent class labels corresponding to the utterances.
        """
        pairs, labels_ = construct_samples(utterances, labels, balancing_factor=1)
        self._fit(pairs, labels_)  # type: ignore[arg-type]

    def predict(self, pairs: list[list[str]]) -> npt.NDArray[Any]:
        """
        Predict probabilities of two utterances having the same intent label.

        :param pairs: List of text pairs to classify.
        :return: Numpy array of probabilities.
        """
        features = self.get_features(pairs)
        return self._clf.predict_proba(features)[:, 1]  # type: ignore[no-any-return]

    def save(self, path: str) -> None:
        """
        Save the model and classifier to disk.

        :param path: Directory path to save the model and classifier.
        """
        dump_dir = Path(path)

        crossencoder_dir = str(dump_dir / "crossencoder")
        self.cross_encoder.save(crossencoder_dir)

        clf_path = dump_dir / "classifier.joblib"
        joblib.dump(self._clf, clf_path)

    def set_classifier(self, clf: LogisticRegressionCV) -> None:
        """
        Set the logistic regression classifier.

        :param clf: LogisticRegressionCV instance.
        """
        self._clf = clf

    @classmethod
    def load(cls, path: str) -> Self:
        """
        Load the model and classifier from disk.

        :param path: Directory path containing the saved model and classifier.
        :return: Initialized CrossEncoderWithLogreg instance.
        """
        dump_dir = Path(path)

        # Load sklearn model
        clf_path = dump_dir / "classifier.joblib"
        clf = joblib.load(clf_path)

        # Load sentence transformer model
        crossencoder_dir = str(dump_dir / "crossencoder")
        model = CrossEncoder(crossencoder_dir)

        res = cls(model)
        res.set_classifier(clf)

        return res
