"""Ranker class for cross-encoder-based estimation of meaning closeness.

Can be used to rank retrieved sentences by meaning closeness to provided utterance.
"""

import itertools as it
import json
import logging
from pathlib import Path
from random import shuffle
from typing import Any, TypedDict

import joblib
import numpy as np
import numpy.typing as npt
import sentence_transformers as st
import torch
from sklearn.linear_model import LogisticRegressionCV
from torch import nn

from autointent.custom_types import LabelType

logger = logging.getLogger(__name__)


class CrossEncoderMetadata(TypedDict):
    model_name: str
    train_classifier: bool
    device: str
    max_length: int | None
    batch_size: int


def construct_samples(
    texts: list[str],
    labels: list[Any],
    balancing_factor: int | None = None,
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


class Ranker:
    r"""
    Cross-encoder for NLI.

    In the hart this class uses a SentenceTransformers Ranker model to extract features.
    Then it uses either the model's clissifier or our custom trained LogisticRegressionCV
    (custom classifier layer in the future) to rank documents using similarity score to the query.

    :ivar cross_encoder: The Ranker model used to extract features.
    :ivar batch_size: Batch size for processing text pairs.
    :ivar _clf: The trained LogisticRegressionCV classifier.
    :ivar model_subdir: Directory for storing the cross-encoder model files.

    Examples
    --------
    Creating and fitting the CrossEncoderWithLogreg:
    >>> from autointent import Ranker
    >>> scorer = Ranker("cross-encoder-model")
    >>> utterances = ["What is your name?", "How old are you?"]
    >>> labels = [1, 0]
    >>> scorer.fit(utterances, labels)

    Predicting probabilities:
    >>> test_pairs = [["What is your name?", "Hello!"], ["How old are you?", "What is your age?"]]
    >>> probs = scorer.predict(test_pairs)
    >>> print(probs)

    Saving and loading the model:
    >>> scorer.save("outputs/")
    >>> loaded_scorer = Ranker.load("outputs/")
    """

    metadata_file_name = "metadata.json"
    classifier_file_name = "classifier.joblib"

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        train_classifier: bool = False,
        batch_size: int = 326,
        max_length: int | None = None,
        classifier_head: LogisticRegressionCV | None = None,
    ) -> None:
        """
        Initialize the Ranker.

        :param model: The cross-encoder hugging face model name to use.
        :param device: Device to run operations on, e.g., "cpu" or "cuda".
        :param train_classifier: Whether to train a custom classifier, defaults to False.
        :param batch_size: Batch size for processing text pairs, defaults to 326.
        :param max_length (int, optional): Max length for input sequences for the cross encoder.
        :param classifier_head (LogisticRegressionCV, optional): Classifier (to be used in restore procedure mainly).
        """
        self.model_name = model_name
        self.device = device
        self.cross_encoder = st.CrossEncoder(model_name, trust_remote_code=True, device=device, max_length=max_length)  # type: ignore[arg-type]
        self.train_classifier = False
        self.batch_size = batch_size
        self.max_length = max_length
        self._clf = classifier_head

        if classifier_head is not None or train_classifier:
            self.train_classifier = True
            self._activations_list: list[npt.NDArray[Any]] = []
            self._hook_handler = self.cross_encoder.model.classifier.register_forward_hook(self._classifier_hook)

    def _classifier_hook(self, _module, input_tensor, _output_tensor) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001
        self._activations_list.append(input_tensor[0].cpu().numpy())

    @torch.no_grad()
    def _get_features_or_predictions(self, pairs: list[tuple[str, str]]) -> npt.NDArray[Any]:
        """
        Extract features or get predictions using the Ranker model.

        If :py:attr:`~train_classifier` is ``True``, return raw activations from
        cross-encoder transformer. Otherwise, get predictions from cross-encoder head.

        :param pairs: List of text pairs.
        :return: Numpy array of extracted features.
        """
        if not self.train_classifier:
            return np.array(self.cross_encoder.predict(pairs, batch_size=self.batch_size, activation_fct=nn.Sigmoid()))

        # put the data through, features will be taken in the hook
        self.cross_encoder.predict(pairs, batch_size=self.batch_size)

        res = np.concatenate(self._activations_list, axis=0)
        self._activations_list.clear()
        return res  # type: ignore[no-any-return]

    def _fit(self, pairs: list[tuple[str, str]], labels: list[LabelType]) -> None:
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

        features = self._get_features_or_predictions(pairs)

        # TODO: LogisticRegressionCV has class_weight="balanced". Is it better to use it instead of balance_factor in
        # construct_samples?
        clf = LogisticRegressionCV()
        clf.fit(features, labels)

        self._clf = clf

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        """
        Construct training samples and train the logistic regression classifier.

        :param utterances: List of utterances (texts).
        :param labels: Intent class labels corresponding to the utterances.
        """
        if not self.train_classifier:
            return  # do nothing if the classifier is not to be re-trained

        pairs, labels_ = construct_samples(utterances, labels, balancing_factor=1)
        self._fit(pairs, labels_)  # type: ignore[arg-type]

    def predict(self, pairs: list[tuple[str, str]]) -> npt.NDArray[Any]:
        """
        Predict probabilities of two utterances having the same intent label.

        :param pairs: List of text pairs to classify.
        :return: Numpy array of probabilities.
        """
        if self.train_classifier and self._clf is None:
            msg = "Classifier is not trained yet"
            raise ValueError(msg)

        features = self._get_features_or_predictions(pairs)

        if self._clf is not None:
            return np.array(self._clf.predict_proba(features)[:, 1])

        return features

    def rank(
        self,
        query: str,
        query_docs: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rank documents according to meaning closeness to the query.

        :param query: The reference document.
        :query_docs: List of documents to rank
        :top_k: how many document to return
        :return: array of dictionaries of ranked items.
        """
        query_doc_pairs = [(query, doc) for doc in query_docs]
        scores = self.predict(query_doc_pairs)

        if top_k is None:
            top_k = len(query_docs)

        results = [{"corpus_id": i, "score": scores[i]} for i in range(len(query_docs))]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, path: str) -> None:
        """
        Save the model and classifier to disk.

        :param path: Directory path to save the model and classifier.
        """
        dump_dir = Path(path)
        dump_dir.mkdir(parents=True)

        metadata = CrossEncoderMetadata(
            model_name=self.model_name,
            train_classifier=self.train_classifier,
            device=self.device,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )

        with (dump_dir / self.metadata_file_name).open("w") as file:
            json.dump(metadata, file, indent=4)

        joblib.dump(self._clf, dump_dir / self.classifier_file_name)

    @classmethod
    def load(cls, path: Path) -> "Ranker":
        """
        Load the model and classifier from disk.

        :param path: Directory path containing the saved model and classifier.
        :return: Initialized Ranker instance.
        """
        clf = joblib.load(path / cls.classifier_file_name)

        with (path / cls.metadata_file_name).open() as file:
            metadata: CrossEncoderMetadata = json.load(file)

        return cls(**metadata, classifier_head=clf)
