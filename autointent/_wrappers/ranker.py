"""Module for cross-encoder-based meaning closeness estimation using ranking.

This module provides functionality for ranking retrieved sentences by meaning closeness
to provided utterances using cross-encoder models.
"""

import gc
import itertools as it
import json
import logging
from pathlib import Path
from random import shuffle
from typing import Any, Literal, TypedDict

import joblib
import numpy as np
import numpy.typing as npt
import sentence_transformers as st
import torch
from sklearn.linear_model import LogisticRegressionCV
from torch import nn

from autointent.configs import CrossEncoderConfig
from autointent.custom_types import ListOfLabels

logger = logging.getLogger(__name__)


class CrossEncoderMetadata(TypedDict):
    """Metadata for CrossEncoder model.

    Attributes:
        model_name: Name of the model
        train_head: Whether to train a classifier
        device: Device to use for inference
        max_length: Maximum sequence length
        batch_size: Batch size for inference
    """

    model_name: str
    train_head: bool
    device: str | None
    max_length: int | None
    batch_size: int


def construct_samples(
    texts: list[str],
    labels: list[Any],
    balancing_factor: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct balanced samples of text pairs for training.

    Args:
        texts: List of texts to create pairs from
        labels: List of labels corresponding to the texts
        balancing_factor: Factor for balancing positive and negative samples

    Returns:
        Tuple containing:
            - List of text pairs
            - List of corresponding binary labels
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
    """Cross-encoder for Natural Language Inference (NLI).

    This class uses :py:class:`sentence_transformers.cross_encoder.CrossEncoder` model to extract features.
    It can use either the model's classifier or a custom trained :py:class:`sklearn.linear_model.LogisticRegressionCV`
    to rank documents using similarity scores to the query.
    """

    _metadata_file_name = "metadata.json"
    _classifier_file_name = "classifier.joblib"
    config: CrossEncoderConfig
    cross_encoder: st.CrossEncoder

    def __init__(
        self,
        cross_encoder_config: CrossEncoderConfig | str | dict[str, Any],
        classifier_head: LogisticRegressionCV | None = None,
        output_range: Literal["sigmoid", "tanh"] = "sigmoid",
    ) -> None:
        """Initialize the Ranker.

        Args:
            cross_encoder_config: Configuration for the cross-encoder model
            classifier_head: Optional pre-trained classifier head
            output_range: Range of the output probabilities ([0, 1] for sigmoid, [-1, 1] for tanh)
        """
        self.config = CrossEncoderConfig.from_search_config(cross_encoder_config)
        self.cross_encoder = st.CrossEncoder(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            device=self.config.device,
            max_length=self.config.tokenizer_config.max_length,  # type: ignore[arg-type]
        )
        self._train_head = False
        self._clf = classifier_head
        self.output_range = output_range

        if classifier_head is not None or self.config.train_head:
            self._train_head = True
            self._activations_list: list[npt.NDArray[Any]] = []
            self._hook_handler = self.cross_encoder.model.classifier.register_forward_hook(self._classifier_hook)

    def _classifier_hook(self, _module, input_tensor, _output_tensor) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001
        """Hook to capture classifier activations.

        Args:
            _module: Module being hooked
            input_tensor: Input tensor to the classifier
            _output_tensor: Output tensor from the classifier
        """
        self._activations_list.append(input_tensor[0].cpu().numpy())

    @torch.no_grad()
    def _get_features_or_predictions(self, pairs: list[tuple[str, str]]) -> npt.NDArray[Any]:
        """Extract features or get predictions using the Ranker model.

        Args:
            pairs: List of text pairs

        Returns:
            Array of extracted features or predictions
        """
        if not self._train_head:
            return np.array(
                self.cross_encoder.predict(
                    pairs,
                    batch_size=self.config.batch_size,
                    activation_fct=nn.Sigmoid() if self.output_range == "sigmoid" else nn.Tanh(),
                )
            )

        self.cross_encoder.predict(pairs, batch_size=self.config.batch_size)
        res = np.concatenate(self._activations_list, axis=0)
        self._activations_list.clear()
        return res  # type: ignore[no-any-return]

    def _fit(self, pairs: list[tuple[str, str]], labels: ListOfLabels) -> None:
        """Train the logistic regression model on cross-encoder features.

        Args:
            pairs: List of text pairs
            labels: Binary labels (1 = same class, 0 = different classes)

        Raises:
            ValueError: If number of pairs and labels don't match
        """
        n_samples = len(pairs)
        if n_samples != len(labels):
            msg = "Number of pairs and labels do not match."
            logger.error(msg)
            raise ValueError(msg)

        features = self._get_features_or_predictions(pairs)
        clf = LogisticRegressionCV()
        clf.fit(features, labels)
        self._clf = clf

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        """Construct training samples and train the logistic regression classifier.

        Args:
            utterances: List of utterances (texts)
            labels: Intent class labels corresponding to the utterances
        """
        if not self._train_head:
            return

        pairs, labels_ = construct_samples(utterances, labels, balancing_factor=1)
        self._fit(pairs, labels_)  # type: ignore[arg-type]

    def predict(self, pairs: list[tuple[str, str]]) -> npt.NDArray[Any]:
        """Predict probabilities of two utterances having the same intent label.

        Args:
            pairs: List of text pairs to classify

        Returns:
            Array of probabilities

        Raises:
            ValueError: If classifier is not trained yet
        """
        if self._train_head and self._clf is None:
            msg = "Classifier is not trained yet"
            raise ValueError(msg)

        features = self._get_features_or_predictions(pairs)

        if self._clf is not None:
            probs = np.array(self._clf.predict_proba(features)[:, 1])
            if self.output_range == "tanh":
                probs = probs * 2 - 1
            return probs
        return features

    def rank(
        self,
        query: str,
        query_docs: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rank documents according to meaning closeness to the query.

        Args:
            query: Reference document
            query_docs: List of documents to rank
            top_k: Number of documents to return

        Returns:
            List of dictionaries containing ranked items with scores
        """
        query_doc_pairs = [(query, doc) for doc in query_docs]
        scores = self.predict(query_doc_pairs)

        if top_k is None:
            top_k = len(query_docs)

        results = [{"corpus_id": i, "score": scores[i]} for i in range(len(query_docs))]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, path: str) -> None:
        """Save the model and classifier to disk.

        Args:
            path: Directory path to save the model and classifier
        """
        dump_dir = Path(path)
        dump_dir.mkdir(parents=True)

        metadata = CrossEncoderMetadata(
            model_name=self.config.model_name,
            train_head=self._train_head,
            device=self.config.device,
            max_length=self.config.tokenizer_config.max_length,
            batch_size=self.config.batch_size,
        )

        with (dump_dir / self._metadata_file_name).open("w") as file:
            json.dump(metadata, file, indent=4)

        joblib.dump(self._clf, dump_dir / self._classifier_file_name)

    @classmethod
    def load(cls, path: Path, override_config: CrossEncoderConfig | None = None) -> "Ranker":
        """Load the model and classifier from disk.

        Args:
            path: Directory path containing the saved model and classifier
            override_config: one can override presaved settings

        Returns:
            Initialized Ranker instance
        """
        clf = joblib.load(path / cls._classifier_file_name)

        with (path / cls._metadata_file_name).open(encoding="utf-8") as file:
            metadata: CrossEncoderMetadata = json.load(file)

        if override_config is not None:
            kwargs = {**metadata, **override_config.model_dump(exclude_unset=True)}
        else:
            kwargs = metadata  # type: ignore[assignment]

        max_length = kwargs.pop("max_length", None)
        if max_length is not None:
            kwargs["tokenizer_config"] = {"max_length": max_length}

        return cls(
            CrossEncoderConfig(**kwargs),
            classifier_head=clf,
        )

    def clear_ram(self) -> None:
        """Clear model from RAM and GPU memory."""
        self.cross_encoder.model.cpu()
        del self.cross_encoder
        gc.collect()
        torch.cuda.empty_cache()
