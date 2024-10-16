"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continuous labels 0...1 to indicate
the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_stsbenchmark.py
"""

import itertools as it
import logging
from random import shuffle
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sentence_transformers import CrossEncoder
from sklearn.linear_model import LogisticRegressionCV

logger = logging.getLogger(__name__)


def construct_samples(
    texts: list[str], labels: list[Any], balancing_factor: int | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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


class CrossEncoderWithLogreg:
    # TODO refactor

    def __init__(self, model: CrossEncoder, batch_size: int = 16, verbose: bool = False) -> None:
        self.cross_encoder = model
        self.batch_size = batch_size
        self.verbose = verbose

    @torch.no_grad()
    def get_features(self, pairs: list[tuple[str, str]]) -> npt.NDArray[Any]:
        logits_list: list[npt.NDArray[Any]] = []

        def hook_function(module, input_tensor, output_tensor) -> None:  # type: ignore[no-untyped-def] # noqa: ARG001, ANN001
            logits_list.append(input_tensor[0].cpu().numpy())

        handler = self.cross_encoder.model.classifier.register_forward_hook(hook_function)

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            self.cross_encoder.predict(batch)

        handler.remove()

        return np.concatenate(logits_list, axis=0)

    def _fit(self, pairs: list[tuple[str, str]], labels: list[int]) -> None:
        """
        Arguments
        ---
        - `pairs`: list of text pairs as strings
        - `labels`: binary labels (1 = same class, 0 = different classes)
        """
        n_samples = len(pairs)
        if n_samples != len(labels):
            msg = "Something went wrong"
            logger.error(msg)
            raise ValueError(msg)

        features = self.get_features(pairs)

        clf = LogisticRegressionCV()
        clf.fit(features, labels)

        self._clf = clf

    def fit(self, utterances: list[str], labels: list[int]) -> None:
        """
        Construct train samples for binary classifier over cross-encoder features

        Arguments
        ---
        - `utterances`: list of text pairs as strings
        - `labels`: intent class labels
        """
        pairs, labels_ = construct_samples(utterances, labels, balancing_factor=1)
        self._fit(pairs, labels_)  # type: ignore[arg-type]

    def predict(self, pairs: list[tuple[str, str]]) -> npt.NDArray[Any]:
        """
        Return probabilities of two utterances having the same intent label
        """
        features = self.get_features(pairs)

        return self._clf.predict_proba(features)[:, 1]  # type: ignore[no-any-return]
