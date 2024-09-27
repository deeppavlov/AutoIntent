"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continuous labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_stsbenchmark.py
"""

import itertools as it
from random import shuffle

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from sklearn.linear_model import LogisticRegressionCV


def construct_samples(texts, labels, balancing_factor: int | None = None) -> tuple[list[dict], list[dict]]:
    samples = [[], []]

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

    pairs = [dct["texts"] for dct in samples]
    labels = [dct["label"] for dct in samples]
    return pairs, labels


class CrossEncoderWithLogreg:
    def __init__(self, model: CrossEncoder, batch_size=16, verbose=False):
        self.cross_encoder = model
        self.batch_size = batch_size
        self.verbose = verbose

    @torch.no_grad()
    def get_features(self, pairs):
        logits_list = []

        def hook_function(module, input, output):
            logits_list.append(input[0].cpu().numpy())

        handler = self.cross_encoder.model.classifier.register_forward_hook(hook_function)

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            self.cross_encoder.predict(batch)

        handler.remove()

        return np.concatenate(logits_list, axis=0)


    def _fit(self, pairs: list[tuple[str, str]], labels: list[int]):
        """
        Arguments
        ---
        - `pairs`: list of text pairs as strings
        - `labels`: binary labels (1 = same class, 0 = different classes)
        """
        n_samples = len(pairs)
        assert n_samples == len(labels)

        features = self.get_features(pairs)

        clf = LogisticRegressionCV()
        clf.fit(features, labels)

        self._clf = clf

    def fit(self, utterances: list[str], labels: list[int]):
        """
        Construct train samples for binary classifier over cross-encoder features

        Arguments
        ---
        - `utterances`: list of text pairs as strings
        - `labels`: intent class labels
        """
        pairs, labels = construct_samples(utterances, labels, balancing_factor=1)
        self._fit(pairs, labels)

    def predict(self, pairs):
        """
        Return probabilities of two utterances having the same intent label
        """
        features = self.get_features(pairs)

        return self._clf.predict_proba(features)[:, 1]
