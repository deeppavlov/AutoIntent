"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continuous labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_stsbenchmark.py
"""

import itertools as it
import os
from random import shuffle

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split


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

    texts = [dct["texts"] for dct in samples]
    labels = [dct["label"] for dct in samples]

    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=0,
        stratify=labels,
        shuffle=True,
    )

    return texts_train, texts_test, labels_train, labels_test


class CrossEncoderWithLogreg:
    def __init__(self, model, batch_size=16, verbose=False):
        self.cross_encoder = model
        self.batch_size = batch_size
        self.verbose = verbose

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

    def fit(self, pairs, labels):
        n_samples = len(pairs)
        assert n_samples == len(labels)

        features = self.get_features(pairs)

        clf = LogisticRegressionCV()
        clf.fit(features, labels)

        self._clf = clf

    def predict_proba(self, pairs):
        features = self.get_features(pairs)

        return self._clf.predict_proba(features)[:, 1]

    def score(self, pairs, labels, dump_logs=False):
        probas = self.predict_proba(pairs)

        acc, thresh1 = find_best_acc_and_threshold(probas, labels)
        f1, prec, rec, thresh2 = find_best_f1_and_threshold(probas, labels)

        if self.verbose:
            pass

        if dump_logs:
            return {
                "best_accuracy": acc,
                "optimal_thresh_acc": thresh1,
                "best_f1": f1,
                "optimal_thresh_f1": thresh2,
                "best_precision": prec,
                "best_recall": rec,
            }
        return None

    def save_model(self, path: os.PathLike):
        joblib.dump(self._clf, path)

    def load_model(self, path: os.PathLike):
        self._clf = joblib.load(path)


def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool = True):
    assert len(scores) == len(labels)
    rows = list(zip(scores, labels, strict=False))

    rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

    max_acc = 0
    best_threshold = -1

    positive_so_far = 0
    remaining_negatives = sum(np.array(labels) == 0)

    for i in range(len(rows) - 1):
        score, label = rows[i]
        if label == 1:
            positive_so_far += 1
        else:
            remaining_negatives -= 1

        acc = (positive_so_far + remaining_negatives) / len(labels)
        if acc > max_acc:
            max_acc = acc
            best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

    return max_acc, best_threshold


def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool = True):
    assert len(scores) == len(labels)

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    rows = list(zip(scores, labels, strict=False))

    rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

    best_f1 = best_precision = best_recall = 0
    threshold = 0
    nextract = 0
    ncorrect = 0
    total_num_duplicates = sum(labels)

    for i in range(len(rows) - 1):
        score, label = rows[i]
        nextract += 1

        if label == 1:
            ncorrect += 1

        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / total_num_duplicates
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                threshold = (rows[i][0] + rows[i + 1][0]) / 2

    return best_f1, best_precision, best_recall, threshold
