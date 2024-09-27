"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continuous labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_stsbenchmark.py
"""

import sys

sys.path.append("/home/voorhs/repos/AutoIntent")

import itertools as it
import os
from random import shuffle

import joblib
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split


def construct_samples(texts, labels, balancing_factor: int = None) -> tuple[list[dict], list[dict]]:
    samples = [[], []]

    for (i, text1), (j, text2) in it.combinations(enumerate(texts), 2):
        pair = [text1, text2]
        label = int(labels[i] == labels[j])
        sample = dict(texts=pair, label=label)
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

        features = np.concatenate(logits_list, axis=0)

        return features

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
            print(f"Best accuracy: {acc:2.2f} (threshold {thresh1:.2f})")
            print(f"Best f1: {f1:2.2f} (threshold {thresh2:.2f})")
            print(f"Best precision: {prec:2.2f}")
            print(f"Best recall: {rec:2.2f}")

        if dump_logs:
            return dict(
                best_accuracy=acc,
                optimal_thresh_acc=thresh1,
                best_f1=f1,
                optimal_thresh_f1=thresh2,
                best_precision=prec,
                best_recall=rec,
            )

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


if __name__ == "__main__":
    import json
    from argparse import ArgumentParser
    from datetime import datetime

    from autointent.data_handler import get_sample_utterances

    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, default="llmrails/ember-v1")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="experiments/cross-encoder-training/logs-sklearn",
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    run_name = args.run_name if args.run_name != "" else args.model_name.replace("/", "_")
    run_name = run_name + "_" + datetime.now().strftime("%m-%d-%Y_%H:%M:%S")

    run_dir = os.path.join(args.logs_dir, run_name)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Define our Cross-Encoder
    model = CrossEncoderWithLogreg(CrossEncoder("llmrails/ember-v1"), batch_size=args.batch_size)

    # Read dataset
    dataset_path = "data/intent_records/banking77.json"
    intent_records = json.load(open(dataset_path))

    utterances, labels = get_sample_utterances(intent_records)
    texts_train, texts_test, labels_train, labels_test = construct_samples(utterances, labels, balancing_factor=3)

    # train
    with torch.no_grad():
        model.fit(texts_train, labels_train)
        logs = model.score(texts_test, labels_test, dump_logs=True)

    logs_path = os.path.join(run_dir, "logs.json")
    json.dump(logs, open(logs_path, "w"), indent=4, ensure_ascii=False)
    model.save_model(os.path.join(run_dir, "checkpoint.joblib"))
