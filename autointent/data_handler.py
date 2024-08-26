import itertools as it
import os
from pprint import pprint

import numpy as np
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification
from sklearn.utils import indexable, _safe_indexing


class DataHandler:
    def __init__(
        self, intent_records: os.PathLike, db_path: os.PathLike = "../data/chroma", multilabel: bool = False
    ):
        self.multilabel = multilabel

        (
            self.n_classes,
            self.oos_utterances,
            self.utterances_train,
            self.utterances_test,
            self.labels_train,
            self.labels_test,
        ) = split_sample_utterances(intent_records, multilabel)

        if not multilabel:
            self.regexp_patterns = [
                dict(
                    intent_id=intent["intent_id"],
                    regexp_full_match=intent['regexp_full_match'],
                    regexp_partial_match=intent['regexp_partial_match'],
                )
                for intent in intent_records
            ]

        self.client = PersistentClient(path=db_path)
        self.cache = dict(
            best_assets=dict(
                regexp=None,    # TODO: choose the format
                retrieval=None,  # str, name of best retriever
                scoring=dict(test_scores=None, oos_scores=None),  # dict with values of two np.ndarrays of shape (n_samples, n_classes), from best scorer
                prediction=None,  # np.ndarray of shape (n_samples,), from best predictor
            ),
            metrics=dict(regexp=[], retrieval=[], scoring=[], prediction=[]),
            configs=dict(regexp=[], retrieval=[], scoring=[], prediction=[]),
        )

    def get_collection(self, model_name: str, device="cuda"):
        emb_func = SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            trust_remote_code=True,
            device=device
        )
        db_name = model_name.replace("/", "_")
        collection = self.client.get_or_create_collection(
            name=db_name,
            embedding_function=emb_func,
            metadata={"n_classes": self.n_classes, "multilabel": self.multilabel},
        )
        return collection

    def create_collection(self, model_name: str, device="cuda"):
        collection = self.get_collection(model_name, device)
        db_name = model_name.replace("/", "_")
        
        if self.multilabel:
            metadatas = multilabel_labels_as_metadata(self.labels_train, self.n_classes)
        else:
            metadatas = multiclass_labels_as_metadata(self.labels_train)
        
        collection.add(
            documents=self.utterances_train,
            ids=[f"{i}-{db_name}" for i in range(len(self.utterances_train))],
            metadatas=metadatas,
        )
        return collection

    def delete_collection(self, model_name: str):
        db_name = model_name.replace("/", "_")
        self.client.delete_collection(db_name)

    def log_module_optimization(
            self,
            node_type: str,
            module_type: str,
            module_config: dict,
            metric_value: float,
            metric_name: str,
            assets,
            verbose=False,
        ):
        """
        Purposes:
        - save optimization results in a text form (hyperparameters and corresponding metrics)
        - update best assets
        """

        # "update leaderboard" if it's a new best metric
        metrics_list = self.cache["metrics"][node_type]
        previous_best = max(metrics_list, default=-float("inf"))
        if metric_value > previous_best:
            self.cache["best_assets"][node_type] = assets

        # logging
        logs = dict(
            module_type=module_type,
            metric_name=metric_name,
            metric_value=metric_value,
            **module_config,
        )
        self.cache["configs"][node_type].append(logs)
        if verbose:
            pprint(logs)
        metrics_list.append(metric_value)

    def get_best_collection(self, device="cuda"):
        model_name = self.cache["best_assets"]["retrieval"]
        return self.get_collection(model_name, device)

    def get_best_test_scores(self):
        return self.cache["best_assets"]["scoring"]["test_scores"]
    
    def get_best_oos_scores(self):
        return self.cache["best_assets"]["scoring"]["oos_scores"]

    def dump_logs(self):
        res = dict(
            metrics=self.cache["metrics"],
            configs=self.cache["configs"],
        )
        return res

    def get_prediction_evaluation_data(self):
        labels = self.labels_test
        scores = self.get_best_test_scores()

        oos_scores = self.get_best_oos_scores()
        if oos_scores is not None:
            if self.multilabel:
                oos_labels = [[0] * self.n_classes] * len(oos_scores)
            else:
                oos_labels = [-1] * len(oos_scores)
            labels = np.concatenate([labels, oos_labels])
            scores = np.concatenate([scores, oos_scores])

        return labels, scores


def get_sample_utterances(intent_records: list[dict]):
    """get plain list of all sample utterances and their intent labels"""
    utterances = [intent["sample_utterances"] for intent in intent_records]
    labels = [
        [intent["intent_id"]] * len(uts) for intent, uts in zip(intent_records, utterances)
    ]

    utterances = list(it.chain.from_iterable(utterances))
    labels = list(it.chain.from_iterable(labels))

    return utterances, labels


def split_sample_utterances(intent_records: list[dict], multilabel: bool):
    """
    Return: utterances_train, utterances_test, labels_train, labels_test

    TODO: ensure stratified train test splitting (test set must contain all classes)
    """

    if not multilabel:
        utterances, labels = get_sample_utterances(intent_records)
        in_domain_mask = np.array(labels) != -1

        in_domain_utterances = [ut for ut, is_in_domain in zip(utterances, in_domain_mask) if is_in_domain]
        in_domain_labels = [lab for lab, is_in_domain in zip(labels, in_domain_mask) if is_in_domain]
        oos_utterances = [ut for ut, is_in_domain in zip(utterances, in_domain_mask) if not is_in_domain]
        
        n_classes = len(set(in_domain_labels))
        splits = train_test_split(
            in_domain_utterances,
            in_domain_labels,
            test_size=0.25,
            random_state=0,
            stratify=in_domain_labels,
            shuffle=True,
        )
    else:
        utterance_records = intent_records
        utterances = [dct["utterance"] for dct in utterance_records]
        labels = [dct["labels"] for dct in utterance_records]

        n_classes = len(set(it.chain.from_iterable(labels)))

        in_domain_utterances = [ut for ut, lab in zip(utterances, labels) if len(lab) > 0]
        in_domain_labels = [[int(i in lab) for i in range(n_classes)] for lab in labels if len(lab) > 0]    # binary labels
        oos_utterances = [ut for ut, lab in zip(utterances, labels) if len(lab) == 0]
        
        splits = multilabel_train_test_split(
            in_domain_utterances,
            in_domain_labels,
            test_size=0.25,
        )
    
    res = [n_classes, oos_utterances] + splits
    return res


def multiclass_labels_as_metadata(labels_list: list[int]):
    return [{"intent_id": lab} for lab in labels_list]


def multilabel_labels_as_metadata(labels_list: list[list[int]], n_classes):
    """labels_list is already in binary format"""
    return [{str(i): lab for i, lab in enumerate(labs)} for labs in labels_list]


def multiclass_metadata_as_labels(metadata: list[dict]):
    return [dct["intent_id"] for dct in metadata]


def multilabel_metadata_as_labels(metadata: list[dict], n_classes):
    return [[dct[str(i)] for i in range(n_classes)] for dct in metadata]


def multilabel_train_test_split(*arrays, stratify=None, test_size=0.25):
    if stratify is None:
        stratify = np.array(arrays[-1])

    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)

    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0-test_size])
    train, test = next(stratifier.split(arrays[0], stratify))

    return list(
        it.chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )