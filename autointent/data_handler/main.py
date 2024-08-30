import os
from pprint import pprint

import numpy as np
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# from .embedding_func import MyEmbeddingFunction

from .stratified_splitting import (
    multiclass_labels_as_metadata,
    multilabel_labels_as_metadata,
    split_sample_utterances,
)


class DataHandler:
    def __init__(
        self, intent_records: os.PathLike, device: str, db_path: os.PathLike = "../data/chroma", multilabel: bool = False
    ):
        self.multilabel = multilabel
        self.device = device

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

        settings = Settings(
            chroma_segment_cache_policy="LRU",
            chroma_memory_limit_bytes=2*1024*1024*1024,  # 2 GB
        )
        self.client = PersistentClient(path=db_path, settings=settings)
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

    def get_collection(self, model_name: str, device=None):
        device = device if device is not None else self.device
        emb_func = SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            trust_remote_code=True,
            device=device,
            tokenizer_kwargs=dict(truncation=True)
        )
        db_name = model_name.replace("/", "_")
        collection = self.client.get_or_create_collection(
            name=db_name,
            embedding_function=emb_func,
            metadata={"n_classes": self.n_classes, "multilabel": self.multilabel},
        )
        return collection

    def create_collection(self, model_name: str, device=None):
        device = device if device is not None else self.device

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
