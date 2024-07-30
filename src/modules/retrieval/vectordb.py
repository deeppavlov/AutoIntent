from typing import Callable

from ..base import DataHandler
from .base import RetrievalModule


class VectorDBModule(RetrievalModule):
    def __init__(self, k: int, model_name: str):
        self.model_name = model_name
        self.k = k

    def fit(self, data_handler: DataHandler):
        data_handler.create_collection(self.model_name)

    def score(self, data_handler: DataHandler, metric_fn: Callable):
        query_res = data_handler.collection.query(
            query_texts=data_handler.utterances_test,
            n_results=self.k,
            include=["metadatas", "documents"],  # one can add "embeddings", "distances"
        )
        labels_pred = [
            [cand["intent_id"] for cand in candidates]
            for candidates in query_res["metadatas"]
        ]
        return metric_fn(data_handler.labels_test, labels_pred)
