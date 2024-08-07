from typing import Callable

from ..base import DataHandler
from .base import RetrievalModule


class VectorDBModule(RetrievalModule):
    def __init__(self, k: int, model_name: str, device="cuda"):
        self.model_name = model_name
        self.k = k
        self.device = device

    def fit(self, data_handler: DataHandler):
        data_handler.create_collection(
            self.model_name,
            device=self.device,
        )

    def score(self, data_handler: DataHandler, metric_fn: Callable) -> tuple[float, str]:
        """
        Return
        ---
        - metric calculated on test set
        - name of embedding model used
        """
        collection = data_handler.get_collection(self.model_name, self.device) 
        query_res = collection.query(
            query_texts=data_handler.utterances_test,
            n_results=self.k,
            include=["metadatas", "documents"],  # one can add "embeddings", "distances"
        )
        labels_pred = [
            [cand["intent_id"] for cand in candidates]
            for candidates in query_res["metadatas"]
        ]
        metric_value = metric_fn(data_handler.labels_test, labels_pred)
        return metric_value, self.model_name
