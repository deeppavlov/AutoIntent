from typing import Callable

from chromadb import Collection

from ...data_handler import (
    DataHandler,
    multiclass_metadata_as_labels,
    multilabel_metadata_as_labels,
)
from .base import RetrievalModule


class VectorDBModule(RetrievalModule):
    def __init__(self, k: int, model_name: str):
        self.model_name = model_name
        self.k = k

    def fit(self, data_handler: DataHandler):
        self.collection = data_handler.create_collection(self.model_name)

    def score(self, data_handler: DataHandler, metric_fn: Callable) -> tuple[float, str]:
        """
        Return
        ---
        - metric calculated on test set
        - name of embedding model used
        """
        labels_pred = retrieve_candidates_labels(self.collection, self.k, data_handler.utterances_test)
        metric_value = metric_fn(data_handler.labels_test, labels_pred)
        return metric_value, self.model_name
    
    def clear_cache(self):
        model = self.collection._embedding_function._model
        model.to(device='cpu')
        del model
        self.collection = None


def retrieve_candidates_labels(collection: Collection, k: int, utterances: list[str]) -> list[int] | list[list[int]]:
    query_res = collection.query(
        query_texts=utterances,
        n_results=k,
        include=["metadatas", "documents"],  # one can add "embeddings", "distances"
    )
    if not collection.metadata["multilabel"]:
        res = [
            multiclass_metadata_as_labels(candidates)
            for candidates in query_res["metadatas"]
        ]
    else:
        res = [
            multilabel_metadata_as_labels(candidates, collection.metadata["n_classes"])
            for candidates in query_res["metadatas"]
        ]
    return res