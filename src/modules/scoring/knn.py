import numpy as np

from .base import DataHandler, ScoringModule


class KNNScorer(ScoringModule):
    """
    TODO:
    - add weighted knn?
    """

    def __init__(self, k, device="cuda"):
        self.k = k
        self.device = device

    def fit(self, data_handler: DataHandler):
        self._collection = data_handler.get_best_collection(self.device)
        self._n_classes = data_handler.n_classes

    def predict(self, utterances: list[str]):
        query_res = self._collection.query(
            query_texts=utterances,
            n_results=self.k,
            include=["metadatas", "documents"],  # one can add "embeddings", "distances"
        )

        labels_pred = [
            [cand["intent_id"] for cand in candidates]
            for candidates in query_res["metadatas"]
        ]
        y = np.array(labels_pred)

        counts = get_counts(y, self._n_classes)

        return counts / counts.sum(axis=1, keepdims=True)


def get_counts(labels, n_classes):
    """
    Arguments
    ---
    `y`: np.ndarray of shape (n_samples, n_candidates) with integer labels from `[0,n_classes-1]`

    Return
    ---
    np.ndarray of shape (n_samples, n_classes) with statistics of how many times each class label occured in candidates
    """
    n_queries = labels.shape[0]
    labels += n_classes * np.arange(n_queries)[:, None]
    counts = np.bincount(labels.ravel(), minlength=n_classes * n_queries).reshape(
        n_queries, n_classes
    )
    return counts
