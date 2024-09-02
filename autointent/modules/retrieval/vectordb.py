from functools import partial
from typing import Callable, Literal

import numpy as np
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
        labels_pred = retrieve_candidates(self.collection, self.k, data_handler.utterances_test)
        metric_value = metric_fn(data_handler.labels_test, labels_pred)
        return metric_value, self.model_name
    
    def clear_cache(self):
        model = self.collection._embedding_function._model
        model.to(device='cpu')
        del model
        self.collection = None


def retrieve_candidates(
        collection: Collection,
        k: int,
        utterances: list[str],
        weights: Literal["uniform", "distance", "closest"] | bool = False
    ) -> list[int] | list[list[int]]:
    """
    Return
    ---
    `labels`:
        - multiclass case: np.ndarray of shape (n_samples, n_candidates) with integer labels from `[0,n_classes-1]`
        - multilabel case: np.ndarray of shape (n_samples, n_candidates, n_classes) with binary labels
    `weights`:
        - multiclass case: np.ndarray of shape (n_samples, n_candidates)
        - multilabel case: np.ndarray of shape (n_samples, n_candidates, n_classes)
    """
    n_classes = collection.metadata["n_classes"]
    multilabel = collection.metadata["multilabel"]

    query_res = collection.query(
        query_texts=utterances,
        n_results=k,
        include=["metadatas", "documents", "distances"],  # one can add "embeddings"
    )

    if not multilabel:
        convert = multiclass_metadata_as_labels
    else:
        convert = partial(multilabel_metadata_as_labels, n_classes=n_classes)

    res_labels = np.array([convert(candidates) for candidates in query_res["metadatas"]])

    if not weights:
        return res_labels

    res_weights = get_weights(
        distances=np.array(query_res["distances"]),
        labels=res_labels,
        weights=weights,
        n_classes=n_classes,
        multilabel=multilabel
    )

    return res_labels, res_weights


def to_onehot(labels: np.ndarray, n_classes):
    """convert nd array of ints to (n+1)d array of zeros and ones"""
    new_shape = labels.shape+(n_classes,)
    onehot_labels = np.zeros(shape=new_shape)
    indices = tuple(np.indices(labels.shape)) + (labels,)
    onehot_labels[indices] = 1
    return onehot_labels


def closest_weighting(labels: np.ndarray, distances: np.ndarray):
    """
    TODO test this function

    Arguments
    ---
    `labels`: array of shape (n_samples, n_candidates, n_classes) with binary labels
    `distances`: array of shape (n_samples, n_candidates, n_classes) with float values
    
    Return
    ---
    array of shape (n_samples, n_candidates, n_classes), an entry is nonzero iff a candidate is the closest neighbor for each class
    """
    # broadcast to (n_samples, n_candidates, n_classes)
    broadcasted_distances = np.broadcast_to(distances[..., None], shape=labels.shape)
    expanded_distances_view = np.where(labels != 0, broadcasted_distances, np.inf)  
    
    # select min distance for each query-class pair
    min_distances = np.min(expanded_distances_view, axis=1, keepdims=True)
    res_weights = 1 / (expanded_distances_view + 1e-5) * (expanded_distances_view == min_distances)
    return res_weights


def get_weights(
        distances: np.ndarray,
        labels: np.ndarray,
        weights: Literal["uniform", "distance", "closest"] | bool,
        n_classes: int,
        multilabel: bool,
    ):
    """
    TODO test this function

    Arguments
    ---
    `distances`: array of shape (n_samples, n_candidates) with float values
    `labels`: needed only by "closest", 

    Return
    ---
    - multiclass case: np.ndarray of shape (n_samples, n_candidates)
    - multilabel case: np.ndarray of shape (n_samples, n_candidates, n_classes)
    """
    if isinstance(weights, bool) and weights:
        weights = "distance"

    n_samples, n_candidates = distances.shape

    if weights == "uniform":
        res = np.ones((n_samples, n_candidates))
        if multilabel:  # simply repeat the weights for each class
            res = np.broadcast_to(res[..., None], shape=(n_samples, n_candidates, n_classes))
    elif weights == "distance":
        res = 1 / (distances + 1e-5)
        if multilabel:  # simply repeat the weights for each class
            res = np.broadcast_to(res[..., None], shape=(n_samples, n_candidates, n_classes))
    elif weights == "closest":
        if not multilabel:
            onehot_labels = to_onehot(labels, n_classes)
            res = closest_weighting(onehot_labels, distances)
            res = np.sum(res, axis=2)   # there are only one non zero value
        else:
            res = closest_weighting(labels, distances)
    
    return res