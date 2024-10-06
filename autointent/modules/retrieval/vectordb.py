from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from chromadb import Collection

from autointent.context import Context
from autointent.context.optimization_info import RetrieverArtifact
from autointent.metrics import RetrievalMetricFn

from .base import RetrievalModule


class VectorDBModule(RetrievalModule):
    def __init__(self, k: int, model_name: str) -> None:
        self.model_name = model_name
        self.k = k

    def fit(self, context: Context) -> None:
        self.collection = context.vector_index.create_collection(self.model_name, context.data_handler)

    def score(self, context: Context, metric_fn: RetrievalMetricFn) -> tuple[float, npt.NDArray[Any]]:
        labels_pred = retrieve_candidates(
            self.collection,
            self.k,
            context.data_handler.utterances_test,
            converter=context.vector_index.metadata_as_labels,
        )
        return metric_fn(context.data_handler.labels_test, labels_pred), labels_pred

    def get_assets(self) -> RetrieverArtifact:
        return RetrieverArtifact(embedder_name=self.model_name)

    def clear_cache(self) -> None:
        model = self.collection._embedding_function._model  # noqa: SLF001
        model.to(device="cpu")
        del model
        self.collection = None


def retrieve_candidates(
    collection: Collection,
    k: int,
    utterances: list[str],
    converter: Callable[[Any], Any],
) -> npt.NDArray[Any]:
    """
    Return
    ---
    `labels`:
        - multiclass case: np.ndarray of shape (n_samples, n_candidates) with integer labels from `[0,n_classes-1]`
        - multilabel case: np.ndarray of shape (n_samples, n_candidates, n_classes) with binary labels
    """
    query_res = collection.query(
        query_texts=utterances,
        n_results=k,
        include=["metadatas", "documents", "distances"],  # one can add "embeddings"
    )

    return np.array([converter(candidates) for candidates in query_res["metadatas"]])
