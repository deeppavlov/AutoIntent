from autointent.context import Context
from autointent.context.optimization_info import RetrieverArtifact
from autointent.metrics import RetrievalMetricFn

from .base import RetrievalModule


class VectorDBModule(RetrievalModule):
    def __init__(self, k: int, model_name: str) -> None:
        self.model_name = model_name
        self.k = k

    def fit(self, context: Context) -> None:
        self.collection_name = context.vector_index.create_index(self.model_name, context.data_handler)

    def score(self, context: Context, metric_fn: RetrievalMetricFn) -> float:
        labels_pred, _ = context.vector_index.query(
            self.collection_name,
            context.data_handler.utterances_test,
            self.k,
            context.vector_index.metadata_as_labels,
        )
        return metric_fn(context.data_handler.labels_test, labels_pred)

    def get_assets(self) -> RetrieverArtifact:
        return RetrieverArtifact(embedder_name=self.model_name)

    def clear_cache(self) -> None:
        pass
