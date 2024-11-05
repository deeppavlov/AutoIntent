import json
from pathlib import Path

from autointent.context import Context
from autointent.context.optimization_info import RetrieverArtifact
from autointent.context.vector_index_client import VectorIndex, VectorIndexClient
from autointent.metrics import RetrievalMetricFn

from .base import RetrievalModule


class VectorDBModule(RetrievalModule):
    vector_index: VectorIndex
    name = "vector_db"

    def __init__(self, k: int, model_name: str) -> None:
        self.model_name = model_name
        self.k = k

    def fit(self, context: Context) -> None:
        self.vector_index_client_kwargs = {
            "device": context.device,
            "db_dir": str(context.db_dir),
            "embedder_batch_size": context.embedder_batch_size,
        }

        self.vector_index = context.vector_index_client.create_index(self.model_name, context.data_handler)

    def score(self, context: Context, metric_fn: RetrievalMetricFn) -> float:
        labels_pred, _, _ = self.vector_index.query(
            context.data_handler.utterances_test,
            self.k,
        )
        return metric_fn(context.data_handler.labels_test, labels_pred)

    def get_assets(self) -> RetrieverArtifact:
        return RetrieverArtifact(embedder_name=self.model_name)

    def clear_cache(self) -> None:
        self.vector_index.delete()

    def dump(self, path: str) -> None:
        dump_dir = Path(path)
        with (dump_dir / "vector_index_client_kwargs.json").open("w") as file:
            json.dump(self.vector_index_client_kwargs, file, indent=4)

    def load(self, path: str) -> None:
        dump_dir = Path(path)
        with (dump_dir / "vector_index_client_kwargs.json").open() as file:
            self.vector_index_client_kwargs = json.load(file)

        vector_index_client = VectorIndexClient(**self.vector_index_client_kwargs)  # type: ignore[arg-type]
        self.vector_index = vector_index_client.get_index(self.model_name)

    def predict(self, utterances: list[str]) -> tuple[list[list[int | list[int]]], list[list[float]], list[list[str]]]:
        """
        return labels, distances and texts of retrieved nearest neighbors
        """
        return self.vector_index.query(
            utterances,
            self.k,
        )
