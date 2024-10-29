import json
from pathlib import Path
from typing import Any

from typing_extensions import Self

from autointent.context import Context
from autointent.context.optimization_info import RetrieverArtifact
from autointent.context.vector_index_client import VectorIndex, VectorIndexClient
from autointent.custom_types import LABEL_TYPE
from autointent.metrics import RetrievalMetricFn
from autointent.modules.base import BaseMetadataDict

from .base import RetrievalModule


class VectorDBMetadata(BaseMetadataDict):
    k: int
    model_name: str
    device: str
    db_dir: str


class VectorDBModule(RetrievalModule):
    vector_index: VectorIndex

    def __init__(self, k: int, model_name: str, db_dir: str, device: str = "cpu", embedding_batch_size: int = 1) -> None:
        self.model_name = model_name
        self.device = device
        self.db_dir = db_dir
        self.embedding_batch_size = embedding_batch_size

        self.metadata = VectorDBMetadata(
            k=k,
            model_name=model_name,
            device=device,
            db_dir=db_dir,
        )

        super().__init__(k=k)

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: int = 5,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs: dict[str, Any],
    ) -> Self:
        return cls(
            k=k,
            model_name=model_name,
            db_dir=context.db_dir,
            device=context.device,
            embedding_batch_size=context.embedder_batch_size,
        )

    def fit(self, utterances: list[str], labels: list[LABEL_TYPE], **kwargs: dict[str, Any]) -> None:
        self.vector_index_client_kwargs = {
            "device": self.device,
            "db_dir": str(self.db_dir),
            "embedder_batch_size": self.embedder_batch_size,
        }
        vector_index_client = VectorIndexClient(self.device, self.db_dir)

        self.vector_index = vector_index_client.create_index(self.model_name, utterances, labels)
        self.vector_index.dump(Path(self.db_dir))

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
            json.dump(self.metadata, file, indent=4)
        self.vector_index.dump(dump_dir)

    def load(self, path: str) -> None:
        dump_dir = Path(path)
        with (dump_dir / "vector_index_client_kwargs.json").open() as file:
            self.metadata: VectorDBMetadata = json.load(file)

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
