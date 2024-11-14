import json
from pathlib import Path

from typing_extensions import Self

from autointent.context import Context
from autointent.context.optimization_info import RetrieverArtifact
from autointent.context.vector_index_client import VectorIndex, VectorIndexClient
from autointent.context.vector_index_client.cache import get_db_dir
from autointent.custom_types import BaseMetadataDict, LabelType
from autointent.metrics import RetrievalMetricFn

from .base import RetrievalModule


class VectorDBMetadata(BaseMetadataDict):
    db_dir: str
    batch_size: int
    max_length: int | None


class VectorDBModule(RetrievalModule):
    vector_index: VectorIndex
    name = "vector_db"

    def __init__(
        self,
        k: int,
        embedder_name: str,
        db_dir: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int | None = None,
    ) -> None:
        self.embedder_name = embedder_name
        self.device = device
        self._db_dir = db_dir
        self.batch_size = batch_size
        self.max_length = max_length

        super().__init__(k=k)

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: int,
        embedder_name: str,
    ) -> Self:
        return cls(
            k=k,
            embedder_name=embedder_name,
            db_dir=str(context.get_db_dir()),
            device=context.get_device(),
            batch_size=context.get_batch_size(),
            max_length=context.get_max_length(),
        )

    @property
    def db_dir(self) -> str:
        if self._db_dir is None:
            self._db_dir = str(get_db_dir())
        return self._db_dir

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        vector_index_client = VectorIndexClient(
            self.device, self.db_dir, embedder_batch_size=self.batch_size, embedder_max_length=self.max_length
        )

        self.vector_index = vector_index_client.create_index(self.embedder_name, utterances, labels)

    def score(self, context: Context, metric_fn: RetrievalMetricFn) -> float:
        labels_pred, _, _ = self.vector_index.query(
            context.data_handler.test_utterances,
            self.k,
        )
        return metric_fn(context.data_handler.test_labels, labels_pred)

    def get_assets(self) -> RetrieverArtifact:
        return RetrieverArtifact(embedder_name=self.embedder_name)

    def clear_cache(self) -> None:
        self.vector_index.clear_ram()

    def dump(self, path: str) -> None:
        self.metadata = VectorDBMetadata(
            batch_size=self.batch_size,
            max_length=self.max_length,
            db_dir=self.db_dir,
        )

        dump_dir = Path(path)
        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)
        self.vector_index.dump(dump_dir)

    def load(self, path: str) -> None:
        dump_dir = Path(path)
        with (dump_dir / self.metadata_dict_name).open() as file:
            self.metadata: VectorDBMetadata = json.load(file)

        vector_index_client = VectorIndexClient(
            device=self.device,
            db_dir=self.metadata["db_dir"],
            embedder_batch_size=self.metadata["batch_size"],
            embedder_max_length=self.metadata["max_length"],
        )
        self.vector_index = vector_index_client.get_index(self.embedder_name)

    def predict(self, utterances: list[str]) -> tuple[list[list[int | list[int]]], list[list[float]], list[list[str]]]:
        """
        return labels, distances and texts of retrieved nearest neighbors
        """
        return self.vector_index.query(
            utterances,
            self.k,
        )
