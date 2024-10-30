import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import scipy
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity
from typing_extensions import Self

from autointent.context import Context
from autointent.context.vector_index_client import VectorIndex, VectorIndexClient
from autointent.context.vector_index_client.cache import get_db_dir
from autointent.custom_types import LABEL_TYPE
from autointent.modules.scoring.base import ScoringModule


class DescriptionScorerDumpMetadata(TypedDict):
    device: str
    db_dir: str
    n_classes: int
    multilabel: bool
    model_name: str


class DescriptionScorer(ScoringModule):
    weights_file_name: str = "description_vectors.npy"
    vector_index: VectorIndex
    prebuilt_index: bool = False

    def __init__(
        self,
        model_name: str,
        db_dir: Path | None = None,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        if db_dir is None:
            db_dir = get_db_dir()
        self.temperature = temperature
        self.device = device
        self.db_dir = db_dir
        self.model_name = model_name

    @classmethod
    def from_context(
        cls,
        context: Context,
        temperature: float,
        model_name: str | None = None,
    ) -> Self:
        if model_name is None:
            model_name = context.optimization_info.get_best_embedder()
            prebuilt_index = True
        else:
            prebuilt_index = context.vector_index_client.exists(model_name)

        instance = cls(
            temperature=temperature,
            device=context.device,
            db_dir=context.db_dir,
            model_name=model_name,
        )
        instance.prebuilt_index = prebuilt_index
        return instance

    def fit(
        self,
        utterances: list[str],
        labels: list[LABEL_TYPE],
        descriptions: list[str | None] | None = None,
    ) -> None:
        if descriptions is None:
            msg = "Descriptions are required for training."
            raise ValueError(msg)

        if isinstance(labels[0], list):
            self.n_classes = len(labels[0])
            self.multilabel = True
        else:
            self.n_classes = len(set(labels))
            self.multilabel = False

        vector_index_client = VectorIndexClient(self.device, self.db_dir)

        if self.prebuilt_index:
            # this happens only when LinearScorer is within Pipeline opimization after RetrievalNode optimization
            self.vector_index = vector_index_client.get_index(self.model_name)
            if len(utterances) != len(self.vector_index.texts):
                msg = "Vector index mismatches provided utterances"
                raise ValueError(msg)
        else:
            self.vector_index = vector_index_client.create_index(self.model_name, utterances, labels)

        if any(description is None for description in descriptions):
            error_text = (
                "Some intent descriptions (label_description) are missing (None). "
                "Please ensure all intents have descriptions."
            )
            raise ValueError(error_text)

        self.description_vectors = self.vector_index.embedder.embed([desc for desc in descriptions if desc is not None])

        self.metadata = DescriptionScorerDumpMetadata(
            device=self.device,
            db_dir=str(self.db_dir),
            n_classes=self.n_classes,
            multilabel=self.multilabel,
            model_name=self.vector_index.model_name,
        )

    def predict(self, utterances: list[str]) -> NDArray[np.float64]:
        utterance_vectors = self.vector_index.embedder.embed(utterances)
        similarities: NDArray[np.float64] = cosine_similarity(utterance_vectors, self.description_vectors)

        if self.multilabel:
            probabilites = scipy.special.expit(similarities / self.temperature)
        else:
            probabilites = scipy.special.softmax(similarities / self.temperature, axis=1)
        return probabilites  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        self.vector_index.delete()

    def dump(self, path: str) -> None:
        dump_dir = Path(path)
        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

        np.save(dump_dir / self.weights_file_name, self.description_vectors)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        self.description_vectors = np.load(dump_dir / self.weights_file_name)

        with (dump_dir / self.metadata_dict_name).open() as file:
            self.metadata: DescriptionScorerDumpMetadata = json.load(file)

        self.n_classes = self.metadata["n_classes"]
        self.multilabel = self.metadata["multilabel"]

        vector_index_client = VectorIndexClient(device=self.metadata["device"], db_dir=self.metadata["db_dir"])
        self.vector_index = vector_index_client.get_index(self.metadata["model_name"])
