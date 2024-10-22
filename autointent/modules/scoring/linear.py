import json
from pathlib import Path
from typing import Any, TypedDict

import joblib
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from typing_extensions import Self

from autointent.context import Context
from autointent.context.data_handler import Tag
from autointent.context.vector_index_client import VectorIndexClient
from autointent.custom_types import LABEL_TYPE

from .base import ScoringModule


class LinearScorerDumpDict(TypedDict):
    multilabel: bool
    device: str


class LinearScorer(ScoringModule):
    """
    TODO:
    - implement different modes (incremental learning with SGD and simple learning with LogisticRegression)
    - control n_jobs
    - adjust cv
    - separate the sklearn fit() process and transformers tokenizers process (from vector_index embedding function)
        to avoid the warnings:
    ```
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling \
        parallelism to avoid deadlocks...
    To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    ```
    """

    classifier_file_name: str = "classifier.joblib"
    embedding_model_subdir: str = "embedding_model"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        db_dir: str = "vector_index",
        cv: int = 3,
        n_jobs: int = -1,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        self.cv = cv
        self.n_jobs = n_jobs
        self.device = device
        self.db_dir = db_dir
        self.seed = seed
        self.model_name = model_name

    @classmethod
    def from_context(
        cls,
        context: Context,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cv: int = 3,
        n_jobs: int = -1,
    ) -> Self:
        return cls(
            model_name=model_name,
            db_dir=context.db_dir,
            cv=cv,
            n_jobs=n_jobs,
            device=context.device,
            seed=context.seed,
        )

    def fit(self, utterances: list[str], labels: list[LABEL_TYPE], tags: list[Tag] | None = None) -> None:
        self._multilabel = isinstance(labels[0], list)

        vector_index_client = VectorIndexClient(self.device, self.db_dir)
        vector_index = vector_index_client.get_or_create_index(self.model_name)

        features = vector_index.embed(utterances) if vector_index.is_empty() else vector_index.get_all_embeddings()

        if self._multilabel:
            base_clf = LogisticRegression()
            clf = MultiOutputClassifier(base_clf)
        else:
            clf = LogisticRegressionCV(cv=self.cv, n_jobs=self.n_jobs, random_state=self.seed)

        clf.fit(features, labels)

        self._clf = clf
        self._embedder = vector_index.embedding_model

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        features = self._embedder.encode(utterances)
        probas = self._clf.predict_proba(features)
        if self._multilabel:
            probas = np.stack(probas, axis=1)[..., 1]
        return probas  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        self._embedder.cpu()
        del self._embedder

    def dump(self, path: str) -> None:
        metadata = LinearScorerDumpDict(
            multilabel=self._multilabel,
            device=str(self._embedder.device),
        )

        dump_dir = Path(path)

        metadata_path = dump_dir / self.metadata_dict_name
        with metadata_path.open("w") as file:
            json.dump(metadata, file, indent=4)

        # dump sklearn model
        clf_path = dump_dir / self.classifier_file_name
        joblib.dump(self._clf, clf_path)

        # dump sentence transformer model
        embedder_dir = str(dump_dir / self.embedding_model_subdir)
        self._embedder.save(embedder_dir)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        metadata_path = dump_dir / self.metadata_dict_name
        with metadata_path.open() as file:
            metadata: LinearScorerDumpDict = json.load(file)
        self._multilabel = metadata["multilabel"]

        # load sklearn model
        clf_path = dump_dir / self.classifier_file_name
        self._clf = joblib.load(clf_path)

        # load sentence transformer model
        embedder_dir = str(dump_dir / self.embedding_model_subdir)
        self._embedder = SentenceTransformer(embedder_dir, device=metadata["device"])
