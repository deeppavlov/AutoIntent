import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from typing_extensions import Self

from autointent.context import Context
from autointent.context.embedder import Embedder
from autointent.context.vector_index_client import VectorIndexClient
from autointent.custom_types import BaseMetadataDict, LabelType

from .base import ScoringModule


class LinearScorerDumpDict(BaseMetadataDict):
    multilabel: bool
    batch_size: int
    max_length: int | None


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
    ```.
    """

    classifier_file_name: str = "classifier.joblib"
    embedding_model_subdir: str = "embedding_model"
    precomputed_embeddings: bool = False
    db_dir: str
    name = "linear"

    def __init__(
        self,
        embedder_name: str,
        cv: int = 3,
        n_jobs: int = -1,
        device: str = "cpu",
        seed: int = 0,
        batch_size: int = 32,
        max_length: int | None = None,
    ) -> None:
        self.cv = cv
        self.n_jobs = n_jobs
        self.device = device
        self.seed = seed
        self.embedder_name = embedder_name
        self.batch_size = batch_size
        self.max_length = max_length

    @classmethod
    def from_context(
        cls,
        context: Context,
        embedder_name: str | None = None,
    ) -> Self:
        if embedder_name is None:
            embedder_name = context.optimization_info.get_best_embedder()
            precomputed_embeddings = True
        else:
            precomputed_embeddings = context.vector_index_client.exists(embedder_name)

        instance = cls(
            embedder_name=embedder_name,
            device=context.get_device(),
            seed=context.seed,
            batch_size=context.get_batch_size(),
            max_length=context.get_max_length(),
        )
        instance.precomputed_embeddings = precomputed_embeddings
        instance.db_dir = str(context.get_db_dir())
        return instance

    def get_embedder_name(self) -> str:
        return self.embedder_name

    def fit(
        self,
        utterances: list[str],
        labels: list[LabelType],
    ) -> None:
        self._multilabel = isinstance(labels[0], list)

        if self.precomputed_embeddings:
            # this happens only when LinearScorer is within Pipeline opimization after RetrievalNode optimization
            vector_index_client = VectorIndexClient(self.device, self.db_dir, self.batch_size, self.max_length)
            vector_index = vector_index_client.get_index(self.embedder_name)
            features = vector_index.get_all_embeddings()
            if len(features) != len(utterances):
                msg = "Vector index mismatches provided utterances"
                raise ValueError(msg)
            embedder = vector_index.embedder
        else:
            embedder = Embedder(
                device=self.device,
                model_name=self.embedder_name,
                batch_size=self.batch_size,
                max_length=self.max_length,
            )
            features = embedder.embed(utterances)

        if self._multilabel:
            base_clf = LogisticRegression()
            clf = MultiOutputClassifier(base_clf)
        else:
            clf = LogisticRegressionCV(cv=self.cv, n_jobs=self.n_jobs, random_state=self.seed)

        clf.fit(features, labels)

        self._clf = clf
        self._embedder = embedder

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        features = self._embedder.embed(utterances)
        probas = self._clf.predict_proba(features)
        if self._multilabel:
            probas = np.stack(probas, axis=1)[..., 1]
        return probas  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        self._embedder.clear_ram()

    def dump(self, path: str) -> None:
        self.metadata = LinearScorerDumpDict(
            multilabel=self._multilabel,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        dump_dir = Path(path)

        metadata_path = dump_dir / self.metadata_dict_name
        with metadata_path.open("w") as file:
            json.dump(self.metadata, file, indent=4)

        # dump sklearn model
        clf_path = dump_dir / self.classifier_file_name
        joblib.dump(self._clf, clf_path)

        # dump sentence transformer model
        self._embedder.dump(dump_dir / self.embedding_model_subdir)

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
        embedder_dir = dump_dir / self.embedding_model_subdir
        self._embedder = Embedder(
            device=self.device,
            model_name=embedder_dir,
            batch_size=metadata["batch_size"],
            max_length=metadata["max_length"],
        )
