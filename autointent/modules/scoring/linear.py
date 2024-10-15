import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier

from autointent import Context

from .base import ScoringModule


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

    def __init__(self, cv: int = 3, n_jobs: int = -1) -> None:
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, context: Context) -> None:
        self._multilabel = context.multilabel
        vector_index = context.get_best_index()
        features = vector_index.get_all_embeddings()
        labels = vector_index.get_all_labels()

        if self._multilabel:
            base_clf = LogisticRegression()
            clf = MultiOutputClassifier(base_clf)
        else:
            clf = LogisticRegressionCV(cv=self.cv, n_jobs=self.n_jobs, random_state=context.seed)

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
        pass

    def dump(self, path: str) -> None:
        metadata = {
            "multilabel": self._multilabel,
            "device": str(self._embedder.device),
        }

        dump_dir = Path(path)

        metadata_path = dump_dir / "metadata.json"
        with metadata_path.open("w") as file:
            json.dump(metadata, file, indent=4)

        # dump sklearn model
        clf_path = dump_dir / "classifier.joblib"
        joblib.dump(self._clf, clf_path)

        # dump sentence transformer model
        embedder_dir = str(dump_dir / "embedding_model")
        self._embedder.save(embedder_dir)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        metadata_path = dump_dir / "metadata.json"
        with metadata_path.open() as file:
            metadata = json.load(file)
        self._multilabel = metadata["multilabel"]

        # load sklearn model
        clf_path = dump_dir / "classifier.joblib"
        self._clf = joblib.load(clf_path)

        # load sentence transformer model
        embedder_dir = str(dump_dir / "embedding_model")
        self._embedder = SentenceTransformer(embedder_dir, device=metadata["device"])
