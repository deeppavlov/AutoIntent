from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier

from .base import ScoringModule
from ... import Context


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

    def __init__(self, multilabel: bool = False) -> None:
        self.multilabel = multilabel

    def fit(self, context: Context) -> None:
        collection = context.get_best_collection()
        dataset = collection.get(include=["embeddings", "metadatas"])
        features = dataset["embeddings"]

        labels = context.vector_index.metadata_as_labels(dataset["metadatas"])
        if self.multilabel:
            base_clf = LogisticRegression()
            clf = MultiOutputClassifier(base_clf)
        else:
            clf = LogisticRegressionCV(cv=3, n_jobs=8)

        clf.fit(features, labels)

        self._clf = clf
        self._emb_func = collection._embedding_function  # noqa: SLF001

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        features = self._emb_func(utterances)
        probas = self._clf.predict_proba(features)
        if self.multilabel:
            probas = np.stack(probas, axis=1)[..., 1]
        return probas  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        model = self._emb_func._model  # noqa: SLF001
        model.to(device="cpu")
        del model
        self.collection = None
