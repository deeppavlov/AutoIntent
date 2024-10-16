from typing import Any

import numpy as np
import numpy.typing as npt
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
        self._emb_func = vector_index.embed

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        features = self._emb_func(utterances)
        probas = self._clf.predict_proba(features)
        if self._multilabel:
            probas = np.stack(probas, axis=1)[..., 1]
        return probas  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        pass
