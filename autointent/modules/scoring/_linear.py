"""LinearScorer class for linear classification."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier

from autointent import Context, Embedder
from autointent.custom_types import BaseMetadataDict, LabelType
from autointent.modules.abc import ScoringModule


class LinearScorerDumpDict(BaseMetadataDict):
    """
    Metadata for dumping the state of a LinearScorer.

    :ivar multilabel: Whether the task is multilabel classification.
    :ivar batch_size: Batch size used for embedding.
    :ivar max_length: Maximum sequence length for embedding, or None if not specified.
    """

    multilabel: bool
    batch_size: int
    max_length: int | None


class LinearScorer(ScoringModule):
    """
    Scoring module for linear classification using logistic regression.

    This module uses embeddings generated from a transformer model to train a
    logistic regression classifier for intent classification.

    :ivar classifier_file_name: Filename for saving the classifier to disk.
    :ivar embedding_model_subdir: Directory for saving the embedding model.
    :ivar name: Name of the scorer, defaults to "linear".

    Example
    --------
    .. testcode::

        from autointent.modules import LinearScorer
        scorer = LinearScorer(
            embedder_name="sergeyzh/rubert-tiny-turbo", cv=2
        )
        utterances = ["hello", "goodbye", "allo", "sayonara"]
        labels = [0, 1, 0, 1]
        scorer.fit(utterances, labels)
        test_utterances = ["hi", "bye"]
        probabilities = scorer.predict(test_utterances)
        print(probabilities)

    .. testoutput::

        [[0.50000032 0.49999968]
         [0.50000032 0.49999968]]

    """

    classifier_file_name: str = "classifier.joblib"
    embedding_model_subdir: str = "embedding_model"
    name = "linear"

    def __init__(
        self,
        embedder_name: str,
        cv: int = 3,
        n_jobs: int | None = None,
        embedder_device: str = "cpu",
        seed: int = 0,
        batch_size: int = 32,
        max_length: int | None = None,
        embedder_use_cache: bool = False,
    ) -> None:
        """
        Initialize the LinearScorer.

        :param embedder_name: Name of the embedder model.
        :param cv: Number of cross-validation folds, defaults to 3.
        :param n_jobs: Number of parallel jobs for cross-validation, defaults to -1 (all CPUs).
        :param embedder_device: Device to run operations on, e.g., "cpu" or "cuda".
        :param seed: Random seed for reproducibility, defaults to 0.
        :param batch_size: Batch size for embedding generation, defaults to 32.
        :param max_length: Maximum sequence length for embedding, or None for default.
        :param embedder_use_cache: Flag indicating whether to cache intermediate embeddings.
        """
        self.cv = cv
        self.n_jobs = n_jobs
        self.embedder_device = embedder_device
        self.seed = seed
        self.embedder_name = embedder_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedder_use_cache = embedder_use_cache

    @classmethod
    def from_context(
        cls,
        context: Context,
        embedder_name: str | None = None,
    ) -> "LinearScorer":
        """
        Create a LinearScorer instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param embedder_name: Name of the embedder, or None to use the best embedder.
        :return: Initialized LinearScorer instance.
        """
        return cls(
            embedder_name=embedder_name if embedder_name else context.optimization_info.get_best_embedder(),
            embedder_device=context.get_device(),
            seed=context.seed,
            batch_size=context.get_batch_size(),
            max_length=context.get_max_length(),
            embedder_use_cache=context.get_use_cache(),
        )

    def get_embedder_name(self) -> str:
        """
        Get the name of the embedder.

        :return: Embedder name.
        """
        return self.embedder_name

    def fit(
        self,
        utterances: list[str],
        labels: list[LabelType],
    ) -> None:
        """
        Train the logistic regression classifier.

        :param utterances: List of training utterances.
        :param labels: List of labels corresponding to the utterances.
        :raises ValueError: If the vector index mismatches the provided utterances.
        """
        self._multilabel = isinstance(labels[0], list)

        embedder = Embedder(
            device=self.embedder_device,
            model_name=self.embedder_name,
            batch_size=self.batch_size,
            max_length=self.max_length,
            use_cache=self.embedder_use_cache,
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
        """
        Predict probabilities for the given utterances.

        :param utterances: List of query utterances.
        :return: Array of predicted probabilities for each class.
        """
        features = self._embedder.embed(utterances)
        probas = self._clf.predict_proba(features)
        if self._multilabel:
            probas = np.stack(probas, axis=1)[..., 1]
        return probas  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the embedder."""
        self._embedder.clear_ram()

    def dump(self, path: str) -> None:
        """
        Save the LinearScorer's metadata, classifier, and embedder to disk.

        :param path: Path to the directory where assets will be dumped.
        """
        self.metadata = LinearScorerDumpDict(
            multilabel=self._multilabel,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        dump_dir = Path(path)

        metadata_path = dump_dir / self.metadata_dict_name
        with metadata_path.open("w") as file:
            json.dump(self.metadata, file, indent=4)

        # Dump sklearn model
        clf_path = dump_dir / self.classifier_file_name
        joblib.dump(self._clf, clf_path)

        # Dump sentence transformer model
        self._embedder.dump(dump_dir / self.embedding_model_subdir)

    def load(self, path: str) -> None:
        """
        Load the LinearScorer's metadata, classifier, and embedder from disk.

        :param path: Path to the directory containing the dumped assets.
        """
        dump_dir = Path(path)

        metadata_path = dump_dir / self.metadata_dict_name
        with metadata_path.open() as file:
            metadata: LinearScorerDumpDict = json.load(file)
        self._multilabel = metadata["multilabel"]

        # Load sklearn model
        clf_path = dump_dir / self.classifier_file_name
        self._clf = joblib.load(clf_path)

        # Load sentence transformer model
        embedder_dir = dump_dir / self.embedding_model_subdir
        self._embedder = Embedder(
            device=self.embedder_device,
            model_name=embedder_dir,
            batch_size=metadata["batch_size"],
            max_length=metadata["max_length"],
            use_cache=self.embedder_use_cache,
        )
