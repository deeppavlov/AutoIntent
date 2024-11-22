"""LinearScorer class for linear classification."""

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
    :ivar precomputed_embeddings: Flag indicating if embeddings are precomputed.
    :ivar db_dir: Path to the database directory.
    :ivar name: Name of the scorer, defaults to "linear".
    """

    # TODO:
    # - implement different modes (incremental learning with SGD and simple learning with LogisticRegression)
    # - control n_jobs
    # - adjust cv
    # - separate the sklearn fit() process and transformers tokenizers process (from vector_index embedding function)
    #     to avoid the warnings:
    # ```
    # huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling \
    #     parallelism to avoid deadlocks...
    # To disable this warning, you can either:
    #     - Avoid using `tokenizers` before the fork if possible
    #     - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    # ```

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
        """
        Initialize the LinearScorer.

        :param embedder_name: Name of the embedder model.
        :param cv: Number of cross-validation folds, defaults to 3.
        :param n_jobs: Number of parallel jobs for cross-validation, defaults to -1 (all CPUs).
        :param device: Device to run operations on, e.g., "cpu" or "cuda".
        :param seed: Random seed for reproducibility, defaults to 0.
        :param batch_size: Batch size for embedding generation, defaults to 32.
        :param max_length: Maximum sequence length for embedding, or None for default.
        """
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
        """
        Create a LinearScorer instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param embedder_name: Name of the embedder, or None to use the best embedder.
        :return: Initialized LinearScorer instance.
        """
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
            device=self.device,
            model_name=embedder_dir,
            batch_size=metadata["batch_size"],
            max_length=metadata["max_length"],
        )
