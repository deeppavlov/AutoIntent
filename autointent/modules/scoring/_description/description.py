"""DescriptionScorer class for scoring utterances based on intent descriptions."""

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import scipy
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity

from autointent import Context, Embedder
from autointent.custom_types import LabelType
from autointent.modules.abc import ScoringModule


class DescriptionScorerDumpMetadata(TypedDict):
    """Metadata for dumping the state of a DescriptionScorer."""

    n_classes: int
    multilabel: bool
    batch_size: int
    max_length: int | None


class DescriptionScorer(ScoringModule):
    r"""
    Scoring module that scores utterances based on similarity to intent descriptions.

    DescriptionScorer embeds both the utterances and the intent descriptions, then computes a similarity score
    between the two, using either cosine similarity and softmax.

    :ivar weights_file_name: Filename for saving the description vectors (`description_vectors.npy`).
    :ivar embedding_model_subdir: Directory for storing the embedder's model files.
    :ivar name: Name of the scorer, defaults to "description".

    """

    weights_file_name: str = "description_vectors.npy"
    embedding_model_subdir: str = "embedding_model"
    name = "description"

    def __init__(
        self,
        embedder_name: str,
        temperature: float = 1.0,
        embedder_device: str = "cpu",
        batch_size: int = 32,
        max_length: int | None = None,
        embedder_use_cache: bool = False,
    ) -> None:
        """
        Initialize the DescriptionScorer.

        :param embedder_name: Name of the embedder model.
        :param temperature: Temperature parameter for scaling logits, defaults to 1.0.
        :param embedder_device: Device to run the embedder on, e.g., "cpu" or "cuda".
        :param batch_size: Batch size for embedding generation, defaults to 32.
        :param max_length: Maximum sequence length for embedding, defaults to None.
        :param embedder_use_cache: Flag indicating whether to cache intermediate embeddings.
        """
        self.temperature = temperature
        self.embedder_device = embedder_device
        self.embedder_name = embedder_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedder_use_cache = embedder_use_cache

    @classmethod
    def from_context(
        cls,
        context: Context,
        temperature: float,
        embedder_name: str | None = None,
    ) -> "DescriptionScorer":
        """
        Create a DescriptionScorer instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param temperature: Temperature parameter for scaling logits.
        :param embedder_name: Name of the embedder model. If None, the best embedder is used.
        :return: Initialized DescriptionScorer instance.
        """
        return cls(
            temperature=temperature,
            embedder_device=context.get_device(),
            embedder_name=embedder_name if embedder_name is not None else context.optimization_info.get_best_embedder(),
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
        descriptions: list[str],
    ) -> None:
        """
        Fit the scorer by embedding utterances and descriptions.

        :param utterances: List of utterances to embed.
        :param labels: List of labels corresponding to the utterances.
        :param descriptions: List of intent descriptions.
        :raises ValueError: If descriptions contain None values or embeddings mismatch utterances.
        """
        if isinstance(labels[0], list):
            self.n_classes = len(labels[0])
            self.multilabel = True
        else:
            self.n_classes = len(set(labels))
            self.multilabel = False

        embedder = Embedder(
            device=self.embedder_device,
            model_name=self.embedder_name,
            batch_size=self.batch_size,
            max_length=self.max_length,
            use_cache=self.embedder_use_cache,
        )

        if any(description is None for description in descriptions):
            error_text = (
                "Some intent descriptions (label_description) are missing (None). "
                "Please ensure all intents have descriptions."
            )
            raise ValueError(error_text)

        self.description_vectors = embedder.embed([desc for desc in descriptions if desc])
        self._embedder = embedder

    def predict(self, utterances: list[str]) -> NDArray[np.float64]:
        """
        Predict scores for utterances based on similarity to intent descriptions.

        :param utterances: List of utterances to score.
        :return: Array of probabilities for each utterance.
        """
        utterance_vectors = self._embedder.embed(utterances)
        similarities: NDArray[np.float64] = cosine_similarity(utterance_vectors, self.description_vectors)

        if self.multilabel:
            probabilites = scipy.special.expit(similarities / self.temperature)
        else:
            probabilites = scipy.special.softmax(similarities / self.temperature, axis=1)
        return probabilites  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the embedder."""
        self._embedder.clear_ram()

    def dump(self, path: str) -> None:
        """
        Save the scorer's metadata, description vectors, and embedder state.

        :param path: Path to the directory where assets will be dumped.
        """
        self.metadata = DescriptionScorerDumpMetadata(
            n_classes=self.n_classes,
            multilabel=self.multilabel,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        dump_dir = Path(path)
        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

        np.save(dump_dir / self.weights_file_name, self.description_vectors)
        self._embedder.dump(dump_dir / self.embedding_model_subdir)

    def load(self, path: str) -> None:
        """
        Load the scorer's metadata, description vectors, and embedder state.

        :param path: Path to the directory containing the dumped assets.
        """
        dump_dir = Path(path)

        self.description_vectors = np.load(dump_dir / self.weights_file_name)

        with (dump_dir / self.metadata_dict_name).open() as file:
            self.metadata: DescriptionScorerDumpMetadata = json.load(file)

        self.n_classes = self.metadata["n_classes"]
        self.multilabel = self.metadata["multilabel"]

        embedder_dir = dump_dir / self.embedding_model_subdir
        self._embedder = Embedder(
            device=self.embedder_device,
            model_name=embedder_dir,
            batch_size=self.metadata["batch_size"],
            max_length=self.metadata["max_length"],
            use_cache=self.embedder_use_cache,
        )
