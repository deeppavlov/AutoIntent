"""Module for artifact and trial management in the pipeline.

This module defines data models for managing artifacts and trials in the pipeline,
including their configurations, outputs, and optimization details.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from autointent.custom_types import NodeType


class Artifact(BaseModel):
    """Base class for artifacts generated by pipeline nodes."""


class RegexpArtifact(Artifact):
    """Artifact containing results from the regexp node."""


class RetrieverArtifact(Artifact):
    """Artifact containing details from the retrieval node."""

    embedder_name: str


class ScorerArtifact(Artifact):
    """Artifact containing outputs from the scoring node."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    test_scores: NDArray[np.float64] | None = Field(None, description="Scorer outputs for test utterances")
    oos_scores: NDArray[np.float64] | None = Field(None, description="Scorer outputs for out-of-scope utterances")


class PredictorArtifact(Artifact):
    """Artifact containing outputs from the predictor node."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    labels: NDArray[np.float64]


def validate_node_name(value: str) -> str:
    """
    Validate and return the node type.

    :param value: Node type as a string.
    :return: Validated node type string.
    :raises ValueError: If the node type is invalid.
    """
    if value in [NodeType.retrieval, NodeType.scoring, NodeType.prediction, NodeType.regexp]:
        return value
    msg = f"Unknown node_type: {value}. Expected one of ['regexp', 'retrieval', 'scoring', 'prediction']"
    raise ValueError(msg)


class Artifacts(BaseModel):
    """Container for storing and managing artifacts generated by pipeline nodes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    regexp: list[RegexpArtifact] = []
    retrieval: list[RetrieverArtifact] = []
    scoring: list[ScorerArtifact] = []
    prediction: list[PredictorArtifact] = []

    def add_artifact(self, node_type: str, artifact: Artifact) -> None:
        """
        Add an artifact to the specified node type.

        :param node_type: Node type as a string.
        :param artifact: The artifact to add.
        """
        self.get_artifacts(node_type).append(artifact)

    def get_artifacts(self, node_type: str) -> list[Artifact]:
        """
        Retrieve all artifacts for a specified node type.

        :param node_type: Node type as a string.
        :return: A list of artifacts for the node type.
        """
        return getattr(self, validate_node_name(node_type))  # type: ignore[no-any-return]

    def get_best_artifact(self, node_type: str, idx: int) -> Artifact:
        """
        Retrieve the best artifact for a specified node type and index.

        :param node_type: Node type as a string.
        :param idx: Index of the artifact.
        :return: The best artifact.
        """
        return self.get_artifacts(node_type)[idx]


class Trial(BaseModel):
    """Representation of an individual optimization trial."""

    module_type: str
    """Type of the module being optimized."""
    module_params: dict[str, Any]
    """Parameters of the module for the trial."""
    metric_name: str
    """Name of the evaluation metric."""
    metric_value: float
    """Value of the evaluation metric for this trial."""
    module_dump_dir: str | None
    """Directory where the module is dumped."""


class Trials(BaseModel):
    """Container for managing optimization trials for pipeline nodes."""

    regexp: list[Trial] = []
    retrieval: list[Trial] = []
    scoring: list[Trial] = []
    prediction: list[Trial] = []

    def get_trial(self, node_type: str, idx: int) -> Trial:
        """
        Retrieve a specific trial for a node type and index.

        :param node_type: Node type as a string.
        :param idx: Index of the trial.
        :return: The requested trial.
        """
        return self.get_trials(node_type)[idx]

    def get_trials(self, node_type: str) -> list[Trial]:
        """
        Retrieve all trials for a specified node type.

        :param node_type: Node type as a string.
        :return: A list of trials for the node type.
        """
        return getattr(self, validate_node_name(node_type))  # type: ignore[no-any-return]

    def add_trial(self, node_type: str, trial: Trial) -> None:
        """
        Add a trial to a specified node type.

        :param node_type: Node type as a string.
        :param trial: The trial to add.
        """
        self.get_trials(node_type).append(trial)


class TrialsIds(BaseModel):
    """Representation of the best trial IDs for each pipeline node."""

    regexp: int | None = None
    """Best trial index for the regexp node."""
    retrieval: int | None = None
    """Best trial index for the retrieval node."""
    scoring: int | None = None
    """Best trial index for the scoring"""
    prediction: int | None = None
    """Best trial index for the prediction node."""

    def get_best_trial_idx(self, node_type: str) -> int | None:
        """
        Retrieve the best trial index for a specified node type.

        :param node_type: Node type as a string.
        :return: The index of the best trial, or None if not set.
        """
        return getattr(self, validate_node_name(node_type))  # type: ignore[no-any-return]

    def set_best_trial_idx(self, node_type: str, idx: int) -> None:
        """
        Set the best trial index for a specified node type.

        :param node_type: Node type as a string.
        :param idx: Index of the best trial.
        """
        setattr(self, validate_node_name(node_type), idx)
