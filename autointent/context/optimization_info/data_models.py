from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from autointent.custom_types import NodeType


class Artifact(BaseModel): ...


class RegexpArtifact(Artifact): ...


class RetrieverArtifact(Artifact):
    """Name of the embedding model chosen after retrieval optimization."""

    embedder_name: str


class ScorerArtifact(Artifact):
    """Outputs from best scorer, numpy arrays of shape (n_samples, n_classes)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    test_scores: NDArray[np.float64] | None = Field(None, description="Scorer outputs for test utterances")
    oos_scores: NDArray[np.float64] | None = Field(None, description="Scorer outputs for out-of-scope utterances")


class PredictorArtifact(Artifact):
    """
    Outputs from best predictor, numpy array of shape (n_samples,) or
    (n_samples, n_classes) depending on classification mode (multi-class or multi-label).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    labels: NDArray[np.float64]


def validate_node_name(value: str) -> str:
    if value in [NodeType.retrieval, NodeType.scoring, NodeType.prediction, NodeType.regexp]:
        return value
    msg = f"Unknown node_type: {value}. Expected one of ['regexp', 'retrieval', 'scoring', 'prediction']"
    raise ValueError(msg)


class Artifacts(BaseModel):
    """Modules hyperparams and outputs. The best ones are transmitted between nodes of the pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    regexp: list[RegexpArtifact] = []
    retrieval: list[RetrieverArtifact] = []
    scoring: list[ScorerArtifact] = []
    prediction: list[PredictorArtifact] = []

    def add_artifact(self, node_type: str, artifact: Artifact) -> None:
        self.get_artifacts(node_type).append(artifact)

    def get_artifacts(self, node_type: str) -> list[Artifact]:
        return getattr(self, validate_node_name(node_type))  # type: ignore[no-any-return]

    def get_best_artifact(self, node_type: str, idx: int) -> Artifact:
        return self.get_artifacts(node_type)[idx]


class Trial(BaseModel):
    """Detailed representation of one optimization trial."""

    module_type: str
    module_params: dict[str, Any]
    metric_name: str
    metric_value: float
    module_dump_dir: str | None


class Trials(BaseModel):
    """Detailed representation of optimization results."""

    regexp: list[Trial] = []
    retrieval: list[Trial] = []
    scoring: list[Trial] = []
    prediction: list[Trial] = []

    def get_trial(self, node_type: str, idx: int) -> Trial:
        return self.get_trials(node_type)[idx]

    def get_trials(self, node_type: str) -> list[Trial]:
        return getattr(self, validate_node_name(node_type))  # type: ignore[no-any-return]

    def add_trial(self, node_type: str, trial: Trial) -> None:
        self.get_trials(node_type).append(trial)


class TrialsIds(BaseModel):
    """Detailed representation of optimization results."""

    regexp: int | None = None
    retrieval: int | None = None
    scoring: int | None = None
    prediction: int | None = None

    def get_best_trial_idx(self, node_type: str) -> int | None:
        return getattr(self, validate_node_name(node_type))  # type: ignore[no-any-return]

    def set_best_trial_idx(self, node_type: str, idx: int) -> None:
        setattr(self, validate_node_name(node_type), idx)
