from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class Artifact(BaseModel): ...


class RegexpArtifact(Artifact): ...


class RetrieverArtifact(Artifact):
    """
    Name of the embedding model chosen after retrieval optimization
    """

    embedder_name: str


class ScorerArtifact(Artifact):
    """
    Outputs from best scorer, numpy arrays of shape (n_samples, n_classes)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    test_scores: NDArray[np.float64] | None = Field(None, description="Scorer outputs for test utterances")
    oos_scores: NDArray[np.float64] | None = Field(None, description="Scorer outputs for out-of-scope utterances")


class PredictorArtifact(Artifact):
    """
    Outputs from best predictor, numpy array of shape (n_samples,) or
    (n_samples, n_classes) depending on classification mode (multi-class or multi-label)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    labels: NDArray[np.float64]


class Artifacts(BaseModel):
    """
    Modules hyperparams and outputs. The best ones are transmitted between nodes of the pipeline
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    regexp: list[RegexpArtifact] = []
    retrieval: list[RetrieverArtifact] = []
    scoring: list[ScorerArtifact] = []
    prediction: list[PredictorArtifact] = []

    def __getitem__(self, node_type: str) -> list:
        return getattr(self, node_type)


class Trial(BaseModel):
    """
    Detailed representation of one optimization trial
    """

    module_type: str
    module_params: dict[str, Any]
    metric_name: str
    metric_value: float


class Trials(BaseModel):
    """
    Detailed representation of optimization results
    """

    regexp: list[Trial] = []
    retrieval: list[Trial] = []
    scoring: list[Trial] = []
    prediction: list[Trial] = []

    def __getitem__(self, node_type: str) -> list[Trial]:
        return getattr(self, node_type)


class TrialsIds(BaseModel):
    """
    Detailed representation of optimization results
    """

    regexp: int | None = None
    retrieval: int | None = None
    scoring: int | None = None
    prediction: int | None = None

    def __getitem__(self, node_type: str) -> list[Trial]:
        return getattr(self, node_type)

    def __setitem__(self, node_type: str, idx: int) -> None:
        setattr(self, node_type, idx)
