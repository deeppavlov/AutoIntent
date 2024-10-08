from typing import Any

import numpy as np
from numpy.typing import NDArray

from autointent.logger import get_logger

from .data_models import Artifact, Artifacts, RetrieverArtifact, ScorerArtifact, Trial, Trials, TrialsIds


class OptimizationInfo:
    """TODO continous IO with file system (to be able to restore the state of optimization)"""

    # todo how to set trials

    def __init__(self) -> None:
        self._logger = get_logger()

        self.artifacts = Artifacts()
        self.trials = Trials()
        self._trials_best_ids = TrialsIds()

    def log_module_optimization(
        self,
        node_type: str,
        module_type: str,
        module_params: dict[str, Any],
        metric_value: float,
        metric_name: str,
        artifact: Artifact,
    ) -> None:
        """
        Purposes:
        - save optimization results in a text form (hyperparameters and corresponding metrics)
        - update best assets
        """

        # save trial
        trial = Trial(
            module_type=module_type,
            metric_name=metric_name,
            metric_value=metric_value,
            module_params=module_params,
        )
        self.trials[node_type].append(trial)
        self._logger.info(trial.model_dump())

        # save artifact
        self.artifacts[node_type].append(artifact)

    def _get_metrics_values(self, node_type: str) -> list[float]:
        return [trial.metric_value for trial in self.trials[node_type]]

    def _get_best_trial_idx(self, node_type: str) -> int:
        res = self._trials_best_ids[node_type]
        if res is not None:
            return res
        self._trials_best_ids[node_type] = np.argmax(self._get_metrics_values(node_type))
        return self._trials_best_ids[node_type]

    def _get_best_artifact(self, node_type: str) -> ScorerArtifact:
        i_best = self._get_best_trial_idx(node_type)
        return self.artifacts[node_type][i_best]

    def get_best_embedder(self) -> str:
        best_retriever_artifact: RetrieverArtifact = self._get_best_artifact(node_type="retrieval")
        return best_retriever_artifact.embedder_name

    def get_best_test_scores(self) -> NDArray[np.float64] | None:
        best_scorer_artifact: ScorerArtifact = self._get_best_artifact(node_type="scoring")
        return best_scorer_artifact.test_scores

    def get_best_oos_scores(self) -> NDArray[np.float64] | None:
        best_scorer_artifact: ScorerArtifact = self._get_best_artifact(node_type="scoring")
        return best_scorer_artifact.oos_scores

    def dump(self) -> dict[str, dict[str, list[float]]]:
        node_wise_metrics = {
            node_type: self._get_metrics_values(node_type)
            for node_type in ["regexp", "retrieval", "scoring", "prediction"]
        }
        return {
            "metrics": node_wise_metrics,
            "configs": self.trials.model_dump(),
        }
