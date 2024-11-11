from typing import Any

import numpy as np
from numpy.typing import NDArray

from autointent.configs.node import InferenceNodeConfig
from autointent.custom_types import NODE_TYPES, NodeType
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
        module_dump_dir: str,
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
            module_dump_dir=module_dump_dir,
        )
        self.trials.add_trial(node_type, trial)
        self._logger.info(trial.model_dump())

        # save artifact
        self.artifacts.add_artifact(node_type, artifact)

    def _get_metrics_values(self, node_type: str) -> list[float]:
        return [trial.metric_value for trial in self.trials.get_trials(node_type)]

    def _get_best_trial_idx(self, node_type: str) -> int | None:
        if len(self.trials.get_trials(node_type)) == 0:
            return None
        res = self._trials_best_ids.get_best_trial_idx(node_type)
        if res is not None:
            return res
        self._trials_best_ids.set_best_trial_idx(node_type, int(np.argmax(self._get_metrics_values(node_type))))
        return self._trials_best_ids.get_best_trial_idx(node_type)

    def _get_best_artifact(self, node_type: str) -> RetrieverArtifact | ScorerArtifact | Artifact:
        i_best = self._get_best_trial_idx(node_type)
        if i_best is None:
            msg = f"No best trial for {node_type}"
            raise ValueError(msg)
        return self.artifacts.get_best_artifact(node_type, i_best)

    def get_best_embedder(self) -> str:
        best_retriever_artifact: RetrieverArtifact = self._get_best_artifact(node_type=NodeType.retrieval)  # type: ignore[assignment]
        return best_retriever_artifact.embedder_name

    def get_best_test_scores(self) -> NDArray[np.float64] | None:
        best_scorer_artifact: ScorerArtifact = self._get_best_artifact(node_type=NodeType.scoring)  # type: ignore[assignment]
        return best_scorer_artifact.test_scores

    def get_best_oos_scores(self) -> NDArray[np.float64] | None:
        best_scorer_artifact: ScorerArtifact = self._get_best_artifact(node_type=NodeType.scoring)  # type: ignore[assignment]
        return best_scorer_artifact.oos_scores

    def dump_evaluation_results(self) -> dict[str, dict[str, list[float]]]:
        node_wise_metrics = {node_type.value: self._get_metrics_values(node_type) for node_type in NODE_TYPES}
        return {
            "metrics": node_wise_metrics,
            "configs": self.trials.model_dump(),
        }

    def get_inference_nodes_config(self) -> list[InferenceNodeConfig]:
        trial_ids = [self._get_best_trial_idx(node_type) for node_type in NODE_TYPES]
        res = []
        for idx, node_type in zip(trial_ids, NODE_TYPES, strict=True):
            if idx is None:
                continue
            trial = self.trials.get_trial(node_type, idx)
            res.append(
                InferenceNodeConfig(
                    node_type=node_type,
                    module_type=trial.module_type,
                    module_config=trial.module_params,
                    load_path=trial.module_dump_dir,
                )
            )
        return res
