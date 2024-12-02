"""Module for managing pipeline optimization.

This module handles the tracking, logging, and retrieval of optimization artifacts,
trials, and modules during the pipeline's execution.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from autointent.configs import InferenceNodeConfig
from autointent.custom_types import NodeType

from ._data_models import Artifact, Artifacts, RetrieverArtifact, ScorerArtifact, Trial, Trials, TrialsIds
from ._logger import get_logger

if TYPE_CHECKING:
    from autointent.modules import Module


@dataclass
class ModulesList:
    """Container for managing lists of modules for each node type."""

    regexp: list["Module"] = field(default_factory=list)
    retrieval: list["Module"] = field(default_factory=list)
    scoring: list["Module"] = field(default_factory=list)
    prediction: list["Module"] = field(default_factory=list)

    def get(self, node_type: str) -> list["Module"]:
        """
        Retrieve the list of modules for a specific node type.

        :param node_type: The type of node (e.g., "regexp", "retrieval").
        :return: List of modules for the specified node type.
        """
        return getattr(self, node_type)  # type: ignore[no-any-return]

    def add_module(self, node_type: str, module: "Module") -> None:
        """
        Add a module to the list for a specific node type.

        :param node_type: The type of node.
        :param module: The module to add.
        """
        self.get(node_type).append(module)


class OptimizationInfo:
    """
    Tracks optimization results, including trials, artifacts, and modules.

    This class provides methods for logging optimization results, retrieving
    the best-performing modules and artifacts, and generating configuration
    for inference nodes.
    """

    def __init__(self) -> None:
        """Initialize optimization info."""
        self._logger = get_logger()

        self.artifacts = Artifacts()
        self.trials = Trials()
        self._trials_best_ids = TrialsIds()
        self.modules = ModulesList()

    def log_module_optimization(
        self,
        node_type: str,
        module_type: str,
        module_params: dict[str, Any],
        metric_value: float,
        metric_name: str,
        artifact: Artifact,
        module_dump_dir: str | None,
        module: "Module | None" = None,
    ) -> None:
        """
        Log optimization results for a module.

        :param node_type: Type of the node being optimized.
        :param module_type: Type of the module.
        :param module_params: Parameters of the module for the trial.
        :param metric_value: Metric value achieved by the module.
        :param metric_name: Name of the evaluation metric.
        :param artifact: Artifact generated by the module.
        :param module_dump_dir: Directory where the module is dumped.
        :param module: The module instance, if available.
        """
        trial = Trial(
            module_type=module_type,
            metric_name=metric_name,
            metric_value=metric_value,
            module_params=module_params,
            module_dump_dir=module_dump_dir,
        )
        self.trials.add_trial(node_type, trial)
        self._logger.info(trial.model_dump())

        if module:
            self.modules.add_module(node_type, module)

        self.artifacts.add_artifact(node_type, artifact)

    def _get_metrics_values(self, node_type: str) -> list[float]:
        """Retrieve all metric values for a specific node type."""
        return [trial.metric_value for trial in self.trials.get_trials(node_type)]

    def _get_best_trial_idx(self, node_type: str) -> int | None:
        """
        Retrieve the index of the best trial for a node type.

        :param node_type: Type of the node.
        :return: Index of the best trial, or None if no trials exist.
        """
        if not self.trials.get_trials(node_type):
            return None
        best_idx = self._trials_best_ids.get_best_trial_idx(node_type)
        if best_idx is not None:
            return best_idx
        best_idx = int(np.argmax(self._get_metrics_values(node_type)))
        self._trials_best_ids.set_best_trial_idx(node_type, best_idx)
        return best_idx

    def _get_best_artifact(self, node_type: str) -> RetrieverArtifact | ScorerArtifact | Artifact:
        """
        Retrieve the best artifact for a specific node type.

        :param node_type: Type of the node.
        :return: The best artifact for the node type.
        :raises ValueError: If no best trial exists for the node type.
        """
        best_idx = self._get_best_trial_idx(node_type)
        if best_idx is None:
            msg = f"No best trial for {node_type}"
            raise ValueError(msg)
        return self.artifacts.get_best_artifact(node_type, best_idx)

    def get_best_embedder(self) -> str:
        """
        Retrieve the name of the best embedder from the retriever node.

        :return: Name of the best embedder.
        """
        best_retriever_artifact: RetrieverArtifact = self._get_best_artifact(node_type=NodeType.retrieval)  # type: ignore[assignment]
        return best_retriever_artifact.embedder_name

    def get_best_test_scores(self) -> NDArray[np.float64] | None:
        """
        Retrieve the test scores from the best scorer node.

        :return: Test scores as a numpy array.
        """
        best_scorer_artifact: ScorerArtifact = self._get_best_artifact(node_type=NodeType.scoring)  # type: ignore[assignment]
        return best_scorer_artifact.test_scores

    def get_best_oos_scores(self) -> NDArray[np.float64] | None:
        """
        Retrieve the out-of-scope scores from the best scorer node.

        :return: Out-of-scope scores as a numpy array.
        """
        best_scorer_artifact: ScorerArtifact = self._get_best_artifact(node_type=NodeType.scoring)  # type: ignore[assignment]
        return best_scorer_artifact.oos_scores

    def dump_evaluation_results(self) -> dict[str, Any]:
        """
        Dump evaluation results for all nodes.

        :return: Dictionary containing metrics and configurations for all nodes.
        """
        node_wise_metrics = {node_type: self._get_metrics_values(node_type) for node_type in NodeType}
        return {
            "metrics": node_wise_metrics,
            "configs": self.trials.model_dump(),
        }

    def get_inference_nodes_config(self) -> list[InferenceNodeConfig]:
        """
        Generate configuration for inference nodes based on the best trials.

        :return: List of `InferenceNodeConfig` objects for inference nodes.
        """
        trial_ids = [self._get_best_trial_idx(node_type) for node_type in NodeType]
        res = []
        for idx, node_type in zip(trial_ids, NodeType, strict=True):
            if idx is None:
                continue
            trial = self.trials.get_trial(node_type, idx)
            res.append(
                InferenceNodeConfig(
                    node_type=node_type,
                    module_type=trial.module_type,
                    module_config=trial.module_params,
                    load_path=trial.module_dump_dir,
                ),
            )
        return res

    def _get_best_module(self, node_type: str) -> "Module | None":
        """
        Retrieve the best module for a specific node type.

        :param node_type: Type of the node.
        :return: The best module, or None if no best trial exists.
        """
        idx = self._get_best_trial_idx(node_type)
        if idx is not None:
            return self.modules.get(node_type)[idx]
        return None

    def get_best_modules(self) -> dict[NodeType, "Module"]:
        """
        Retrieve the best modules for all node types.

        :return: Dictionary of the best modules for each node type.
        """
        res = {nt: self._get_best_module(nt) for nt in NodeType}
        return {nt: m for nt, m in res.items() if m is not None}
