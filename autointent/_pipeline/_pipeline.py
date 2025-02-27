"""Pipeline optimizer module.

This module defines the Pipeline class, which is responsible for optimizing and managing a pipeline of inference nodes.
It provides functionality for configuration, optimization, validation, and inference.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import yaml
from typing_extensions import assert_never

from autointent import Context, Dataset
from autointent.configs import (
    CrossEncoderConfig,
    DataConfig,
    EmbedderConfig,
    LoggingConfig,
)
from autointent.custom_types import (
    ListOfGenericLabels,
    NodeType,
    SamplerType,
    SearchSpaceValidationMode,
)
from autointent.nodes import InferenceNode, NodeOptimizer

from ._schemas import InferencePipelineOutput, InferencePipelineUtteranceOutput

if TYPE_CHECKING:
    from autointent.modules.base import BaseDecision, BaseScorer


class Pipeline:
    """Pipeline optimizer for managing and optimizing inference nodes.

    This class is responsible for initializing and optimizing a sequence of nodes that perform inference tasks.
    It supports loading configurations, validating data, and making predictions.

    Attributes:
        nodes: Dictionary of node types mapped to their respective objects.
        sampler: Sampling method used for optimization.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        nodes: list[NodeOptimizer] | list[InferenceNode],
        sampler: SamplerType = "brute",
        seed: int = 42,
    ) -> None:
        """Initialize the pipeline optimizer.

        Args:
            nodes: List of nodes to be optimized or used for inference.
            sampler: Sampling strategy for optimization. Defaults to "brute".
            seed: Random seed for reproducibility. Defaults to 42.

        Raises:
            ValueError: If the provided sampler type is invalid.
        """
        self._logger = logging.getLogger(__name__)
        self.nodes = {node.node_type: node for node in nodes}
        self.seed = seed
        if sampler not in get_args(SamplerType):
            msg = f"Sampler should be one of {get_args(SamplerType)}"
            raise ValueError(msg)

        self.sampler = sampler

        if isinstance(nodes[0], NodeOptimizer):
            self.logging_config = LoggingConfig(dump_dir=None)
            self.embedder_config = EmbedderConfig()
            self.cross_encoder_config = CrossEncoderConfig()
            self.data_config = DataConfig()
        elif not isinstance(nodes[0], InferenceNode):
            assert_never(nodes)

    def set_config(self, config: LoggingConfig | EmbedderConfig | CrossEncoderConfig | DataConfig) -> None:
        """Set configuration for the pipeline.

        Args:
            config: Configuration object.
        """
        if isinstance(config, LoggingConfig):
            self.logging_config = config
        elif isinstance(config, EmbedderConfig):
            self.embedder_config = config
        elif isinstance(config, CrossEncoderConfig):
            self.cross_encoder_config = config
        elif isinstance(config, DataConfig):
            self.data_config = config
        else:
            assert_never(config)

    def fit(self, dataset: Dataset) -> Context:
        """Optimize the pipeline using a dataset.

        Args:
            dataset: The dataset used for optimization.

        Returns:
            Context: The resulting context after optimization.
        """
        if self._is_inference():
            msg = "Pipeline in inference mode cannot be fitted"
            raise RuntimeError(msg)

        context = Context()
        context.set_dataset(dataset, self.data_config)
        context.configure_logging(self.logging_config)
        context.configure_transformer(self.embedder_config)
        context.configure_transformer(self.cross_encoder_config)

        self._fit(context, self.sampler)
        return context

    def validate_modules(self, dataset: Dataset, mode: SearchSpaceValidationMode) -> None:
        """Validate nodes against a dataset.

        Args:
            dataset: Dataset used for validation.
            mode: Validation mode.
        """
        for node in self.nodes.values():
            if isinstance(node, NodeOptimizer):
                node.validate_nodes_with_dataset(dataset, mode)

    def _is_inference(self) -> bool:
        """Check whether the pipeline is in inference mode.

        Returns:
            True if pipeline is in inference mode, otherwise False.
        """
        return isinstance(self.nodes[NodeType.scoring], InferenceNode)

    @classmethod
    def load(cls, path: str | Path) -> "Pipeline":
        """Load a pipeline from a given directory.

        Args:
            path: Path to the directory containing the pipeline configuration.

        Returns:
            Loaded pipeline instance.
        """
        with (Path(path) / "inference_config.yaml").open() as file:
            inference_dict_config = yaml.safe_load(file)
        return cls.from_dict_config(inference_dict_config["nodes_configs"])

    def predict(self, utterances: list[str]) -> ListOfGenericLabels:
        """Predict labels for a list of utterances.

        Args:
            utterances: List of utterances to predict labels for.

        Returns:
            ListOfGenericLabels: Predicted labels for the utterances.

        Raises:
            RuntimeError: If the pipeline is not in inference mode.
        """
        if not self._is_inference():
            msg = "Pipeline in optimization mode cannot perform inference"
            raise RuntimeError(msg)

        scoring_module: BaseScorer = self.nodes[NodeType.scoring].module  # type: ignore[assignment,union-attr]
        decision_module: BaseDecision = self.nodes[NodeType.decision].module  # type: ignore[assignment,union-attr]

        scores = scoring_module.predict(utterances)
        return decision_module.predict(scores)

    def _refit(self, context: Context) -> None:
        """Fit pipeline of already selected modules with all train data.

        Args:
            context: context object to take data from
        """
        if not self._is_inference():
            msg = "Pipeline in optimization mode cannot perform inference"
            raise RuntimeError(msg)

        scoring_module: BaseScorer = self.nodes[NodeType.scoring].module  # type: ignore[assignment,union-attr]
        decision_module: BaseDecision = self.nodes[NodeType.decision].module  # type: ignore[assignment,union-attr]

        context.data_handler.prepare_for_refit()

        scoring_module.fit(*scoring_module.get_train_data(context))
        scores = scoring_module.predict(context.data_handler.train_utterances(1))

        decision_module.fit(scores, context.data_handler.train_labels(1), context.data_handler.tags)

    def predict_with_metadata(self, utterances: list[str]) -> InferencePipelineOutput:
        """Predict the labels for the utterances with metadata.

        Args:
            utterances: list of utterances

        Returns:
            InferencePipelineOutput: prediction output

        Raises:
            RuntimeError: If the pipeline is not in inference mode.
        """
        if not self._is_inference():
            msg = "Pipeline in optimization mode cannot perform inference"
            raise RuntimeError(msg)

        scores, scores_metadata = self.nodes[NodeType.scoring].module.predict_with_metadata(utterances)  # type: ignore[union-attr]
        predictions = self.nodes[NodeType.decision].module.predict(scores)  # type: ignore[union-attr,arg-type]
        regex_predictions, regex_predictions_metadata = None, None
        if NodeType.regex in self.nodes:
            regex_predictions, regex_predictions_metadata = self.nodes[NodeType.regex].module.predict_with_metadata(  # type: ignore[union-attr]
                utterances,
            )

        outputs = []
        for idx, utterance in enumerate(utterances):
            output = InferencePipelineUtteranceOutput(
                utterance=utterance,
                prediction=predictions[idx],
                regex_prediction=regex_predictions[idx] if regex_predictions is not None else None,
                regex_prediction_metadata=regex_predictions_metadata[idx]
                if regex_predictions_metadata is not None
                else None,
                score=scores[idx],
                score_metadata=scores_metadata[idx] if scores_metadata is not None else None,
            )
            outputs.append(output)

        return InferencePipelineOutput(
            predictions=predictions,
            regex_predictions=regex_predictions,
            utterances=outputs,
        )


def make_report(logs: dict[str, Any], nodes: list[NodeType]) -> str:
    """Generate a report from optimization logs.

    Args:
        logs: Dictionary containing optimization logs.
        nodes: List of node types.

    Returns:
        Formatted report string.
    """
    ids = [np.argmax(logs["metrics"][node]) for node in nodes]
    configs = []
    for i, node in zip(ids, nodes, strict=False):
        cur_config = logs["configs"][node][i]
        cur_config["metric_value"] = logs["metrics"][node][i]
        configs.append(cur_config)
    messages = [json.dumps(c, indent=4) for c in configs]
    msg = "\n".join(messages)
    return "resulting pipeline configuration is the following:\n" + msg
