"""Pipeline optimizer."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import yaml
from typing_extensions import assert_never

from autointent import Context, Dataset, OptimizationConfig
from autointent.configs import (
    CrossEncoderConfig,
    DataConfig,
    EmbedderConfig,
    InferenceNodeConfig,
    LoggingConfig,
)
from autointent.custom_types import (
    ListOfGenericLabels,
    NodeType,
    SamplerType,
    SearchSpacePresets,
    SearchSpaceValidationMode,
)
from autointent.metrics import DECISION_METRICS
from autointent.nodes import InferenceNode, NodeOptimizer
from autointent.utils import load_preset, load_search_space

from ._schemas import InferencePipelineOutput, InferencePipelineUtteranceOutput

if TYPE_CHECKING:
    from autointent.modules.base import BaseDecision, BaseScorer


class Pipeline:
    """Pipeline optimizer class."""

    def __init__(
        self,
        nodes: list[NodeOptimizer] | list[InferenceNode],
        sampler: SamplerType = "brute",
        seed: int = 42,
    ) -> None:
        """Initialize the pipeline optimizer.

        Args:
            nodes: List of nodes.
            sampler: Sampler type.
            seed: Random seed.
        """
        self._logger = logging.getLogger(__name__)
        self.nodes = {node.node_type: node for node in nodes}
        self.seed = seed
        if sampler not in get_args(SamplerType):
            msg = f"Sampler should be one of {get_args(SamplerType)}"
            raise ValueError(msg)

        self.sampler = sampler

        if isinstance(nodes[0], NodeOptimizer):
            self.logging_config = LoggingConfig()
            self.embedder_config = EmbedderConfig()
            self.cross_encoder_config = CrossEncoderConfig()
            self.data_config = DataConfig()
        elif not isinstance(nodes[0], InferenceNode):
            assert_never(nodes)

    def set_config(self, config: LoggingConfig | EmbedderConfig | CrossEncoderConfig | DataConfig) -> None:
        """Set the configuration for the pipeline.

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

    @classmethod
    def from_search_space(cls, search_space: list[dict[str, Any]] | Path | str, seed: int = 42) -> "Pipeline":
        """Search space to pipeline optimizer.

        Args:
            search_space: Search space.
            seed: Random seed.

        Returns:
            Pipeline optimizer.
        """
        if not isinstance(search_space, list):
            search_space = load_search_space(search_space)
        nodes = [NodeOptimizer(**node) for node in search_space]
        return cls(nodes=nodes, seed=seed)

    @classmethod
    def from_preset(cls, name: SearchSpacePresets, seed: int = 42) -> "Pipeline":
        optimization_config = load_preset(name)
        config = OptimizationConfig(seed=seed, **optimization_config)
        return cls.from_optimization_config(config=config)

    @classmethod
    def from_optimization_config(cls, config: dict[str, Any] | Path | str | OptimizationConfig) -> "Pipeline":
        """Create pipeline optimizer from optimization config.

        :param config: Optimization config
        :return:
        """
        if isinstance(config, OptimizationConfig):
            optimization_config = config
        else:
            if isinstance(config, dict):
                dict_params = config
            else:
                with Path(config).open() as file:
                    dict_params = yaml.safe_load(file)
            optimization_config = OptimizationConfig(**dict_params)

        pipeline = cls(
            [NodeOptimizer(**node) for node in optimization_config.search_space],
            optimization_config.sampler,
            optimization_config.seed,
        )
        pipeline.set_config(optimization_config.logging_config)
        pipeline.set_config(optimization_config.data_config)
        pipeline.set_config(optimization_config.embedder_config)
        pipeline.set_config(optimization_config.cross_encoder_config)
        return pipeline

    def _fit(self, context: Context, sampler: SamplerType) -> None:
        """Optimize the pipeline.

        Args:
            context: Context object.
            sampler: Sampler type.
        """
        self.context = context
        self._logger.info("starting pipeline optimization...")
        self.context.callback_handler.start_run(
            run_name=self.context.logging_config.get_run_name(),
            dirpath=self.context.logging_config.dirpath,
        )
        for node_type in NodeType:
            node_optimizer = self.nodes.get(node_type, None)
            if node_optimizer is not None:
                node_optimizer.fit(context, sampler)  # type: ignore[union-attr]
        self.context.callback_handler.end_run()

    def _is_inference(self) -> bool:
        """Check the mode in which pipeline is.

        Returns:
            True if pipeline is in inference mode, False otherwise.
        """
        return isinstance(self.nodes[NodeType.scoring], InferenceNode)

    def fit(
        self,
        dataset: Dataset,
        refit_after: bool = False,
        sampler: SamplerType | None = None,
        incompatible_search_space: SearchSpaceValidationMode = "filter",
    ) -> Context:
        """Optimize the pipeline from dataset.

        Args:
            dataset: Dataset for optimization.
            refit_after: Whether to refit after optimization.
            sampler: Sampler type to use.
            incompatible_search_space: How to handle incompatible search space.

        Returns:
            Context object.

        Raises:
            RuntimeError: If pipeline is in inference mode.
        """
        if self._is_inference():
            msg = "Pipeline in inference mode cannot be fitted"
            raise RuntimeError(msg)

        context = Context()
        context.set_dataset(dataset, self.data_config)
        context.configure_logging(self.logging_config)
        context.configure_transformer(self.embedder_config)
        context.configure_transformer(self.cross_encoder_config)

        self.validate_modules(dataset, mode=incompatible_search_space)

        test_utterances = context.data_handler.test_utterances()
        if test_utterances is None:
            self._logger.warning(
                "Test data is not provided. Final test metrics won't be calculated after pipeline optimization."
            )

        if sampler is None:
            sampler = self.sampler

        self._fit(context, sampler)

        if context.is_ram_to_clear():
            nodes_configs = context.optimization_info.get_inference_nodes_config()
            nodes_list = [InferenceNode.from_config(cfg) for cfg in nodes_configs]
        else:
            modules_dict = context.optimization_info.get_best_modules()
            nodes_list = [InferenceNode(module, node_type) for node_type, module in modules_dict.items()]

        self.nodes = {node.node_type: node for node in nodes_list}

        if refit_after:
            # TODO reflect this refitting in dumped version of pipeline
            self._refit(context)

        if test_utterances is not None:
            predictions = self.predict(test_utterances)
            for metric_name, metric in DECISION_METRICS.items():
                context.optimization_info.pipeline_metrics[metric_name] = metric(
                    context.data_handler.test_labels(),
                    predictions,
                )
            context.callback_handler.log_final_metrics(context.optimization_info.dump_evaluation_results())

        return context

    def validate_modules(self, dataset: Dataset, mode: SearchSpaceValidationMode) -> None:
        """Validate modules with dataset.

        Args:
            dataset: Dataset for validation.
            mode: Validation mode.
        """
        for node in self.nodes.values():
            if isinstance(node, NodeOptimizer):
                node.validate_nodes_with_dataset(dataset, mode)

    @classmethod
    def from_dict_config(cls, nodes_configs: list[dict[str, Any]]) -> "Pipeline":
        """Create inference pipeline from dictionary config.

        Args:
            nodes_configs: list of config for nodes

        Returns:
            Inference pipeline
        """
        return cls.from_config([InferenceNodeConfig(**cfg) for cfg in nodes_configs])

    @classmethod
    def from_config(cls, nodes_configs: list[InferenceNodeConfig]) -> "Pipeline":
        """Create inference pipeline from config.

        Args:
            nodes_configs: list of config for nodes

        Returns:
            Inference pipeline
        """
        nodes = [InferenceNode.from_config(cfg) for cfg in nodes_configs]
        return cls(nodes)

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedder_config: EmbedderConfig | None = None,
        cross_encoder_config: CrossEncoderConfig | None = None,
    ) -> "Pipeline":
        """Load pipeline in inference mode.

        Args:
            path: Path to load
            embedder_config: one can override presaved settings
            cross_encoder_config: one can override presaved settings

        Returns:
            Inference pipeline
        """
        with (Path(path) / "inference_config.yaml").open() as file:
            inference_dict_config: dict[str, Any] = yaml.safe_load(file)

        inference_config = [
            InferenceNodeConfig(
                **node_config, embedder_config=embedder_config, cross_encoder_config=cross_encoder_config
            )
            for node_config in inference_dict_config["nodes_configs"]
        ]

        return cls.from_config(inference_config)

    def predict(self, utterances: list[str]) -> ListOfGenericLabels:
        """Predict the labels for the utterances.

        Args:
            utterances: list of utterances

        Returns:
            list of predicted labels
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
            context: Context object.

        Raises:
            RuntimeError: If pipeline is in optimization mode.
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
            Inference pipeline output
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
        logs: Logs dictionary.
        nodes: List of node types.

    Returns:
        String report.
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
