"""Pipeline optimizer."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from autointent import Context, Dataset
from autointent.configs import DataConfig, InferenceNodeConfig, LoggingConfig, VectorIndexConfig
from autointent.custom_types import ListOfGenericLabels, NodeType, SamplerType
from autointent.metrics import DECISION_METRICS
from autointent.nodes import InferenceNode, NodeOptimizer
from autointent.nodes.schemes import OptimizationConfig
from autointent.utils import load_default_search_space, load_search_space

from ._schemas import InferencePipelineOutput, InferencePipelineUtteranceOutput

if TYPE_CHECKING:
    from autointent.modules.abc import BaseDecision, BaseScorer


class Pipeline:
    """Pipeline optimizer class."""

    def __init__(
        self,
        nodes: list[NodeOptimizer] | list[InferenceNode],
        seed: int = 42,
    ) -> None:
        """
        Initialize the pipeline optimizer.

        :param nodes: list of nodes
        :param seed: random seed
        """
        self._logger = logging.getLogger(__name__)
        self.nodes = {node.node_type: node for node in nodes}
        self.seed = seed

        if isinstance(nodes[0], NodeOptimizer):
            self.logging_config = LoggingConfig(dump_dir=None)
            self.vector_index_config = VectorIndexConfig()
            self.data_config = DataConfig()
        elif not isinstance(nodes[0], InferenceNode):
            msg = "Pipeline should be initialized with list of NodeOptimizers or InferenceNodes"
            raise TypeError(msg)

    def set_config(self, config: LoggingConfig | VectorIndexConfig | DataConfig) -> None:
        """
        Set configuration for the optimizer.

        :param config: Configuration
        """
        if isinstance(config, LoggingConfig):
            self.logging_config = config
        elif isinstance(config, VectorIndexConfig):
            self.vector_index_config = config
        elif isinstance(config, DataConfig):
            self.data_config = config
        else:
            msg = "unknown config type"
            raise TypeError(msg)

    @classmethod
    def from_search_space(cls, search_space: list[dict[str, Any]] | Path | str, seed: int = 42) -> "Pipeline":
        """
        Create pipeline optimizer from dictionary search space.

        :param search_space: Dictionary config
        :param seed: random seed
        """
        if isinstance(search_space, Path | str):
            search_space = load_search_space(search_space)
        validated_search_space = OptimizationConfig(search_space).model_dump()  # type: ignore[arg-type]
        nodes = [NodeOptimizer(**node) for node in validated_search_space]
        return cls(nodes=nodes, seed=seed)

    @classmethod
    def default_optimizer(cls, multilabel: bool, seed: int = 42) -> "Pipeline":
        """
        Create pipeline optimizer with default search space for given classification task.

        :param multilabel: Whether the task multi-label, or single-label.
        :param seed: random seed

        :return: Pipeline
        """
        return cls.from_search_space(search_space=load_default_search_space(multilabel), seed=seed)

    def _fit(self, context: Context, sampler: SamplerType = "brute") -> None:
        """
        Optimize the pipeline.

        :param context: Context
        """
        self.context = context
        self._logger.info("starting pipeline optimization...")
        self.context.callback_handler.start_run(
            run_name=self.context.logging_config.run_name,
            dirpath=self.context.logging_config.dirpath,
        )
        for node_type in NodeType:
            node_optimizer = self.nodes.get(node_type, None)
            if node_optimizer is not None:
                node_optimizer.fit(context, sampler)  # type: ignore[union-attr]
        if not context.vector_index_config.save_db:
            self._logger.info("removing vector database from file system...")
            # TODO clear cache from appdirs
        self.context.callback_handler.end_run()

    def _is_inference(self) -> bool:
        """
        Check the mode in which pipeline is.

        :return: True if pipeline is in inference mode, False if in optimization mode.
        """
        return isinstance(self.nodes[NodeType.scoring], InferenceNode)

    def fit(
        self,
        dataset: Dataset,
        refit_after: bool = False,
        sampler: SamplerType = "brute",
    ) -> Context:
        """
        Optimize the pipeline from dataset.

        :param dataset: Dataset for optimization
        :return: Context
        """
        if self._is_inference():
            msg = "Pipeline in inference mode cannot be fitted"
            raise RuntimeError(msg)

        context = Context()
        context.set_dataset(dataset, self.data_config)
        context.configure_logging(self.logging_config)
        context.configure_vector_index(self.vector_index_config)

        self.validate_modules(dataset)

        test_utterances = context.data_handler.test_utterances()
        if test_utterances is None:
            self._logger.warning(
                "Test data is not provided. Final test metrics won't be calculated after pipeline optimization."
            )

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

    def validate_modules(self, dataset: Dataset) -> None:
        """
        Validate modules with dataset.

        :param dataset: dataset to validate with
        """
        for node in self.nodes.values():
            if isinstance(node, NodeOptimizer):
                node.validate_nodes_with_dataset(dataset)

    @classmethod
    def from_dict_config(cls, nodes_configs: list[dict[str, Any]]) -> "Pipeline":
        """
        Create inference pipeline from dictionary config.

        :param nodes_configs: list of dictionary config for nodes
        :return: pipeline ready for inference
        """
        return cls.from_config([InferenceNodeConfig(**cfg) for cfg in nodes_configs])

    @classmethod
    def from_config(cls, nodes_configs: list[InferenceNodeConfig]) -> "Pipeline":
        """
        Create inference pipeline from config.

        :param nodes_configs: list of config for nodes
        """
        nodes = [InferenceNode.from_config(cfg) for cfg in nodes_configs]
        return cls(nodes)

    @classmethod
    def load(cls, path: str | Path) -> "Pipeline":
        """
        Load pipeline in inference mode.

        This method loads fitted modules and tuned hyperparameters.
        :path: path to optimization run directory
        :return: initialized pipeline, ready for inference
        """
        with (Path(path) / "inference_config.yaml").open() as file:
            inference_dict_config = yaml.safe_load(file)
        return cls.from_dict_config(inference_dict_config["nodes_configs"])

    def predict(self, utterances: list[str]) -> ListOfGenericLabels:
        """
        Predict the labels for the utterances.

        :param utterances: list of utterances
        :return: list of predicted labels
        """
        if not self._is_inference():
            msg = "Pipeline in optimization mode cannot perform inference"
            raise RuntimeError(msg)

        scoring_module: BaseScorer = self.nodes[NodeType.scoring].module  # type: ignore[assignment,union-attr]
        decision_module: BaseDecision = self.nodes[NodeType.decision].module  # type: ignore[assignment,union-attr]

        scores = scoring_module.predict(utterances)
        return decision_module.predict(scores)

    def _refit(self, context: Context) -> None:
        """
        Fit pipeline of already selected modules with all train data.

        :param context: context object to take data from
        :return: list of predicted labels
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
        """
        Predict the labels for the utterances with metadata.

        :param utterances: list of utterances
        :return: prediction output
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
    """
    Generate a report from optimization logs.

    :param logs: Logs
    :param nodes: Nodes
    :return: String report
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
