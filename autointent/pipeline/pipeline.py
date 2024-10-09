import json
import logging
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import numpy as np
import yaml
from hydra.utils import instantiate

from autointent import Context
from autointent.configs.pipeline import PipelineSearchSpace
from autointent.nodes import NodeInfo, NodeOptimizer, PredictionNodeInfo, RetrievalNodeInfo, ScoringNodeInfo

from .utils import NumpyEncoder

PipelineType = TypeVar("PipelineType", bound="Pipeline")


class Pipeline:
    available_nodes: ClassVar[dict[str, NodeInfo]] = {
        "retrieval": RetrievalNodeInfo(),
        "scoring": ScoringNodeInfo(),
        "prediction": PredictionNodeInfo(),
    }

    def __init__(self, nodes: list[NodeOptimizer]) -> None:
        self._logger = logging.getLogger(__name__)
        self.nodes = nodes

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> PipelineType:
        return instantiate(PipelineSearchSpace, **config)

    def optimize(self, context: Context) -> None:
        self.context = context
        self._logger.info("starting pipeline optimization...")
        for node_optimizer in self.nodes:
            node_optimizer.fit(context)

    def dump(self, logs_dir: str, run_name: str) -> None:
        self._logger.debug("dumping logs...")
        optimization_results = self.context.optimization_info.dump_evaluation_results()

        # create appropriate directory
        logs_dir_path = Path.cwd() if logs_dir == "" else Path(logs_dir)
        logs_dir_path = logs_dir_path / run_name
        logs_dir_path.mkdir(parents=True)

        # dump search space and evaluation results
        logs_path = logs_dir_path / "logs.json"
        with logs_path.open("w") as file:
            json.dump(optimization_results, file, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        # config_path = logs_dir / "config.yaml"
        # with config_path.open("w") as file:
        #     yaml.dump(self.config, file)

        nodes = [node_config.node_info.node_type for node_config in self.nodes]
        self._logger.info(make_report(optimization_results, nodes=nodes))

        # dump train and test data splits
        train_data, test_data = self.context.data_handler.dump()
        train_path = logs_dir_path / "train_data.json"
        test_path = logs_dir_path / "test_data.json"
        with train_path.open("w") as file:
            json.dump(train_data, file, indent=4, ensure_ascii=False)
        with test_path.open("w") as file:
            json.dump(test_data, file, indent=4, ensure_ascii=False)

        self._logger.info("logs and other assets are saved to %s", logs_dir_path)

        # dump optimization results (config for inference)
        inference_config = self.context.get_inference_config()
        inference_config_path = logs_dir_path / "inference_config.yaml"
        with inference_config_path.open("w") as file:
            yaml.dump(inference_config, file)


def make_report(logs: dict[str, Any], nodes: list[str]) -> str:
    ids = [np.argmax(logs["metrics"][node]) for node in nodes]
    configs = []
    for i, node in zip(ids, nodes, strict=False):
        cur_config = logs["configs"][node][i]
        cur_config["metric_value"] = logs["metrics"][node][i]
        configs.append(cur_config)
    messages = [json.dumps(c, indent=4) for c in configs]
    msg = "\n".join(messages)
    return "resulting pipeline configuration is the following:\n" + msg
