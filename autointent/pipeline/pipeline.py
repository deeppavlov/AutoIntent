import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import yaml
from hydra.utils import instantiate

from autointent import Context
from autointent.configs.modules import MODULES_CONFIGS, create_search_space_dataclass
from autointent.configs.node import NodeOptimizerConfig
from autointent.nodes import NodeInfo, NodeOptimizer, PredictionNodeInfo, RetrievalNodeInfo, ScoringNodeInfo

from .utils import NumpyEncoder


class Pipeline:
    available_nodes: ClassVar[dict[str, NodeInfo]] = {
        "retrieval": RetrievalNodeInfo(),
        "scoring": ScoringNodeInfo(),
        "prediction": PredictionNodeInfo(),
    }

    def __init__(self, optimization_config):
        self._logger = logging.getLogger(__name__)
        self.config = optimization_config

    def optimize(self, context: Context):
        self.context = context
        self._logger.info("starting pipeline optimization...")
        for node_config in self.config["nodes"]:
            node_optimizer_config = NodeOptimizerConfig(
                node_info=self.available_nodes[node_config["node_type"]],
                search_space=[parse_search_space(node_config["node_type"], ss) for ss in node_config["modules"]],
                metric=node_config["metric"],
            )
            node_optimizer: NodeOptimizer = instantiate(node_optimizer_config)
            node_optimizer.fit(context)

    def dump(self, logs_dir: str, run_name: str):
        self._logger.debug("dumping logs...")
        optimization_results = self.context.optimization_info.dump_evaluation_results()

        # create appropriate directory
        logs_dir = Path.cwd() if logs_dir == "" else Path(logs_dir)
        logs_dir = logs_dir / run_name
        logs_dir.mkdir(parents=True)

        # dump search space and evaluation results
        logs_path = logs_dir / "logs.json"
        with logs_path.open("w") as file:
            json.dump(optimization_results, file, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        config_path = logs_dir / "config.yaml"
        with config_path.open("w") as file:
            yaml.dump(self.config, file)

        nodes = [node_config["node_type"] for node_config in self.config["nodes"]]
        self._logger.info(make_report(optimization_results, nodes=nodes))

        # dump train and test data splits
        train_data, test_data = self.context.data_handler.dump()
        train_path = logs_dir / "train_data.json"
        test_path = logs_dir / "test_data.json"
        with train_path.open("w") as file:
            json.dump(train_data, file, indent=4, ensure_ascii=False)
        with test_path.open("w") as file:
            json.dump(test_data, file, indent=4, ensure_ascii=False)

        self._logger.info("logs and other assets are saved to %s", logs_dir)

        # dump optimization results (config for inference)
        inference_config = self.context.get_inference_config()
        inference_config_path = logs_dir / "inference_config.yaml"
        with inference_config_path.open("w") as file:
            yaml.dump(inference_config, file)


def make_report(logs: dict[str], nodes: list[str]) -> str:
    ids = [np.argmax(logs["metrics"][node]) for node in nodes]
    configs = []
    for i, node in zip(ids, nodes, strict=False):
        cur_config = logs["configs"][node][i]
        cur_config["metric_value"] = logs["metrics"][node][i]
        configs.append(cur_config)
    messages = [json.dumps(c, indent=4) for c in configs]
    msg = "\n".join(messages)
    return "resulting pipeline configuration is the following:\n" + msg


def parse_search_space(node_type: str, search_space: dict[str, Any]):
    module_config = MODULES_CONFIGS[node_type][search_space["module_type"]]
    make_search_space_model = create_search_space_dataclass(module_config)
    return make_search_space_model(**search_space)
