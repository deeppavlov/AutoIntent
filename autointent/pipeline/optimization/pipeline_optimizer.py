import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from hydra.utils import instantiate

from autointent import Context
from autointent.configs.pipeline_optimizer import PipelineOptimizerConfig
from autointent.nodes import NodeOptimizer

from .utils import NumpyEncoder


class PipelineOptimizer:
    def __init__(self, nodes: list[NodeOptimizer]) -> None:
        self._logger = logging.getLogger(__name__)
        self.nodes = nodes

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> "PipelineOptimizer":
        return instantiate(PipelineOptimizerConfig, **config)  # type: ignore[no-any-return]

    def optimize(self, context: Context) -> None:
        self.context = context
        self._logger.info("starting pipeline optimization...")
        for node_optimizer in self.nodes:
            node_optimizer.fit(context)

    def dump(self, logs_dir: Path) -> None:
        self._logger.debug("dumping logs...")
        optimization_results = self.context.optimization_info.dump_evaluation_results()

        # create appropriate directory
        logs_dir.mkdir(parents=True, exist_ok=True)

        # dump search space and evaluation results
        logs_path = logs_dir / "logs.json"
        with logs_path.open("w") as file:
            json.dump(optimization_results, file, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        # config_path = logs_dir / "config.yaml"
        # with config_path.open("w") as file:
        #     yaml.dump(self.config, file)

        nodes = [node_config.node_info.node_type for node_config in self.nodes]
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
