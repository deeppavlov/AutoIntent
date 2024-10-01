import importlib.resources as ires
import json
import logging
from collections.abc import Callable
from logging import Logger
from pathlib import Path
from typing import ClassVar

import numpy as np
import yaml

from autointent import Context
from autointent.nodes import Node, PredictionNode, RegExpNode, RetrievalNode, ScoringNode

from .utils import NumpyEncoder


class Pipeline:
    available_nodes: ClassVar[dict[str, Callable]] = {
        "regexp": RegExpNode,
        "retrieval": RetrievalNode,
        "scoring": ScoringNode,
        "prediction": PredictionNode,
    }

    def __init__(self, config_path: str, mode: str):
        # TODO add config validation
        self._logger = logging.getLogger(__name__)

        self._logger.debug("loading optimization search space config...")
        self.config = load_config(config_path, mode, self._logger)

    def optimize(self, context: Context):
        self.context = context
        self._logger.info("starting pipeline optimization...")
        for node_config in self.config["nodes"]:
            node_logger = logging.getLogger(node_config["node_type"])
            node: Node = self.available_nodes[node_config["node_type"]](
                modules_search_spaces=node_config["modules"], metric=node_config["metric"], logger=node_logger
            )
            node.fit(context)

    def dump(self, logs_dir: str, run_name: str):
        self._logger.debug("dumping logs...")
        optimization_results = self.context.dump()

        # create appropriate directory
        logs_dir = Path.cwd() if logs_dir == "" else Path(logs_dir)
        logs_dir = logs_dir / run_name
        logs_dir.mkdir(parents=True)

        # dump config and optimization results
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


def load_config(config_path: str, mode: str, logger: Logger):
    """load config from the given path or load default config which is distributed along with the autointent package"""
    if config_path != "":
        logger.debug("loading optimization search space config from %s...)", config_path)
        with Path(config_path).open() as file:
            file_content = file.read()
    else:
        logger.debug("loading default optimization search space config...")
        config_name = "default-multilabel-config.yaml" if mode != "multiclass" else "default-multiclass-config.yaml"
        with ires.files("autointent.datafiles").joinpath(config_name).open() as file:
            file_content = file.read()
    return yaml.safe_load(file_content)


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
