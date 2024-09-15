import importlib.resources as ires
import json
import os
import pickle
import inspect
import numpy as np
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ..nodes import (
    Node,
    PredictionNode,
    RegExpNode,
    RetrievalNode,
    ScoringNode,
)
from .utils import NumpyEncoder


class Pipeline:
    available_nodes = {
        "regexp": RegExpNode,
        "retrieval": RetrievalNode,
        "scoring": ScoringNode,
        "prediction": PredictionNode,
    }

    def serialize(self):
        return pickle.dumps(self)

    @classmethod
    def load(cls, serialized_pipeline):
        return pickle.loads(serialized_pipeline)

    def predict(self, texts, intents_dict):
        logger.info(f"Starting prediction for texts: {texts}")
        results = []
        for idx, text in enumerate(texts):
            logger.info(f"Processing text {idx}: {text}")
            current_input = text
            for node_config in self.config["nodes"]:
                node_type = node_config["node_type"]
                best_module = self.best_modules[node_type]
                logger.info(f"Processing node type: {node_type}")
                logger.info(f"Current input before processing: {current_input}")
                if node_type == 'scoring':
                    logger.info("Applying scoring module")
                    current_input = best_module.predict([(current_input, '')])
                elif node_type == 'prediction':
                    logger.info("Applying prediction module")
                    current_input = np.atleast_2d(current_input)
                    logger.info(f"Input shape after np.atleast_2d: {current_input.shape}")
                    current_input = best_module.predict(current_input)
                else:
                    logger.info(f"Applying {node_type} module")
                    current_input = best_module.predict([current_input])

                logger.info(f"Output after {node_type} module: {current_input}")

                if isinstance(current_input, (list, np.ndarray)) and len(current_input) == 1:
                    current_input = current_input[0]
                    logger.info(f"Extracted first element: {current_input}")

                if node_type == 'scoring':
                    output = {intents_dict[i]: current_input[i] for i in range(len(current_input))}
                    sorted_output = dict(
                        sorted(output.items(), key=lambda item: item[1], reverse=True))
                    logger.info(f"INTENTS: {sorted_output}")

                elif node_type == 'prediction':
                    output = [intents_dict[i] for i in range(len(current_input)) if current_input[i]==1]
                    logger.info(f"INTENTS: {output}")
                else:
                    output = [intents_dict[j] for i in range(len(current_input))
                               for j in range(len(current_input[i])) if current_input[i][j]==1]
                    logger.info(f"INTENTS: {output}")

            logger.info(f"Final output for text {idx}: {current_input}")
            output = [intents_dict[i] for i in range(len(current_input)) if current_input[i]==1]
            logger.info(f"INTENTS: {output}")
            results.append(current_input)

        logger.info(f"Final results for all texts: {results}")
        return results

    def get_best_module_config(self, node_type):
        node_metrics = self.context.optimization_logs.cache["metrics"][node_type]
        best_index = np.argmax(node_metrics)
        return self.context.optimization_logs.cache["configs"][node_type][best_index]

    def __init__(self, config_path: os.PathLike, mode: str, verbose: bool):
        self.config = load_config(config_path, mode)
        self.verbose = verbose
        self.best_modules = {}
        self.context = None

    def optimize(self, context):
        self.context = context
        for node_config in self.config["nodes"]:
            node_type = node_config["node_type"]
            node: Node = self.available_nodes[node_type](
                modules_search_spaces=node_config["modules"],
                metric=node_config["metric"],
                verbose=self.verbose
            )
            logger.info(f"Optimize node_type: {node_type}")
            node.fit(context)
            print(f"Fitted {node_type}!")

            best_modules = context.optimization_logs.get_best_modules()
            best_config = best_modules.get(node_type)

            if best_config is None:
                logger.warning(f"No best config found for {node_type}. Skipping module creation.")
                continue

            module_class = node.modules_available[best_config['module_type']]

            module_init_params = inspect.signature(module_class.__init__).parameters
            module_params = {k: v for k, v in best_config.items()
                             if k in module_init_params and k != 'module_type'}

            module = module_class(**module_params)

            logger.info(f"Optimize module: {module}")

            context.print_all_fields()
            if hasattr(module, 'fit'):
                module.fit(context)

            self.best_modules[node_type] = module

    def dump(self, logs_dir: os.PathLike, run_name: str):
        optimization_results = self.context.optimization_logs.dump()

        if logs_dir == "":
            logs_dir = os.getcwd()
        logs_dir = os.path.join(logs_dir, run_name)
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        logs_path = os.path.join(logs_dir, "logs.json")
        json.dump(optimization_results, open(logs_path, "w"), indent=4, ensure_ascii=False,
                  cls=NumpyEncoder)
        config_path = os.path.join(logs_dir, "config.yaml")
        yaml.dump(self.config, open(config_path, "w"))

        if self.verbose:
            print(
                make_report(
                    optimization_results,
                    nodes=[node_config["node_type"] for node_config in self.config["nodes"]]
                )
            )

        train_data, test_data = self.context.data_handler.dump()
        train_path = os.path.join(logs_dir, "train_data.json")
        test_path = os.path.join(logs_dir, "test_data.json")
        json.dump(train_data, open(train_path, "w"), indent=4, ensure_ascii=False)
        json.dump(test_data, open(test_path, "w"), indent=4, ensure_ascii=False)


def load_config(config_path: os.PathLike, mode: str):
    if config_path != "":
        file = open(config_path)
    else:
        config_name = "default-multilabel-config.yaml" if mode != "multiclass" else "default-multiclass-config.yaml"
        file = ires.files("autointent.datafiles").joinpath(config_name).open()
    return yaml.safe_load(file)


def make_report(logs: dict[str], nodes: list[str]) -> str:
    ids = [np.argmax(logs["metrics"][node]) for node in nodes]
    configs = []
    for i, node in zip(ids, nodes):
        cur_config = logs["configs"][node][i]
        cur_config["metric_value"] = logs["metrics"][node][i]
        configs.append(cur_config)
    messages = [json.dumps(c, indent=4) for c in configs]
    return "\n".join(messages)
