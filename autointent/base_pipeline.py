import importlib.resources as ires
import json
import os
from argparse import ArgumentParser
from datetime import datetime
from typing import Any

import numpy as np
import yaml

from autointent import Context
from autointent.cache_utils import get_db_dir
from autointent.nodes import (
    Node,
    PredictionNode,
    RegExpNode,
    RetrievalNode,
    ScoringNode,
)


class NumpyEncoder(json.JSONEncoder):
    """Helper for dumping logs. Problem explained: https://stackoverflow.com/q/50916422"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def make_report(logs: dict[str, Any], nodes: list[str]) -> str:
    ids = [np.argmax(logs["metrics"][node]) for node in nodes]
    configs = []
    for i, node in zip(ids, nodes):
        cur_config = logs["configs"][node][i]
        cur_config["metric_value"] = logs["metrics"][node][i]
        configs.append(cur_config)
    messages = [json.dumps(c, indent=4) for c in configs]
    return "\n".join(messages)


def load_data(data_path: os.PathLike, multilabel: bool):
    """load data from the given path or load sample data which is distributed along with the autointent package"""
    if data_path == "default":
        data_name = "dstc3-20shot.json" if multilabel else "banking77.json"
        file = ires.files("autointent.datafiles").joinpath(data_name).open()
    elif data_path != "":
        file = open(data_path)
    else:
        return []
    return json.load(file)


def load_config(config_path: os.PathLike, mode: str):
    """load config from the given path or load default config which is distributed along with the autointent package"""
    if config_path != "":
        file = open(config_path)
    else:
        config_name = "default-multilabel-config.yaml" if mode != "multiclass" else "default-multiclass-config.yaml"
        file = ires.files("autointent.datafiles").joinpath(config_name).open()
    return yaml.safe_load(file)


def get_run_name(run_name: str, config_path: os.PathLike):
    if run_name == "":
        run_name = "example_run_name" if config_path == "" else os.path.basename(config_path).split(".")[0]
    return f"{run_name}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"


class Pipeline:
    available_nodes = {
        "regexp": RegExpNode,
        "retrieval": RetrievalNode,
        "scoring": ScoringNode,
        "prediction": PredictionNode,
    }

    def __init__(self, config_path: os.PathLike, mode: str, verbose: bool):
        # TODO add config validation
        self.config = load_config(config_path, mode)
        self.verbose = verbose

    def optimize(self, context: Context):
        self.logs = context.optimization_logs
        for node_config in self.config["nodes"]:
            node: Node = self.available_nodes[node_config["node_type"]](
                modules_search_spaces=node_config["modules"], metric=node_config["metric"], verbose=self.verbose
            )
            node.fit(context)
            print("fitted!")

    def dump(self, logs_dir: os.PathLike, run_name: str):
        optimization_results = self.logs.dump_logs()

        if logs_dir == "":
            logs_dir = os.getcwd()

        logs_dir = os.path.join(logs_dir, run_name)

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        logs_path = os.path.join(logs_dir, "logs.json")
        json.dump(optimization_results, open(logs_path, "w"), indent=4, ensure_ascii=False, cls=NumpyEncoder)

        config_path = os.path.join(logs_dir, "config.yaml")
        yaml.dump(self.config, open(config_path, "w"))

        if self.verbose:
            print(
                make_report(optimization_results, nodes=[node_config["node_type"] for node_config in self.config["nodes"]])
            )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        help="Path to a yaml configuration file that defines the optimization search space. "
        "Omit this to use the default configuration.",
    )
    parser.add_argument(
        "--multiclass-path",
        type=str,
        default="",
        help="Path to a json file with intent records. "
        'Set to "default" to use banking77 data stored within the autointent package.',
    )
    parser.add_argument(
        "--multilabel-path",
        type=str,
        default="",
        help="Path to a json file with utterance records. "
        'Set to "default" to use dstc3 data stored within the autointent package.',
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="",
        help="Path to a json file with utterance records. "
        "Skip this option if you want to use a random subset of the training sample as test data.",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="",
        help="Location where to save chroma database file. Omit to use your system's default cache directory.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="",
        help="Location where to save optimization logs that will be saved as `<logs_dir>/<run_name>_<cur_datetime>/logs.json`",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Name of the run prepended to optimization logs filename"
    )
    parser.add_argument(
        "--mode",
        choices=["multiclass", "multilabel", "multiclass_as_multilabel"],
        default="multiclass",
        help="Evaluation mode. This parameter must be consistent with provided data.",
    )
    parser.add_argument(
       "--device",
        type=str,
        default="cuda:0",
        help="Specify device in torch notation"
    )
    parser.add_argument(
        "--regex-sampling",
        type=int,
        default=0,
        help="Number of shots per intent to sample from regular expressions. "
        "This option extends sample utterances within multiclass intent records.",
    )
    parser.add_argument(
       "--seed",
        type=int,
        default=0,
        help="Affects the data partitioning"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print to console during optimization"
    )
    parser.add_argument(
        "--multilabel-generation-config",
        type=str,
        default="",
        help='Config string like "[20, 40, 20, 10]" means 20 one-label examples, '
        "40 two-label examples, 20 three-label examples, 10 four-label examples. "
        "This option extends multilabel utterance records.",
    )
    args = parser.parse_args()

    # configure the run and data
    run_name = get_run_name(args.run_name, args.config_path)
    db_dir = get_db_dir(args.db_dir, run_name)

    # create shared objects for a whole pipeline
    context = Context(
        load_data(args.multiclass_path, multilabel=False),
        load_data(args.multilabel_path, multilabel=True),
        load_data(args.test_path, multilabel=True),
        args.device,
        args.mode,
        args.multilabel_generation_config,
        db_dir,
        args.regex_sampling,
        args.seed,
    )

    # run optimization
    pipeline = Pipeline(args.config_path, args.mode, args.verbose)
    pipeline.optimize(context)

    # save results
    pipeline.dump(args.logs_dir, run_name)
