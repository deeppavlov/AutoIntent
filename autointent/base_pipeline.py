import importlib.resources as ires
import json
import os
from argparse import ArgumentParser
from datetime import datetime

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


def make_report(logs: dict, nodes) -> str:
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
    if data_path != "":
        file = open(data_path)
    else:
        data_name = 'dstc3-20shot.json' if multilabel else 'banking77.json'
        file = ires.files('autointent.datafiles').joinpath(data_name).open()
    return json.load(file)


def load_config(config_path: os.PathLike, multilabel: bool):
    """load config from the given path or load default config which is distributed along with the autointent package"""
    if config_path != "":
        file = open(config_path)
    else:
        config_name = 'default-multilabel-config.yaml' if multilabel else 'default-multiclass-config.yaml'
        file = ires.files('autointent.datafiles').joinpath(config_name).open()
    return yaml.safe_load(file)


def dump_logs(logs, logs_dir, run_name: str):
    if logs_dir == "":
        logs_dir = os.getcwd()

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logs_path = os.path.join(logs_dir, f"{run_name}.json")

    json.dump(
        logs, open(logs_path, "w"), indent=4, ensure_ascii=False, cls=NumpyEncoder
    )


def get_run_name(run_name: str, config_path: os.PathLike):
    if run_name == "":
        run_name = (
            "example_run_name"
            if config_path == ""
            else os.path.basename(config_path).split('.')[0]
        )
    return f"{run_name}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
    

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        help="Path to a yaml configuration file that defines the optimization search space. Omit this to use the default configuration."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="",
        help="Path to a json file with intent records. Omit this to use banking77 data stored within the autointent package."
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="",
        help="Location where to save chroma database file. Omit to use your system's default cache directory."
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="",
        help="Location where to save optimization logs that will be saved as `<logs_dir>/<run_name>_<cur_datetime>.json`"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Name of the run prepended to optimization logs filename"
    )
    parser.add_argument(
        "--multilabel",
        action="store_true",
        help="Use this flag if your data is multilabel"
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
        help="Number of shots per intent to sample from regular expressions"
    )
    parser.add_argument(
       "--seed",
        type=int,
        default=0,
        help="Affects the data partitioning"
    )
    args = parser.parse_args()

    # configure the run and data
    run_name = get_run_name(args.run_name, args.config_path)
    db_dir = get_db_dir(args.db_dir, run_name)
    intent_records = load_data(args.data_path, args.multilabel)

    # create shared objects for a whole pipeline
    context = Context(intent_records, args.device, args.multilabel, db_dir, args.regex_sampling, args.seed)

    # run optimization
    available_nodes = {
        "regexp": RegExpNode,
        "retrieval": RetrievalNode,
        "scoring": ScoringNode,
        "prediction": PredictionNode,
    }
    pipeline_config = load_config(args.config_path, args.multilabel)
    for node_config in pipeline_config["nodes"]:
        node: Node = available_nodes[node_config["node_type"]](
            modules_search_spaces=node_config["modules"], metric=node_config["metric"]
        )
        node.fit(context)
        print("fitted!")

    # save results
    logs = context.optimization_logs.dump_logs()
    dump_logs(logs, args.logs_dir, run_name)
    print(make_report(logs, nodes=[node_config["node_type"] for node_config in pipeline_config["nodes"]]))
