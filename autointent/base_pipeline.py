import json
import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import yaml

from autointent import DataHandler
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


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="scripts/base_pipeline.assets/example-config.yaml",
        help="Path to yaml configuration file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/intent_records/banking77.json",
        help="Path to json file with intent records"
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="",
        help="Location where to save chroma database file"
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="scripts/base_pipeline.assets/",
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
    )
    args = parser.parse_args()

    run_name = (
        args.run_name
        if args.run_name != ""
        else os.path.basename(args.config_path).split('.')[0]
    )
    run_name = f"{run_name}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"

    db_dir = (
        args.db_dir
        if args.db_dir != ""
        else os.path.join('data', 'chroma', run_name)
    )

    intent_records = json.load(open(args.data_path))
    data_handler = DataHandler(intent_records, db_dir, args.multilabel)

    available_nodes = {
        "regexp": RegExpNode,
        "retrieval": RetrievalNode,
        "scoring": ScoringNode,
        "prediction": PredictionNode,
    }

    pipeline_config = yaml.safe_load(open(args.config_path))

    for node_config in pipeline_config["nodes"]:
        node: Node = available_nodes[node_config["node_type"]](
            modules_search_spaces=node_config["modules"], metric=node_config["metric"]
        )
        node.fit(data_handler)
        print("fitted!")

    logs = data_handler.dump_logs()

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    logs_path = os.path.join(args.logs_dir, f"{run_name}.json")

    json.dump(
        logs, open(logs_path, "w"), indent=4, ensure_ascii=False, cls=NumpyEncoder
    )

    print(make_report(logs, nodes=[node_config["node_type"] for node_config in pipeline_config["nodes"]]))
