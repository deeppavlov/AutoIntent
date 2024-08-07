import json
import numpy as np


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


def make_report(logs: dict) -> str:
    nodes = ["retrieval", "scoring", "prediction"]
    ids = [np.argmax(logs["metrics"][node]) for node in nodes]
    configs = []
    for i, node in zip(ids, nodes):
        cur_config = logs["configs"][node][i]
        cur_config["metric_value"] = logs["metrics"][node][i]
        configs.append(cur_config)
    messages = [json.dumps(c, indent=4) for c in configs]
    return "\n".join(messages)


if __name__ == "__main__":
    import sys

    sys.path.append("/home/voorhs/repos/AutoIntent")

    import json
    from argparse import ArgumentParser

    import yaml

    from src import DataHandler
    from src.nodes import Node, PredictionNode, RetrievalNode, ScoringNode

    parser = ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="scripts/base_pipeline.assets/example-config.yaml",
    )
    parser.add_argument(
        "--data-path", type=str, default="data/intent_records/banking77.json"
    )
    parser.add_argument(
        "--logs-path",
        type=str,
        default="scripts/base_pipeline.assets/example-logs.json",
    )
    args = parser.parse_args()

    banking77 = json.load(open(args.data_path))
    data_handler = DataHandler(banking77)

    available_nodes = {
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

    import os

    dirname = os.path.dirname(args.logs_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    json.dump(
        logs, open(args.logs_path, "w"), indent=4, ensure_ascii=False, cls=NumpyEncoder
    )

    print(make_report(logs))
