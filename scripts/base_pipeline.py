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
        default="experiments/base_pipeline.assets/example-config.yaml",
    )
    parser.add_argument(
        "--data-path", type=str, default="data/intent_records/banking77.json"
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

    fitted_nodes = []
    for node_config in pipeline_config["nodes"]:
        node: Node = available_nodes[node_config["node_type"]](
            modules_search_spaces=node_config["modules"], metric=node_config["metric"]
        )
        node.fit(data_handler)
        fitted_nodes.append(node)
        print("fitted!")
