import json
import os
from pprint import pprint
import numpy as np
import logging
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

class OptimizationLogs:
    """TODO continous IO with file system (to be able to restore the state of optimization)"""

    def __init__(self, logs_path: str):
        self.logs_path = logs_path
        if os.path.exists(logs_path):
            self.cache = self.load_logs()
        else:
            self.cache = {
                "best_assets": {
                    "retrieval": None,
                    "scoring": {"test_scores": None, "oos_scores": None},
                    "prediction": None,
                },
                "metrics": {
                    "retrieval": np.array([]),
                    "scoring": np.array([]),
                    "prediction": np.array([])
                },
                "configs": {"retrieval": [], "scoring": [], "prediction": []},
            }


    def load_logs(self):
        with open(self.logs_path, 'r') as f:
            data = json.load(f)

        for node_type in data['metrics']:
            if isinstance(data['metrics'][node_type], list):
                data['metrics'][node_type] = np.array(data['metrics'][node_type], dtype=float)
            elif not isinstance(data['metrics'][node_type], np.ndarray):
                logger.warning(
                    f"Unexpected type for metrics of {node_type}: {type(data['metrics'][node_type])}")
                data['metrics'][node_type] = np.array([], dtype=float)

        expected_keys = ['best_assets', 'metrics', 'configs']
        for key in expected_keys:
            if key not in data:
                logger.warning(f"Missing key in loaded data: {key}")
                data[key] = {}

        if 'scoring' in data['best_assets']:
            for key in ['test_scores', 'oos_scores']:
                if data['best_assets']['scoring'][key] is not None:
                    data['best_assets']['scoring'][key] = np.array(
                        data['best_assets']['scoring'][key])

        if 'prediction' in data['best_assets'] and data['best_assets']['prediction'] is not None:
            data['best_assets']['prediction'] = np.array(data['best_assets']['prediction'])

        return data

    def save_logs(self):
        with open(self.logs_path, 'w') as f:
            json.dump(self.cache, f, indent=4, cls=NumpyEncoder)

    def log_module_optimization(self, node_type, module_type, module_config, metric_value,
                                metric_name, assets, verbose=False):
        """
        Purposes:
        - save optimization results in a text form (hyperparameters and corresponding metrics)
        - update best assets
        """
        logger.info(f"Logging optimization for {node_type}: {module_type}")
        metrics_list = self.cache["metrics"][node_type]

        if isinstance(metrics_list, np.ndarray):
            previous_best = np.max(metrics_list) if metrics_list.size > 0 else -float("inf")
        else:
            previous_best = max(metrics_list, default=-float("inf"))

        if metric_value > previous_best:
            logger.info(f"New best {node_type} found. Metric value: {metric_value}")
            self.cache["best_assets"][node_type] = assets
        else:
            logger.info(
                f"Not the best {node_type}. Metric value: {metric_value}, Previous best: {previous_best}")

        logs = dict(module_type=module_type, metric_name=metric_name, metric_value=metric_value,
                    **module_config)
        self.cache["configs"][node_type].append(logs)
        if verbose:
            pprint(logs)

        if isinstance(metrics_list, np.ndarray):
            self.cache["metrics"][node_type] = np.append(metrics_list, metric_value)
        else:
            metrics_list.append(metric_value)

        self.save_logs()
        logger.info(
            f"Optimization logged for {node_type}. Total configurations: {len(self.cache['configs'][node_type])}")

    def get_best_embedder(self):
        return self.cache["best_assets"]["retrieval"]

    def get_best_test_scores(self):
        return self.cache["best_assets"]["scoring"]["test_scores"]

    def get_best_oos_scores(self):
        return self.cache["best_assets"]["scoring"]["oos_scores"]

    def get_best_modules(self):
        best_modules = {}
        for node_type in self.cache["configs"]:
            metrics = self.cache["metrics"][node_type]
            configs = self.cache["configs"][node_type]
            if isinstance(metrics, np.ndarray):
                metrics_not_empty = metrics.size > 0
            else:
                metrics_not_empty = bool(metrics)

            if metrics_not_empty and configs:
                best_index = np.argmax(metrics)
                best_modules[node_type] = configs[best_index]
            else:
                best_modules[node_type] = None
        return best_modules

    def dump(self):
        return dict(metrics=self.cache["metrics"], configs=self.cache["configs"])