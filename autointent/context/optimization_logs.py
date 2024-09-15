from pprint import pprint

import logging

logger = logging.getLogger(__name__)

class OptimizationLogs:
    """TODO continous IO with file system (to be able to restore the state of optimization)"""

    def __init__(
        self,
    ):
        self.cache = dict(
            best_assets=dict(
                regexp=None,  # TODO: choose the format
                retrieval=None,  # str, name of best retriever
                scoring=dict(
                    test_scores=None, oos_scores=None
                ),  # dict with values of two np.ndarrays of shape (n_samples, n_classes), from best scorer
                prediction=None,  # np.ndarray of shape (n_samples,), from best predictor
            ),
            metrics=dict(regexp=[], retrieval=[], scoring=[], prediction=[]),
            configs=dict(regexp=[], retrieval=[], scoring=[], prediction=[]),
        )

    def log_module_optimization(
        self,
        node_type: str,
        module_type: str,
        module_config: dict,
        metric_value: float,
        metric_name: str,
        assets,
        verbose=False,
    ):
        """
        Purposes:
        - save optimization results in a text form (hyperparameters and corresponding metrics)
        - update best assets
        """

        # "update leaderboard" if it's a new best metric
        metrics_list = self.cache["metrics"][node_type]
        previous_best = max(metrics_list, default=-float("inf"))
        if metric_value > previous_best:
            self.cache["best_assets"][node_type] = assets

        # logging
        logs = dict(
            module_type=module_type,
            metric_name=metric_name,
            metric_value=metric_value,
            **module_config,
        )
        self.cache["configs"][node_type].append(logs)
        if verbose:
            pprint(logs)
        metrics_list.append(metric_value)

    def get_best_embedder(self):
        return self.cache["best_assets"]["retrieval"]

    def get_best_test_scores(self):
        return self.cache["best_assets"]["scoring"]["test_scores"]

    def get_best_oos_scores(self):
        return self.cache["best_assets"]["scoring"]["oos_scores"]

    def dump(self):
        res = dict(
            metrics=self.cache["metrics"],
            configs=self.cache["configs"],
        )
        return res

    def print_logs(self):
        logger.info("OptimizationLogs:")
        logger.info("Best assets:")
        for node_type, asset in self.cache["best_assets"].items():
            if isinstance(asset, dict):
                logger.info(f"  {node_type}:")
                for key, value in asset.items():
                    if value is not None:
                        logger.info(f"    {key}: {type(value).__name__}")
                    else:
                        logger.info(f"    {key}: None")
            else:
                logger.info(f"  {node_type}: {asset}")

        logger.info("Metrics:")
        for node_type, metrics in self.cache["metrics"].items():
            if metrics:
                logger.info(
                    f"  {node_type}: min={min(metrics):.4f}, max={max(metrics):.4f}, count={len(metrics)}")
            else:
                logger.info(f"  {node_type}: No metrics recorded")

        logger.info("Configs:")
        for node_type, configs in self.cache["configs"].items():
            logger.info(f"  {node_type}: {len(configs)} configurations")
            if configs:
                logger.info("  Last config:")
                pprint(configs[-1], indent=4)

