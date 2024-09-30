import logging
from pprint import pformat
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class ScorerOutputs(BaseModel):
    test_scores: np.ndarray | None = Field(None, description="Scorer outputs for test utterances")
    oos_scores: np.ndarray | None = Field(None, description="Scorer outputs for out-of-scope utterances")


class Artifacts(BaseModel):
    """
    Modules hyperparams and outputs. The best ones are transmitted between nodes of the pipeline
    """
    regexp: list[None] = Field(
        None,
        description="TODO",
    )
    retrieval: list[str] = Field(
        [],
        description="Name of the embedding model chosen after retrieval optimization",
    )
    scoring: list[ScorerOutputs] = Field(
        [],
        description="Outputs from best scorer, numpy arrays of shape (n_samples, n_classes)",
    )
    prediction: list[np.ndarray] = Field(
        [],
        description="Outputs from best predictor, numpy array of shape (n_samples,) or "
        "(n_samples, n_classes) depending on classification mode (multi-class or multi-label)",
    )


class MetricsValues(BaseModel):
    """
    Concise representation of optimization results. Metric values for each module tested
    """
    regexp: list[str] = []
    retrieval: list[str] = []
    scoring: list[str] = []
    prediction: list[str] = []


class Trial(BaseModel):
    """
    Detailed representation of one optimization trial
    """
    module_type: str
    module_params: dict[str, Any]
    metric_name: str
    metric_value: float
    # module_name

class Trials(BaseModel):
    """
    Detailed representation of optimization results
    """
    regexp: list[Trial] = []
    retrieval: list[Trial] = []
    scoring: list[Trial] = []
    prediction: list[Trial] = []


class OptimizationInfo:
    """TODO continous IO with file system (to be able to restore the state of optimization)"""

    def __init__(self):
        self._logger = self._get_logger()

        self.artifacts = Artifacts()
        self.metrics_values = MetricsValues()
        self.trials_info = Trials()

    def log_module_optimization(
        self,
        node_type: str,
        module_type: str,
        module_config: dict,
        metric_value: float,
        metric_name: str,
        assets,
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
        trial_info = dict(
            module_type=module_type,
            metric_name=metric_name,
            metric_value=metric_value,
            **module_config,
        )
        self.cache["trials"][node_type].append(trial_info)
        self._logger.info(trial_info)
        metrics_list.append(metric_value)

    def get_best_embedder(self):
        return self.cache["best_assets"]["retrieval"]

    def get_best_test_scores(self):
        return self.cache["best_assets"]["scoring"]["test_scores"]

    def get_best_oos_scores(self):
        return self.cache["best_assets"]["scoring"]["oos_scores"]

    def dump(self):
        return {
            "metrics": self.cache["metrics"],
            "configs": self.cache["configs"],
        }

    def _get_logger(self):
        logger = logging.getLogger(__name__)

        formatter = PPrintFormatter()
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger


class PPrintFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(fmt="{asctime} - {name} - {levelname} - {message}", style="{")

    def format(self, record):
        if isinstance(record.msg, dict):
            format_msg = "module scoring results:\n"
            dct_to_str = pformat(record.msg)
            record.msg = format_msg + dct_to_str
        return super().format(record)
