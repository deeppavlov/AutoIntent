from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from autointent import Context, Pipeline
from autointent._callbacks import CallbackHandler, OptimizerCallback
from autointent.configs import DataConfig, FaissConfig, HPOConfig, LoggingConfig
from tests.conftest import setup_environment

if TYPE_CHECKING:
    from autointent import Dataset


class DummyCallback(OptimizerCallback):
    name = "dummy"

    def __init__(self) -> None:
        self.history: list[tuple[str, Any]] = []

    def start_run(self, **kwargs: dict[str, Any]) -> None:
        self.history.append(("start_run", kwargs))

    def start_module(self, **kwargs: dict[str, Any]) -> None:
        self.history.append(("start_module", deepcopy(kwargs)))

    def log_value(self, **kwargs: dict[str, Any]) -> None:
        self.history.append(("log_value", kwargs))

    def log_metrics(self, **kwargs: dict[str, Any]) -> None:
        metrics = kwargs["metrics"]
        metrics = {k: v for k, v in metrics.items() if not k.startswith("emissions/")}
        for metric_name, metric_value in metrics.items():
            if not isinstance(metric_value, str) and np.isnan(metric_value):
                metrics[metric_name] = None
        kwargs["metrics"] = metrics
        self.history.append(("log_metric", kwargs))

    def end_module(self, **kwargs: dict[str, Any]) -> None:
        self.history.append(("end_module", kwargs))

    def end_run(self, **kwargs: dict[str, Any]) -> None:
        self.history.append(("end_run", kwargs))

    def log_final_metrics(self, **kwargs: dict[str, Any]) -> None:
        self.history.append(("log_final_metrics", kwargs))

    def update_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        self.history.append(("update_metrics", metrics))
        return metrics

    def update_final_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        self.history.append(("update_final_metrics", metrics))
        return metrics


def test_pipeline_callbacks(dataset: Dataset) -> None:
    project_dir = setup_environment()

    search_space: list[dict[str, Any]] = [
        {
            "node_type": "embedding",
            "target_metric": "retrieval_hit_rate",
            "search_space": [
                {
                    "module_name": "retrieval",
                    "k": [5, 10],
                    "embedder_config": [
                        {
                            "n_features": 32,
                        }
                    ],
                }
            ],
        },
        {
            "node_type": "scoring",
            "target_metric": "scoring_roc_auc",
            "metrics": [
                "scoring_accuracy",
            ],
            "search_space": [
                {"module_name": "knn", "k": [1], "weights": ["uniform", "distance"]},
                {"module_name": "linear"},
            ],
        },
        {
            "node_type": "decision",
            "target_metric": "decision_accuracy",
            "metrics": [
                "decision_accuracy",
                "decision_f1",
                "decision_precision",
                "decision_recall",
                "decision_roc_auc",
            ],
            "search_space": [{"module_name": "threshold", "thresh": [0.5]}, {"module_name": "argmax"}],
        },
    ]
    pipeline_optimizer = Pipeline.from_search_space(search_space)
    context = Context()
    context.configure_logging(LoggingConfig(run_name="dummy_run_name", project_dir=project_dir, dump_modules=False))
    context.callback_handler = CallbackHandler([DummyCallback])
    context.set_dataset(dataset, DataConfig(scheme="ho"))
    context.configure_hpo(HPOConfig(n_trials=10))
    context.configure_vector_index(FaissConfig())

    pipeline_optimizer._fit(context)

    dummy_callback = context.callback_handler.callbacks[0]
    assert isinstance(dummy_callback, DummyCallback)

    assert len(dummy_callback.history) == 30
    assert dummy_callback.history[0][0] == "start_run"
    assert "run_name" in dummy_callback.history[0][1]
    assert dummy_callback.history[1:] == [
        (
            "start_module",
            {
                "module_kwargs": {
                    "embedder_config": {"n_features": 32},
                    "k": 10,
                },
                "module_name": "retrieval",
                "num": 0,
            },
        ),
        ("update_metrics", {"retrieval_hit_rate": 1.0}),
        ("log_metric", {"metrics": {"retrieval_hit_rate": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {"module_kwargs": {"embedder_config": {"n_features": 32}, "k": 5}, "module_name": "retrieval", "num": 1},
        ),
        ("update_metrics", {"retrieval_hit_rate": 1.0}),
        ("log_metric", {"metrics": {"retrieval_hit_rate": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_kwargs": {
                    "embedder_config": {
                        "analyzer": "word",
                        "binary": False,
                        "classification_prompt": None,
                        "cluster_prompt": None,
                        "default_prompt": None,
                        "dtype": "float32",
                        "lowercase": True,
                        "n_features": 32,
                        "ngram_range": (1, 2),
                        "norm": "l2",
                        "passage_prompt": None,
                        "query_prompt": None,
                        "sts_prompt": None,
                        "use_cache": True,
                    }
                },
                "module_name": "linear",
                "num": 0,
            },
        ),
        ("update_metrics", {"scoring_accuracy": 0.75, "scoring_roc_auc": 0.8333333333333334}),
        ("log_metric", {"metrics": {"scoring_accuracy": 0.75, "scoring_roc_auc": 0.8333333333333334}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_kwargs": {
                    "embedder_config": {
                        "analyzer": "word",
                        "binary": False,
                        "classification_prompt": None,
                        "cluster_prompt": None,
                        "default_prompt": None,
                        "dtype": "float32",
                        "lowercase": True,
                        "n_features": 32,
                        "ngram_range": (1, 2),
                        "norm": "l2",
                        "passage_prompt": None,
                        "query_prompt": None,
                        "sts_prompt": None,
                        "use_cache": True,
                    },
                    "k": 1,
                    "weights": "uniform",
                },
                "module_name": "knn",
                "num": 1,
            },
        ),
        ("update_metrics", {"scoring_accuracy": 0.5, "scoring_roc_auc": 0.6666666666666667}),
        ("log_metric", {"metrics": {"scoring_accuracy": 0.5, "scoring_roc_auc": 0.6666666666666667}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_kwargs": {
                    "embedder_config": {
                        "analyzer": "word",
                        "binary": False,
                        "classification_prompt": None,
                        "cluster_prompt": None,
                        "default_prompt": None,
                        "dtype": "float32",
                        "lowercase": True,
                        "n_features": 32,
                        "ngram_range": (1, 2),
                        "norm": "l2",
                        "passage_prompt": None,
                        "query_prompt": None,
                        "sts_prompt": None,
                        "use_cache": True,
                    },
                    "k": 1,
                    "weights": "distance",
                },
                "module_name": "knn",
                "num": 2,
            },
        ),
        ("update_metrics", {"scoring_accuracy": 0.5, "scoring_roc_auc": 0.6666666666666667}),
        ("log_metric", {"metrics": {"scoring_accuracy": 0.5, "scoring_roc_auc": 0.6666666666666667}}),
        ("end_module", {}),
        ("start_module", {"module_kwargs": {}, "module_name": "argmax", "num": 0}),
        (
            "update_metrics",
            {
                "decision_accuracy": 0.375,
                "decision_f1": 0.3333333333333333,
                "decision_precision": 0.2333333333333333,
                "decision_recall": 0.6,
                "decision_roc_auc": 0.7285714285714285,
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "decision_accuracy": 0.375,
                    "decision_f1": 0.3333333333333333,
                    "decision_precision": 0.2333333333333333,
                    "decision_recall": 0.6,
                    "decision_roc_auc": 0.7285714285714285,
                }
            },
        ),
        ("end_module", {}),
        ("start_module", {"module_kwargs": {"thresh": 0.5}, "module_name": "threshold", "num": 1}),
        (
            "update_metrics",
            {
                "decision_accuracy": 0.5,
                "decision_f1": 0.2533333333333333,
                "decision_precision": 0.2,
                "decision_recall": 0.35,
                "decision_roc_auc": 0.5857142857142857,
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "decision_accuracy": 0.5,
                    "decision_f1": 0.2533333333333333,
                    "decision_precision": 0.2,
                    "decision_recall": 0.35,
                    "decision_roc_auc": 0.5857142857142857,
                }
            },
        ),
        ("end_module", {}),
        ("end_run", {}),
    ]
