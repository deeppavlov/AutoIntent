from typing import Any

import numpy as np

from autointent import Context, Pipeline
from autointent._callbacks import CallbackHandler, OptimizerCallback
from autointent.configs import LoggingConfig, VectorIndexConfig
from tests.conftest import setup_environment


class DummyCallback(OptimizerCallback):
    name = "dummy"

    def __init__(self) -> None:
        self.history = []

    def start_run(self, **kwargs: dict[str, Any]) -> None:
        self.history.append(("start_run", kwargs))

    def start_module(self, **kwargs: dict[str, Any]) -> None:
        self.history.append(("start_module", kwargs))

    def log_value(self, **kwargs: dict[str, Any]) -> None:
        self.history.append(("log_value", kwargs))

    def log_metrics(self, **kwargs: dict[str, Any]) -> None:
        metrics = kwargs["metrics"]
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


def test_pipeline_callbacks(dataset):
    project_dir = setup_environment()

    search_space = [
        {
            "node_type": "embedding",
            "target_metric": "retrieval_hit_rate",
            "metrics": ["retrieval_map", "retrieval_mrr", "retrieval_ndcg", "retrieval_precision"],
            "search_space": [
                {
                    "module_name": "retrieval",
                    "k": [5, 10],
                    "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
                }
            ],
        },
        {
            "node_type": "scoring",
            "target_metric": "scoring_roc_auc",
            "metrics": [
                "scoring_accuracy",
                "scoring_f1",
                "scoring_log_likelihood",
                "scoring_precision",
                "scoring_recall",
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
    context.configure_vector_index(VectorIndexConfig(save_db=True))
    context.configure_logging(LoggingConfig(run_name="dummy_run_name", project_dir=project_dir, dump_modules=False))
    context.callback_handler = CallbackHandler([DummyCallback])
    context.set_dataset(dataset)

    pipeline_optimizer._fit(context)

    dummy_callback = context.callback_handler.callbacks[0]

    assert len(dummy_callback.history) == 23
    assert dummy_callback.history[0][0] == "start_run"
    assert "run_name" in dummy_callback.history[0][1]
    assert dummy_callback.history[1:] == [
        (
            "start_module",
            {
                "module_name": "retrieval",
                "num": 0,
                "module_kwargs": {"k": 5, "embedder_name": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "retrieval_hit_rate": 1.0,
                    "retrieval_map": 0.9875,
                    "retrieval_mrr": 1.0,
                    "retrieval_ndcg": 0.9957230204891719,
                    "retrieval_precision": 0.8500000000000001,
                }
            },
        ),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "retrieval",
                "num": 1,
                "module_kwargs": {"k": 10, "embedder_name": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "retrieval_hit_rate": 1.0,
                    "retrieval_map": 0.9816666666666667,
                    "retrieval_mrr": 1.0,
                    "retrieval_ndcg": 0.9936857382141969,
                    "retrieval_precision": 0.44999999999999996,
                }
            },
        ),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "knn",
                "num": 0,
                "module_kwargs": {"k": 1, "weights": "uniform", "embedder_name": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "scoring_accuracy": 1.0,
                    "scoring_f1": 1.0,
                    "scoring_log_likelihood": 0.0,
                    "scoring_precision": 1.0,
                    "scoring_recall": 1.0,
                    "scoring_roc_auc": 1.0,
                }
            },
        ),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "knn",
                "num": 1,
                "module_kwargs": {"k": 1, "weights": "distance", "embedder_name": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "scoring_accuracy": 1.0,
                    "scoring_f1": 1.0,
                    "scoring_log_likelihood": 0.0,
                    "scoring_precision": 1.0,
                    "scoring_recall": 1.0,
                    "scoring_roc_auc": 1.0,
                }
            },
        ),
        ("end_module", {}),
        (
            "start_module",
            {"module_name": "linear", "num": 0, "module_kwargs": {"embedder_name": "sergeyzh/rubert-tiny-turbo"}},
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "scoring_accuracy": 0.75,
                    "scoring_f1": 0.6666666666666666,
                    "scoring_log_likelihood": -0.439819,
                    "scoring_precision": 0.625,
                    "scoring_recall": 0.75,
                    "scoring_roc_auc": 1.0,
                }
            },
        ),
        ("end_module", {}),
        ("start_module", {"module_name": "threshold", "num": 0, "module_kwargs": {"thresh": 0.5}}),
        (
            "log_metric",
            {
                "metrics": {
                    "decision_accuracy": 0.5,
                    "decision_f1": 0.6133333333333333,
                    "decision_precision": 0.55,
                    "decision_recall": 0.8,
                    "decision_roc_auc": 0.8428571428571429,
                }
            },
        ),
        ("end_module", {}),
        ("start_module", {"module_name": "argmax", "num": 0, "module_kwargs": {}}),
        (
            "log_metric",
            {
                "metrics": {
                    "decision_accuracy": 0.5,
                    "decision_f1": 0.6133333333333333,
                    "decision_precision": 0.55,
                    "decision_recall": 0.8,
                    "decision_roc_auc": 0.8428571428571429,
                }
            },
        ),
        ("end_module", {}),
        ("end_run", {}),
    ]
