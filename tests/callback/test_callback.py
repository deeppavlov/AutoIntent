from pathlib import Path
from typing import Any

import numpy as np

from autointent import Context, Dataset, Pipeline
from autointent._callbacks.base import CallbackHandler, OptimizerCallback
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

    def end_module(self, **kwargs: dict[str, Any]) -> None:
        self.history.append(("end_module", kwargs))

    def end_run(self, **kwargs: dict[str, Any]) -> None:
        self.history.append(("end_run", kwargs))

    def log_final_metrics(self, **kwargs: dict[str, Any]) -> None:
        self.history.append(("log_final_metrics", kwargs))


def test_pipeline_callbacks():
    db_dir, dump_dir, logs_dir = setup_environment()

    dataset = Dataset.from_hub("AutoIntent/clinc150_subset")
    search_space = [
        {
            "node_type": "embedding",
            "metric": "retrieval_hit_rate",
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
            "metric": "scoring_roc_auc",
            "search_space": [
                {"module_name": "knn", "k": [1], "weights": ["uniform", "distance"]},
                {"module_name": "linear"},
            ],
        },
        {
            "node_type": "decision",
            "metric": "decision_accuracy",
            "search_space": [{"module_name": "threshold", "thresh": [0.5]}, {"module_name": "argmax"}],
        },
    ]
    pipeline_optimizer = Pipeline.from_search_space(search_space)
    context = Context()
    context.configure_vector_index(VectorIndexConfig(db_dir=Path(db_dir).resolve(), save_db=True))
    context.configure_logging(
        LoggingConfig(run_name="dummy_run_name", dirpath=Path(logs_dir).resolve(), dump_modules=False)
    )
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
        ("log_value", {"retrieval_hit_rate": 1.0}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "retrieval",
                "num": 1,
                "module_kwargs": {"k": 10, "embedder_name": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        ("log_value", {"retrieval_hit_rate": 1.0}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "knn",
                "num": 0,
                "module_kwargs": {"k": 1, "weights": "uniform", "embedder_name": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        ("log_value", {"scoring_roc_auc": np.float64(1.0)}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "knn",
                "num": 1,
                "module_kwargs": {"k": 1, "weights": "distance", "embedder_name": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        ("log_value", {"scoring_roc_auc": np.float64(1.0)}),
        ("end_module", {}),
        (
            "start_module",
            {"module_name": "linear", "num": 0, "module_kwargs": {"embedder_name": "sergeyzh/rubert-tiny-turbo"}},
        ),
        ("log_value", {"scoring_roc_auc": np.float64(1.0)}),
        ("end_module", {}),
        ("start_module", {"module_name": "threshold", "num": 0, "module_kwargs": {"thresh": 0.5}}),
        ("log_value", {"decision_accuracy": np.float64(0.75)}),
        ("end_module", {}),
        ("start_module", {"module_name": "argmax", "num": 0, "module_kwargs": {}}),
        ("log_value", {"decision_accuracy": np.float64(0.75)}),
        ("end_module", {}),
        ("end_run", {}),
    ]
