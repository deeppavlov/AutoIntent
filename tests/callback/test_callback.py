from copy import deepcopy
from typing import Any

import numpy as np

from autointent import Context, Pipeline
from autointent._callbacks import CallbackHandler, OptimizerCallback
from autointent.configs import DataConfig, HPOConfig, LoggingConfig
from tests.conftest import setup_environment


class DummyCallback(OptimizerCallback):
    name = "dummy"

    def __init__(self) -> None:
        self.history = []

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


def test_pipeline_callbacks(dataset):
    project_dir = setup_environment()

    search_space = [
        {
            "node_type": "embedding",
            "target_metric": "retrieval_hit_rate",
            "search_space": [
                {
                    "module_name": "retrieval",
                    "k": [5, 10],
                    "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
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

    pipeline_optimizer._fit(context)

    dummy_callback = context.callback_handler.callbacks[0]

    assert len(dummy_callback.history) == 122
    assert dummy_callback.history[0][0] == "start_run"
    assert "run_name" in dummy_callback.history[0][1]
    assert dummy_callback.history[1:] == [
        (
            "start_module",
            {
                "module_name": "retrieval",
                "num": 0,
                "module_kwargs": {"k": 10, "embedder_config": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        ("update_metrics", {"retrieval_hit_rate": 1.0}),
        ("log_metric", {"metrics": {"retrieval_hit_rate": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "retrieval",
                "num": 1,
                "module_kwargs": {"k": 5, "embedder_config": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        ("update_metrics", {"retrieval_hit_rate": 1.0}),
        ("log_metric", {"metrics": {"retrieval_hit_rate": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "retrieval",
                "num": 2,
                "module_kwargs": {"k": 5, "embedder_config": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        ("update_metrics", {"retrieval_hit_rate": 1.0}),
        ("log_metric", {"metrics": {"retrieval_hit_rate": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "retrieval",
                "num": 3,
                "module_kwargs": {"k": 10, "embedder_config": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        ("update_metrics", {"retrieval_hit_rate": 1.0}),
        ("log_metric", {"metrics": {"retrieval_hit_rate": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "retrieval",
                "num": 4,
                "module_kwargs": {"k": 10, "embedder_config": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        ("update_metrics", {"retrieval_hit_rate": 1.0}),
        ("log_metric", {"metrics": {"retrieval_hit_rate": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "retrieval",
                "num": 5,
                "module_kwargs": {"k": 10, "embedder_config": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        ("update_metrics", {"retrieval_hit_rate": 1.0}),
        ("log_metric", {"metrics": {"retrieval_hit_rate": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "retrieval",
                "num": 6,
                "module_kwargs": {"k": 5, "embedder_config": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        ("update_metrics", {"retrieval_hit_rate": 1.0}),
        ("log_metric", {"metrics": {"retrieval_hit_rate": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "retrieval",
                "num": 7,
                "module_kwargs": {"k": 10, "embedder_config": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        ("update_metrics", {"retrieval_hit_rate": 1.0}),
        ("log_metric", {"metrics": {"retrieval_hit_rate": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "retrieval",
                "num": 8,
                "module_kwargs": {"k": 10, "embedder_config": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        ("update_metrics", {"retrieval_hit_rate": 1.0}),
        ("log_metric", {"metrics": {"retrieval_hit_rate": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "retrieval",
                "num": 9,
                "module_kwargs": {"k": 5, "embedder_config": "sergeyzh/rubert-tiny-turbo"},
            },
        ),
        ("update_metrics", {"retrieval_hit_rate": 1.0}),
        ("log_metric", {"metrics": {"retrieval_hit_rate": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "linear",
                "num": 0,
                "module_kwargs": {
                    "embedder_config": {
                        "model_name": "sergeyzh/rubert-tiny-turbo",
                        "batch_size": 32,
                        "device": None,
                        "tokenizer_config": {"padding": True, "truncation": True, "max_length": None},
                        "trust_remote_code": False,
                        "default_prompt": None,
                        "classification_prompt": None,
                        "cluster_prompt": None,
                        "sts_prompt": None,
                        "query_prompt": None,
                        "passage_prompt": None,
                        "similarity_fn_name": "cosine",
                        "use_cache": True,
                        "freeze": True,
                    }
                },
            },
        ),
        ("update_metrics", {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}),
        ("log_metric", {"metrics": {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "knn",
                "num": 1,
                "module_kwargs": {
                    "k": 1,
                    "weights": "uniform",
                    "embedder_config": {
                        "model_name": "sergeyzh/rubert-tiny-turbo",
                        "batch_size": 32,
                        "device": None,
                        "tokenizer_config": {"padding": True, "truncation": True, "max_length": None},
                        "trust_remote_code": False,
                        "default_prompt": None,
                        "classification_prompt": None,
                        "cluster_prompt": None,
                        "sts_prompt": None,
                        "query_prompt": None,
                        "passage_prompt": None,
                        "similarity_fn_name": "cosine",
                        "use_cache": True,
                        "freeze": True,
                    },
                },
            },
        ),
        ("update_metrics", {"scoring_accuracy": 1.0, "scoring_roc_auc": 1.0}),
        ("log_metric", {"metrics": {"scoring_accuracy": 1.0, "scoring_roc_auc": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "linear",
                "num": 2,
                "module_kwargs": {
                    "embedder_config": {
                        "model_name": "sergeyzh/rubert-tiny-turbo",
                        "batch_size": 32,
                        "device": None,
                        "tokenizer_config": {"padding": True, "truncation": True, "max_length": None},
                        "trust_remote_code": False,
                        "default_prompt": None,
                        "classification_prompt": None,
                        "cluster_prompt": None,
                        "sts_prompt": None,
                        "query_prompt": None,
                        "passage_prompt": None,
                        "similarity_fn_name": "cosine",
                        "use_cache": True,
                        "freeze": True,
                    }
                },
            },
        ),
        ("update_metrics", {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}),
        ("log_metric", {"metrics": {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "linear",
                "num": 3,
                "module_kwargs": {
                    "embedder_config": {
                        "model_name": "sergeyzh/rubert-tiny-turbo",
                        "batch_size": 32,
                        "device": None,
                        "tokenizer_config": {"padding": True, "truncation": True, "max_length": None},
                        "trust_remote_code": False,
                        "default_prompt": None,
                        "classification_prompt": None,
                        "cluster_prompt": None,
                        "sts_prompt": None,
                        "query_prompt": None,
                        "passage_prompt": None,
                        "similarity_fn_name": "cosine",
                        "use_cache": True,
                        "freeze": True,
                    }
                },
            },
        ),
        ("update_metrics", {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}),
        ("log_metric", {"metrics": {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "linear",
                "num": 4,
                "module_kwargs": {
                    "embedder_config": {
                        "model_name": "sergeyzh/rubert-tiny-turbo",
                        "batch_size": 32,
                        "device": None,
                        "tokenizer_config": {"padding": True, "truncation": True, "max_length": None},
                        "trust_remote_code": False,
                        "default_prompt": None,
                        "classification_prompt": None,
                        "cluster_prompt": None,
                        "sts_prompt": None,
                        "query_prompt": None,
                        "passage_prompt": None,
                        "similarity_fn_name": "cosine",
                        "use_cache": True,
                        "freeze": True,
                    }
                },
            },
        ),
        ("update_metrics", {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}),
        ("log_metric", {"metrics": {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "knn",
                "num": 5,
                "module_kwargs": {
                    "k": 1,
                    "weights": "distance",
                    "embedder_config": {
                        "model_name": "sergeyzh/rubert-tiny-turbo",
                        "batch_size": 32,
                        "device": None,
                        "tokenizer_config": {"padding": True, "truncation": True, "max_length": None},
                        "trust_remote_code": False,
                        "default_prompt": None,
                        "classification_prompt": None,
                        "cluster_prompt": None,
                        "sts_prompt": None,
                        "query_prompt": None,
                        "passage_prompt": None,
                        "similarity_fn_name": "cosine",
                        "use_cache": True,
                        "freeze": True,
                    },
                },
            },
        ),
        ("update_metrics", {"scoring_accuracy": 1.0, "scoring_roc_auc": 1.0}),
        ("log_metric", {"metrics": {"scoring_accuracy": 1.0, "scoring_roc_auc": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "linear",
                "num": 6,
                "module_kwargs": {
                    "embedder_config": {
                        "model_name": "sergeyzh/rubert-tiny-turbo",
                        "batch_size": 32,
                        "device": None,
                        "tokenizer_config": {"padding": True, "truncation": True, "max_length": None},
                        "trust_remote_code": False,
                        "default_prompt": None,
                        "classification_prompt": None,
                        "cluster_prompt": None,
                        "sts_prompt": None,
                        "query_prompt": None,
                        "passage_prompt": None,
                        "similarity_fn_name": "cosine",
                        "use_cache": True,
                        "freeze": True,
                    }
                },
            },
        ),
        ("update_metrics", {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}),
        ("log_metric", {"metrics": {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "knn",
                "num": 7,
                "module_kwargs": {
                    "k": 1,
                    "weights": "uniform",
                    "embedder_config": {
                        "model_name": "sergeyzh/rubert-tiny-turbo",
                        "batch_size": 32,
                        "device": None,
                        "tokenizer_config": {"padding": True, "truncation": True, "max_length": None},
                        "trust_remote_code": False,
                        "default_prompt": None,
                        "classification_prompt": None,
                        "cluster_prompt": None,
                        "sts_prompt": None,
                        "query_prompt": None,
                        "passage_prompt": None,
                        "similarity_fn_name": "cosine",
                        "use_cache": True,
                        "freeze": True,
                    },
                },
            },
        ),
        ("update_metrics", {"scoring_accuracy": 1.0, "scoring_roc_auc": 1.0}),
        ("log_metric", {"metrics": {"scoring_accuracy": 1.0, "scoring_roc_auc": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "linear",
                "num": 8,
                "module_kwargs": {
                    "embedder_config": {
                        "model_name": "sergeyzh/rubert-tiny-turbo",
                        "batch_size": 32,
                        "device": None,
                        "tokenizer_config": {"padding": True, "truncation": True, "max_length": None},
                        "trust_remote_code": False,
                        "default_prompt": None,
                        "classification_prompt": None,
                        "cluster_prompt": None,
                        "sts_prompt": None,
                        "query_prompt": None,
                        "passage_prompt": None,
                        "similarity_fn_name": "cosine",
                        "use_cache": True,
                        "freeze": True,
                    }
                },
            },
        ),
        ("update_metrics", {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}),
        ("log_metric", {"metrics": {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}}),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_name": "linear",
                "num": 9,
                "module_kwargs": {
                    "embedder_config": {
                        "model_name": "sergeyzh/rubert-tiny-turbo",
                        "batch_size": 32,
                        "device": None,
                        "tokenizer_config": {"padding": True, "truncation": True, "max_length": None},
                        "trust_remote_code": False,
                        "default_prompt": None,
                        "classification_prompt": None,
                        "cluster_prompt": None,
                        "sts_prompt": None,
                        "query_prompt": None,
                        "passage_prompt": None,
                        "similarity_fn_name": "cosine",
                        "use_cache": True,
                        "freeze": True,
                    }
                },
            },
        ),
        ("update_metrics", {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}),
        ("log_metric", {"metrics": {"scoring_accuracy": 0.75, "scoring_roc_auc": 1.0}}),
        ("end_module", {}),
        ("start_module", {"module_name": "argmax", "num": 0, "module_kwargs": {}}),
        (
            "update_metrics",
            {
                "decision_accuracy": 0.5,
                "decision_f1": 0.6133333333333333,
                "decision_precision": 0.55,
                "decision_recall": 0.8,
                "decision_roc_auc": 0.8428571428571429,
            },
        ),
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
        ("start_module", {"module_name": "threshold", "num": 1, "module_kwargs": {"thresh": 0.5}}),
        (
            "update_metrics",
            {
                "decision_accuracy": 1.0,
                "decision_f1": 1.0,
                "decision_precision": 1.0,
                "decision_recall": 1.0,
                "decision_roc_auc": 1.0,
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "decision_accuracy": 1.0,
                    "decision_f1": 1.0,
                    "decision_precision": 1.0,
                    "decision_recall": 1.0,
                    "decision_roc_auc": 1.0,
                }
            },
        ),
        ("end_module", {}),
        ("start_module", {"module_name": "threshold", "num": 2, "module_kwargs": {"thresh": 0.5}}),
        (
            "update_metrics",
            {
                "decision_accuracy": 1.0,
                "decision_f1": 1.0,
                "decision_precision": 1.0,
                "decision_recall": 1.0,
                "decision_roc_auc": 1.0,
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "decision_accuracy": 1.0,
                    "decision_f1": 1.0,
                    "decision_precision": 1.0,
                    "decision_recall": 1.0,
                    "decision_roc_auc": 1.0,
                }
            },
        ),
        ("end_module", {}),
        ("start_module", {"module_name": "argmax", "num": 3, "module_kwargs": {}}),
        (
            "update_metrics",
            {
                "decision_accuracy": 0.5,
                "decision_f1": 0.6133333333333333,
                "decision_precision": 0.55,
                "decision_recall": 0.8,
                "decision_roc_auc": 0.8428571428571429,
            },
        ),
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
        ("start_module", {"module_name": "argmax", "num": 4, "module_kwargs": {}}),
        (
            "update_metrics",
            {
                "decision_accuracy": 0.5,
                "decision_f1": 0.6133333333333333,
                "decision_precision": 0.55,
                "decision_recall": 0.8,
                "decision_roc_auc": 0.8428571428571429,
            },
        ),
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
        ("start_module", {"module_name": "argmax", "num": 5, "module_kwargs": {}}),
        (
            "update_metrics",
            {
                "decision_accuracy": 0.5,
                "decision_f1": 0.6133333333333333,
                "decision_precision": 0.55,
                "decision_recall": 0.8,
                "decision_roc_auc": 0.8428571428571429,
            },
        ),
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
        ("start_module", {"module_name": "threshold", "num": 6, "module_kwargs": {"thresh": 0.5}}),
        (
            "update_metrics",
            {
                "decision_accuracy": 1.0,
                "decision_f1": 1.0,
                "decision_precision": 1.0,
                "decision_recall": 1.0,
                "decision_roc_auc": 1.0,
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "decision_accuracy": 1.0,
                    "decision_f1": 1.0,
                    "decision_precision": 1.0,
                    "decision_recall": 1.0,
                    "decision_roc_auc": 1.0,
                }
            },
        ),
        ("end_module", {}),
        ("start_module", {"module_name": "argmax", "num": 7, "module_kwargs": {}}),
        (
            "update_metrics",
            {
                "decision_accuracy": 0.5,
                "decision_f1": 0.6133333333333333,
                "decision_precision": 0.55,
                "decision_recall": 0.8,
                "decision_roc_auc": 0.8428571428571429,
            },
        ),
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
        ("start_module", {"module_name": "argmax", "num": 8, "module_kwargs": {}}),
        (
            "update_metrics",
            {
                "decision_accuracy": 0.5,
                "decision_f1": 0.6133333333333333,
                "decision_precision": 0.55,
                "decision_recall": 0.8,
                "decision_roc_auc": 0.8428571428571429,
            },
        ),
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
        ("start_module", {"module_name": "threshold", "num": 9, "module_kwargs": {"thresh": 0.5}}),
        (
            "update_metrics",
            {
                "decision_accuracy": 1.0,
                "decision_f1": 1.0,
                "decision_precision": 1.0,
                "decision_recall": 1.0,
                "decision_roc_auc": 1.0,
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "decision_accuracy": 1.0,
                    "decision_f1": 1.0,
                    "decision_precision": 1.0,
                    "decision_recall": 1.0,
                    "decision_roc_auc": 1.0,
                }
            },
        ),
        ("end_module", {}),
        ("end_run", {}),
    ]
