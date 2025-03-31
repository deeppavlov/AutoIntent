from copy import deepcopy
from typing import Any

import numpy as np

from autointent import Context, Pipeline
from autointent._callbacks import CallbackHandler, OptimizerCallback
from autointent.configs import DataConfig, LoggingConfig
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

    pipeline_optimizer._fit(context, "brute")

    dummy_callback = context.callback_handler.callbacks[0]

    assert len(dummy_callback.history) == 23
    assert dummy_callback.history[0][0] == "start_run"
    assert "run_name" in dummy_callback.history[0][1]
    assert dummy_callback.history[1:] == [
        (
            "start_module",
            {
                "module_kwargs": {"embedder_config": "sergeyzh/rubert-tiny-turbo", "k": 5},
                "module_name": "retrieval",
                "num": 0,
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "retrieval_hit_rate": 1.0,
                    "emissions/cpu_count": 4,
                    "emissions/cpu_energy": 0.00010161774583333391,
                    "emissions/cpu_power": 140.0,
                    "emissions/duration": -1743448871.1596549,
                    "emissions/emissions": 5.0399069483124444e-05,
                    "emissions/emissions_rate": 1.9287459668038395e-05,
                    "emissions/energy_consumed": 0.00010586762294121488,
                    "emissions/gpu_energy": 0,
                    "emissions/gpu_power": 0.0,
                    "emissions/latitude": 29.4227,
                    "emissions/longitude": -98.4927,
                    "emissions/pue": 1.0,
                    "emissions/ram_energy": 4.249877107880973e-06,
                    "emissions/ram_power": 5.8557257652282715,
                    "emissions/ram_total_size": 15.61526870727539,
                }
            },
        ),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_kwargs": {"embedder_config": "sergeyzh/rubert-tiny-turbo", "k": 10},
                "module_name": "retrieval",
                "num": 1,
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "retrieval_hit_rate": 1.0,
                    "emissions/cpu_count": 4,
                    "emissions/cpu_energy": 4.869346412777829e-05,
                    "emissions/cpu_power": 140.0,
                    "emissions/duration": -1743448872.5204384,
                    "emissions/emissions": 2.415026452774112e-05,
                    "emissions/emissions_rate": 1.9287193027812115e-05,
                    "emissions/energy_consumed": 5.072972825043102e-05,
                    "emissions/gpu_energy": 0,
                    "emissions/gpu_power": 0.0,
                    "emissions/latitude": 29.4227,
                    "emissions/longitude": -98.4927,
                    "emissions/pue": 1.0,
                    "emissions/ram_energy": 2.036264122652723e-06,
                    "emissions/ram_power": 5.8557257652282715,
                    "emissions/ram_total_size": 15.61526870727539,
                }
            },
        ),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_kwargs": {
                    "embedder_config": {
                        "batch_size": 32,
                        "classifier_prompt": None,
                        "cluster_prompt": None,
                        "default_prompt": None,
                        "device": None,
                        "tokenizer_config": {"max_length": None, "truncation": True, "padding": True},
                        "model_name": "sergeyzh/rubert-tiny-turbo",
                        "passage_prompt": None,
                        "query_prompt": None,
                        "sts_prompt": None,
                        "use_cache": False,
                    },
                    "k": 1,
                    "weights": "uniform",
                },
                "module_name": "knn",
                "num": 0,
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "emissions/cpu_count": 4,
                    "emissions/cpu_energy": 5.6463417383332314e-05,
                    "emissions/cpu_power": 140.0,
                    "emissions/duration": -1743448872.3206754,
                    "emissions/emissions": 2.800390409611645e-05,
                    "emissions/emissions_rate": 1.9287278349336493e-05,
                    "emissions/energy_consumed": 5.882463287784108e-05,
                    "emissions/gpu_energy": 0,
                    "emissions/gpu_power": 0.0,
                    "emissions/latitude": 29.4227,
                    "emissions/longitude": -98.4927,
                    "emissions/pue": 1.0,
                    "emissions/ram_energy": 2.3612154945087627e-06,
                    "emissions/ram_power": 5.8557257652282715,
                    "emissions/ram_total_size": 15.61526870727539,
                    "scoring_accuracy": 1.0,
                    "scoring_roc_auc": 1.0,
                }
            },
        ),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_kwargs": {
                    "embedder_config": {
                        "batch_size": 32,
                        "classifier_prompt": None,
                        "cluster_prompt": None,
                        "default_prompt": None,
                        "device": None,
                        "tokenizer_config": {"max_length": None, "truncation": True, "padding": True},
                        "model_name": "sergeyzh/rubert-tiny-turbo",
                        "passage_prompt": None,
                        "query_prompt": None,
                        "sts_prompt": None,
                        "use_cache": False,
                    },
                    "k": 1,
                    "weights": "distance",
                },
                "module_name": "knn",
                "num": 1,
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "emissions/cpu_count": 4,
                    "emissions/cpu_energy": 5.822425206111052e-05,
                    "emissions/cpu_power": 140.0,
                    "emissions/duration": -1743448872.2755105,
                    "emissions/emissions": 2.8877239803966572e-05,
                    "emissions/emissions_rate": 1.9287335939110744e-05,
                    "emissions/energy_consumed": 6.065915038715212e-05,
                    "emissions/gpu_energy": 0,
                    "emissions/gpu_power": 0.0,
                    "emissions/latitude": 29.4227,
                    "emissions/longitude": -98.4927,
                    "emissions/pue": 1.0,
                    "emissions/ram_energy": 2.434898326041607e-06,
                    "emissions/ram_power": 5.8557257652282715,
                    "emissions/ram_total_size": 15.61526870727539,
                    "scoring_accuracy": 1.0,
                    "scoring_roc_auc": 1.0,
                }
            },
        ),
        ("end_module", {}),
        (
            "start_module",
            {
                "module_kwargs": {
                    "embedder_config": {
                        "batch_size": 32,
                        "classifier_prompt": None,
                        "cluster_prompt": None,
                        "default_prompt": None,
                        "device": None,
                        "tokenizer_config": {"max_length": None, "truncation": True, "padding": True},
                        "model_name": "sergeyzh/rubert-tiny-turbo",
                        "passage_prompt": None,
                        "query_prompt": None,
                        "sts_prompt": None,
                        "use_cache": False,
                    },
                },
                "module_name": "linear",
                "num": 0,
            },
        ),
        (
            "log_metric",
            {
                "metrics": {
                    "emissions/cpu_count": 4,
                    "emissions/cpu_energy": 7.098722081111014e-05,
                    "emissions/cpu_power": 140.0,
                    "emissions/duration": -1743448871.9471838,
                    "emissions/emissions": 3.520725453309416e-05,
                    "emissions/emissions_rate": 1.928735273284849e-05,
                    "emissions/energy_consumed": 7.39558961292537e-05,
                    "emissions/gpu_energy": 0,
                    "emissions/gpu_power": 0.0,
                    "emissions/latitude": 29.4227,
                    "emissions/longitude": -98.4927,
                    "emissions/pue": 1.0,
                    "emissions/ram_energy": 2.968675318143547e-06,
                    "emissions/ram_power": 5.8557257652282715,
                    "emissions/ram_total_size": 15.61526870727539,
                    "scoring_accuracy": 0.75,
                    "scoring_roc_auc": 1.0,
                }
            },
        ),
        ("end_module", {}),
        ("start_module", {"module_kwargs": {"thresh": 0.5}, "module_name": "threshold", "num": 0}),
        (
            "log_metric",
            {
                "metrics": {
                    "decision_accuracy": 0.5,
                    "decision_f1": 0.6133333333333333,
                    "decision_precision": 0.55,
                    "decision_recall": 0.8,
                    "decision_roc_auc": 0.8428571428571429,
                    "emissions/cpu_count": 4,
                    "emissions/cpu_energy": 4.606736166661247e-07,
                    "emissions/cpu_power": 140.0,
                    "emissions/duration": -1743448873.7607834,
                    "emissions/emissions": 2.2824627343844443e-07,
                    "emissions/emissions_rate": 1.9233611973570262e-05,
                    "emissions/energy_consumed": 4.794511220531496e-07,
                    "emissions/gpu_energy": 0,
                    "emissions/gpu_power": 0.0,
                    "emissions/latitude": 29.4227,
                    "emissions/longitude": -98.4927,
                    "emissions/pue": 1.0,
                    "emissions/ram_energy": 1.8777505387024907e-08,
                    "emissions/ram_power": 5.8557257652282715,
                    "emissions/ram_total_size": 15.61526870727539,
                }
            },
        ),
        ("end_module", {}),
        ("start_module", {"module_kwargs": {}, "module_name": "argmax", "num": 0}),
        (
            "log_metric",
            {
                "metrics": {
                    "decision_accuracy": 0.5,
                    "decision_f1": 0.6133333333333333,
                    "decision_precision": 0.55,
                    "decision_recall": 0.8,
                    "decision_roc_auc": 0.8428571428571429,
                    "emissions/cpu_count": 4,
                    "emissions/cpu_energy": 4.5328064444384436e-07,
                    "emissions/cpu_power": 140.0,
                    "emissions/duration": -1743448873.7610447,
                    "emissions/emissions": 2.2460812015265647e-07,
                    "emissions/emissions_rate": 1.9237634797985535e-05,
                    "emissions/energy_consumed": 4.7180886507872054e-07,
                    "emissions/gpu_energy": 0,
                    "emissions/gpu_power": 0.0,
                    "emissions/latitude": 29.4227,
                    "emissions/longitude": -98.4927,
                    "emissions/pue": 1.0,
                    "emissions/ram_energy": 1.852822063487629e-08,
                    "emissions/ram_power": 5.8557257652282715,
                    "emissions/ram_total_size": 15.61526870727539,
                }
            },
        ),
        ("end_module", {}),
        ("end_run", {}),
    ]
