import gc
import logging

import numpy as np
import torch

from autointent.nodes import InferenceNode
from autointent.nodes.optimization import NodeOptimizer

logger = logging.getLogger(__name__)


def test_prediction_multiclass(context_multiclass, scoring_optimizer_multiclass):
    scoring_optimizer_multiclass.fit(context_multiclass)

    prediction_optimizer_config = {
        "metric": "prediction_accuracy",
        "node_type": "prediction",
        "search_space": [
            {"module_type": "threshold", "thresh": [0.5]},
            {"module_type": "tunable", "n_trials": [None, 3]},
            {
                "module_type": "argmax",
            },
            {
                "module_type": "jinoos",
            },
        ],
    }

    prediction_optimizer = NodeOptimizer.from_dict_config(prediction_optimizer_config)

    prediction_optimizer.fit(context_multiclass)

    for trial in context_multiclass.optimization_info.trials.prediction:
        config = {
            "node_type": "prediction",
            "module_type": trial.module_type,
            "module_config": trial.module_params,
            "load_path": trial.module_dump_dir,
        }
        node = InferenceNode(**config)
        node.module.predict(np.array([[0.27486506, 0.31681463, 0.37459106], [0.2769358,  0.31536099, 0.37366978]]))
        node.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()


def test_prediction_multilabel(context_multilabel, scoring_optimizer_multilabel):
    scoring_optimizer_multilabel.fit(context_multilabel)

    prediction_optimizer_config = {
        "metric": "prediction_accuracy",
        "node_type": "prediction",
        "search_space": [
            {"module_type": "threshold", "thresh": [0.5]},
            {"module_type": "tunable", "n_trials": [None, 3]},
        ],
    }

    prediction_optimizer = NodeOptimizer.from_dict_config(prediction_optimizer_config)

    prediction_optimizer.fit(context_multilabel)

    for trial in context_multilabel.optimization_info.trials.prediction:
        config = {
            "node_type": "prediction",
            "module_type": trial.module_type,
            "module_config": trial.module_params,
            "load_path": trial.module_dump_dir,
        }
        node = InferenceNode(**config)
        node.module.predict(np.array([[0.27486506, 0.31681463, 0.37459106], [0.2769358,  0.31536099, 0.37366978]]))
        node.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
