import gc
import logging

import numpy as np
import torch

from autointent.configs import InferenceNodeConfig
from autointent.nodes import InferenceNode, NodeOptimizer

logger = logging.getLogger(__name__)


def test_decision_multiclass(scoring_optimizer_multiclass):
    context, scoring_optimizer_multiclass = scoring_optimizer_multiclass
    scoring_optimizer_multiclass.fit(context)

    decision_optimizer_config = {
        "metric": "decision_accuracy",
        "node_type": "decision",
        "search_space": [
            {"module_name": "threshold", "thresh": [0.5]},
            {"module_name": "tunable", "n_trials": [None, 3]},
            {
                "module_name": "argmax",
            },
            {
                "module_name": "jinoos",
            },
        ],
    }

    decision_optimizer = NodeOptimizer(**decision_optimizer_config)

    decision_optimizer.fit(context)

    for trial in context.optimization_info.trials.decision:
        config = InferenceNodeConfig(
            node_type="decision",
            module_name=trial.module_name,
            module_config=trial.module_params,
            load_path=trial.module_dump_dir,
        )
        node = InferenceNode.from_config(config)
        node.module.predict(
            np.array([[0.27486506, 0.31681463, 0.37459106, 0.532], [0.2769358, 0.31536099, 0.37366978, 0.532]])
        )
        node.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()


def test_decision_multilabel(scoring_optimizer_multilabel):
    context, scoring_optimizer_multilabel = scoring_optimizer_multilabel
    scoring_optimizer_multilabel.fit(context)

    decision_optimizer_config = {
        "metric": "decision_accuracy",
        "node_type": "decision",
        "search_space": [
            {"module_name": "threshold", "thresh": [0.5]},
            {"module_name": "tunable", "n_trials": [None, 3]},
            {"module_name": "adaptive"},
        ],
    }

    decision_optimizer = NodeOptimizer(**decision_optimizer_config)

    decision_optimizer.fit(context)

    for trial in context.optimization_info.trials.decision:
        config = InferenceNodeConfig(
            node_type="decision",
            module_name=trial.module_name,
            module_config=trial.module_params,
            load_path=trial.module_dump_dir,
        )
        node = InferenceNode.from_config(config)
        node.module.predict(
            np.array(
                [[0.27486506, 0.31681463, 0.37459106, 0.37459106], [0.2769358, 0.31536099, 0.37366978, 0.37459106]]
            )
        )
        node.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
