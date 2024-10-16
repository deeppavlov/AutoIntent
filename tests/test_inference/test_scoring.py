import gc
import logging

import torch

from autointent.nodes import InferenceNode
from autointent.nodes.optimization import NodeOptimizer

logger = logging.getLogger(__name__)


def test_scoring_multiclass(context_multiclass, retrieval_optimizer_multiclass):
    retrieval_optimizer_multiclass.fit(context_multiclass)

    scoring_optimizer_config = {
        "metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {"k": [3], "module_type": "knn", "weights": ["uniform", "distance", "closest"]},
            {"module_type": "linear"},
            {
                "module_type": "dnnc",
                "model_name": ["cross-encoder/ms-marco-MiniLM-L-6-v2"],
                "k": [3],
                "train_head": [False, True],
            },
        ],
    }

    scoring_optimizer = NodeOptimizer.from_dict_config(scoring_optimizer_config)

    scoring_optimizer.fit(context_multiclass)

    for trial in context_multiclass.optimization_info.trials.scoring:
        config = {
            "node_type": "scoring",
            "module_type": trial.module_type,
            "module_config": trial.module_params,
            "load_path": trial.module_dump_dir,
        }
        node = InferenceNode(**config)
        scores = node.module.predict(["hello", "world"])  # noqa: F841
        node.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()


def test_scoring_multilabel(context_multilabel, retrieval_optimizer_multilabel):
    retrieval_optimizer_multilabel.fit(context_multilabel)

    scoring_optimizer_config = {
        "metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {"k": [3], "module_type": "knn", "weights": ["uniform", "distance", "closest"]},
            {"module_type": "linear"},
            {"module_type": "mlknn", "k": [5]},
        ],
    }

    scoring_optimizer = NodeOptimizer.from_dict_config(scoring_optimizer_config)

    scoring_optimizer.fit(context_multilabel)

    for trial in context_multilabel.optimization_info.trials.scoring:
        config = {
            "node_type": "scoring",
            "module_type": trial.module_type,
            "module_config": trial.module_params,
            "load_path": trial.module_dump_dir,
        }
        node = InferenceNode(**config)
        scores = node.module.predict(["hello", "world"])  # noqa: F841
        node.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
