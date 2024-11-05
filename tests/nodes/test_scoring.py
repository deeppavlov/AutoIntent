import gc
import logging

import torch

from autointent.nodes import InferenceNode
from autointent.nodes.optimization import NodeOptimizer

logger = logging.getLogger(__name__)


def test_scoring_multiclass(context, retrieval_optimizer_multiclass):
    context = context(multilabel=False)
    retrieval_optimizer_multiclass.fit(context)

    scoring_optimizer_config = {
        "metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {
                "module_type": "knn",
                "k": [3],
                "weights": ["uniform", "distance", "closest"],
                "model_name": ["sergeyzh/rubert-tiny-turbo"],
            },
            {
                "module_type": "linear",
                "model_name": ["sergeyzh/rubert-tiny-turbo"],
            },
            {
                "module_type": "dnnc",
                "cross_encoder_name": ["cross-encoder/ms-marco-MiniLM-L-6-v2"],
                "search_model_name": ["sergeyzh/rubert-tiny-turbo"],
                "k": [3],
                "train_head": [False, True],
            },
            {
                "module_type": "description",
                "temperature": [1.0, 0.5, 0.1, 0.05],
                "model_name": ["sergeyzh/rubert-tiny-turbo"],
            },
        ],
    }

    scoring_optimizer = NodeOptimizer.from_dict_config(scoring_optimizer_config)

    scoring_optimizer.fit(context)

    for trial in context.optimization_info.trials.scoring:
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


def test_scoring_multilabel(context, retrieval_optimizer_multilabel):
    context = context(multilabel=True)
    retrieval_optimizer_multilabel.fit(context)

    scoring_optimizer_config = {
        "metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {
                "module_type": "knn",
                "weights": ["uniform", "distance", "closest"],
                "k": [3],
                "model_name": ["sergeyzh/rubert-tiny-turbo"],
            },
            {
                "module_type": "linear",
                "model_name": ["sergeyzh/rubert-tiny-turbo"],
            },
            {"module_type": "mlknn", "k": [5], "model_name": ["sergeyzh/rubert-tiny-turbo"]},
        ],
    }

    scoring_optimizer = NodeOptimizer.from_dict_config(scoring_optimizer_config)

    scoring_optimizer.fit(context)

    for trial in context.optimization_info.trials.scoring:
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
