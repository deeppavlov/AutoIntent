import gc
import logging

import torch

from autointent.configs._node import InferenceNodeConfig
from autointent.nodes import InferenceNode
from autointent.nodes.optimization import NodeOptimizer

from .conftest import get_context

logger = logging.getLogger(__name__)


def test_scoring_multiclass(retrieval_optimizer_multiclass):
    context = get_context(multilabel=False)
    retrieval_optimizer_multiclass.fit(context)

    scoring_optimizer_config = {
        "metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {
                "module_type": "knn",
                "k": [3],
                "weights": ["uniform", "distance", "closest"],
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
            },
            {
                "module_type": "linear",
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
            },
            {
                "module_type": "dnnc",
                "cross_encoder_name": ["cross-encoder/ms-marco-MiniLM-L-6-v2"],
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
                "k": [3],
                "train_head": [False, True],
            },
            {
                "module_type": "description",
                "temperature": [1.0, 0.5, 0.1, 0.05],
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
            },
        ],
    }

    scoring_optimizer = NodeOptimizer.from_dict_config(scoring_optimizer_config)

    scoring_optimizer.fit(context)

    for trial in context.optimization_info.trials.scoring:
        config = InferenceNodeConfig(
            node_type="scoring",
            module_type=trial.module_type,
            module_config=trial.module_params,
            load_path=trial.module_dump_dir,
        )
        node = InferenceNode.from_config(config)
        scores = node.module.predict(["hello", "world"])  # noqa: F841
        node.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()


def test_scoring_multilabel(retrieval_optimizer_multilabel):
    context = get_context(multilabel=True)
    retrieval_optimizer_multilabel.fit(context)

    scoring_optimizer_config = {
        "metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {
                "module_type": "knn",
                "weights": ["uniform", "distance", "closest"],
                "k": [3],
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
            },
            {
                "module_type": "linear",
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
            },
            {"module_type": "mlknn", "k": [5], "embedder_name": ["sergeyzh/rubert-tiny-turbo"]},
        ],
    }

    scoring_optimizer = NodeOptimizer.from_dict_config(scoring_optimizer_config)

    scoring_optimizer.fit(context)

    for trial in context.optimization_info.trials.scoring:
        config = InferenceNodeConfig(
            node_type="scoring",
            module_type=trial.module_type,
            module_config=trial.module_params,
            load_path=trial.module_dump_dir,
        )
        node = InferenceNode.from_config(config)
        scores = node.module.predict(["hello", "world"])  # noqa: F841
        node.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
