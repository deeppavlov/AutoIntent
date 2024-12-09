import gc
import logging

import torch

from autointent.configs import InferenceNodeConfig
from autointent.nodes import InferenceNode, NodeOptimizer

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
                "module_name": "knn",
                "k": [3],
                "weights": ["uniform", "distance", "closest"],
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
            },
            {
                "module_name": "linear",
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
            },
            {
                "module_name": "dnnc",
                "cross_encoder_name": ["cross-encoder/ms-marco-MiniLM-L-6-v2"],
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
                "k": [3],
                "train_head": [False, True],
            },
            {
                "module_name": "description",
                "temperature": [1.0, 0.5, 0.1, 0.05],
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
            },
            {
                "module_name": "rerank",
                "weights": ["uniform", "distance", "closest"],
                "k": [3],
                "m": [2],
                "cross_encoder_name": ["cross-encoder/ms-marco-MiniLM-L-6-v2"],
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
            },
        ],
    }

    scoring_optimizer = NodeOptimizer(**scoring_optimizer_config)

    scoring_optimizer.fit(context)

    for trial in context.optimization_info.trials.scoring:
        config = InferenceNodeConfig(
            node_type="scoring",
            module_name=trial.module_name,
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
                "module_name": "knn",
                "weights": ["uniform", "distance", "closest"],
                "k": [3],
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
            },
            {
                "module_name": "linear",
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
            },
            {"module_name": "mlknn", "k": [5], "embedder_name": ["sergeyzh/rubert-tiny-turbo"]},
            {
                "module_name": "rerank",
                "weights": ["uniform", "distance", "closest"],
                "k": [3],
                "m": [2],
                "cross_encoder_name": ["cross-encoder/ms-marco-MiniLM-L-6-v2"],
                "embedder_name": ["sergeyzh/rubert-tiny-turbo"],
            },
        ],
    }

    scoring_optimizer = NodeOptimizer(**scoring_optimizer_config)

    scoring_optimizer.fit(context)

    for trial in context.optimization_info.trials.scoring:
        config = InferenceNodeConfig(
            node_type="scoring",
            module_name=trial.module_name,
            module_config=trial.module_params,
            load_path=trial.module_dump_dir,
        )
        node = InferenceNode.from_config(config)
        scores = node.module.predict(["hello", "world"])  # noqa: F841
        node.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
