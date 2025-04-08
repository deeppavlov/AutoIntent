import gc
import logging

import torch

from autointent.configs import InferenceNodeConfig
from autointent.nodes import InferenceNode, NodeOptimizer

from .conftest import get_context

logger = logging.getLogger(__name__)


def test_scoring_multiclass(embedding_optimizer_multiclass):
    context = get_context(multilabel=False)
    embedding_optimizer_multiclass.fit(context)

    scoring_optimizer_config = {
        "target_metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {
                "module_name": "knn",
                "k": [3],
                "weights": ["uniform", "distance", "closest"],
                "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
            },
            {
                "module_name": "linear",
                "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
            },
            {
                "module_name": "dnnc",
                "cross_encoder_config": [
                    {"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2", "train_head": False},
                    {"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2", "train_head": True},
                ],
                "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
                "k": [3],
            },
            {
                "module_name": "sklearn",
                "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
                "clf_name": ["LogisticRegression", "RandomForestClassifier"],
            },
            {
                "module_name": "description",
                "temperature": [0.05, 0.1, 0.5, 1.0],
                "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
                "encoder_type": ["bi", "cross"],
                "cross_encoder_config": ["cross-encoder/ms-marco-MiniLM-L-6-v2"],
            },
            {
                "module_name": "rerank",
                "weights": ["uniform", "distance", "closest"],
                "k": [3],
                "m": [2],
                "cross_encoder_config": ["cross-encoder/ms-marco-MiniLM-L-6-v2"],
                "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
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


def test_scoring_multilabel(embedding_optimizer_multilabel):
    context = get_context(multilabel=True)
    embedding_optimizer_multilabel.fit(context)

    scoring_optimizer_config = {
        "target_metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {
                "module_name": "knn",
                "weights": ["uniform", "distance", "closest"],
                "k": [3],
                "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
            },
            {
                "module_name": "linear",
                "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
            },
            {"module_name": "mlknn", "k": [5], "embedder_config": ["sergeyzh/rubert-tiny-turbo"]},
            {
                "module_name": "sklearn",
                "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
                "clf_name": [
                    "LogisticRegression",
                    "RandomForestClassifier",
                ],
            },
            {
                "module_name": "rerank",
                "weights": ["uniform", "distance", "closest"],
                "k": [3],
                "m": [2],
                "cross_encoder_config": ["cross-encoder/ms-marco-MiniLM-L-6-v2"],
                "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
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
