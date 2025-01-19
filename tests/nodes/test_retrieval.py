import gc
import logging

import torch

from autointent.configs import InferenceNodeConfig
from autointent.nodes import InferenceNode, NodeOptimizer

from .conftest import get_context

logger = logging.getLogger(__name__)


def test_embedding_multiclass():
    context = get_context(multilabel=False)
    embedding_optimizer = get_embedding_optimizer(multilabel=False)
    embedding_optimizer.fit(context)

    for trial in context.optimization_info.trials.embedding:
        config = InferenceNodeConfig(
            node_type="embedding",
            module_name=trial.module_name,
            module_config=trial.module_params,
            load_path=trial.module_dump_dir,
        )
        node = InferenceNode.from_config(config)
        labels, distances, texts = node.module.predict(["hello", "card"])
        node.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()


def test_embedding_multilabel():
    context = get_context(multilabel=True)
    embedding_optimizer = get_embedding_optimizer(multilabel=True)
    embedding_optimizer.fit(context)

    for trial in context.optimization_info.trials.embedding:
        config = InferenceNodeConfig(
            node_type="embedding",
            module_name=trial.module_name,
            module_config=trial.module_params,
            load_path=trial.module_dump_dir,
        )
        node = InferenceNode.from_config(config)
        labels, distances, texts = node.module.predict(["hello", "card"])
        node.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()


def get_embedding_optimizer(multilabel: bool):
    metric = "retrieval_hit_rate"
    if multilabel:
        metric = metric + "_intersecting"
    embedding_optimizer_config = {
        "metric": metric,
        "node_type": "embedding",
        "search_space": [
            {
                "k": [10],
                "embedder_name": [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "avsolatorio/GIST-small-Embedding-v0",
                ],
                "module_name": "retrieval",
            },
        ],
    }

    return NodeOptimizer(**embedding_optimizer_config)
