import gc
import logging

import torch

from autointent.configs.node import InferenceNodeConfig
from autointent.nodes import InferenceNode, NodeOptimizer

logger = logging.getLogger(__name__)


def test_retrieval_multiclass(context):
    context = context(multilabel=False)
    retrieval_optimizer = get_retrieval_optimizer(multilabel=False)
    retrieval_optimizer.fit(context)

    for trial in context.optimization_info.trials.retrieval:
        config = InferenceNodeConfig(
            node_type="retrieval",
            module_type=trial.module_type,
            module_config=trial.module_params,
            load_path=trial.module_dump_dir,
        )
        node = InferenceNode.from_config(config)
        labels, distances, texts = node.module.predict(["hello", "card"])
        node.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()


def test_retrieval_multilabel(context):
    context = context(multilabel=True)
    retrieval_optimizer = get_retrieval_optimizer(multilabel=True)
    retrieval_optimizer.fit(context)

    for trial in context.optimization_info.trials.retrieval:
        config = InferenceNodeConfig(
            node_type="retrieval",
            module_type=trial.module_type,
            module_config=trial.module_params,
            load_path=trial.module_dump_dir,
        )
        node = InferenceNode.from_config(config)
        labels, distances, texts = node.module.predict(["hello", "card"])
        node.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()


def get_retrieval_optimizer(multilabel: bool):
    metric = "retrieval_hit_rate"
    if multilabel:
        metric = metric + "_intersecting"
    retrieval_optimizer_config = {
        "metric": metric,
        "node_type": "retrieval",
        "search_space": [
            {
                "k": [10],
                "model_name": [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "avsolatorio/GIST-small-Embedding-v0",
                ],
                "module_type": "vector_db",
            }
        ],
    }

    return NodeOptimizer.from_dict_config(retrieval_optimizer_config)
