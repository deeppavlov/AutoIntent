import pytest

from autointent import Context
from autointent.configs._optimization_cli import DataConfig, EmbedderConfig, LoggingConfig, VectorIndexConfig
from autointent.nodes import NodeOptimizer
from tests.conftest import get_dataset_path, setup_environment


@pytest.fixture
def embedding_optimizer_multiclass():
    return get_embedding_optimizer(multilabel=False)


@pytest.fixture
def embedding_optimizer_multilabel():
    return get_embedding_optimizer(multilabel=True)


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
                ],
                "module_name": "retrieval",
            },
        ],
    }

    return NodeOptimizer(**embedding_optimizer_config)


@pytest.fixture
def scoring_optimizer_multiclass(embedding_optimizer_multiclass):
    context = get_context(multilabel=False)
    embedding_optimizer_multiclass.fit(context)

    scoring_optimizer_config = {
        "metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {"module_name": "linear"},
        ],
    }

    return context, NodeOptimizer(**scoring_optimizer_config)


@pytest.fixture
def scoring_optimizer_multilabel(embedding_optimizer_multilabel):
    context = get_context(multilabel=True)
    embedding_optimizer_multilabel.fit(context)

    scoring_optimizer_config = {
        "metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {"module_name": "linear"},
        ],
    }

    return context, NodeOptimizer(**scoring_optimizer_config)


def get_context(multilabel):
    db_dir, dump_dir, logs_dir = setup_environment()

    res = Context()
    res.configure_data(DataConfig(get_dataset_path(), force_multilabel=multilabel))
    res.configure_logging(LoggingConfig(dirpath=logs_dir, dump_dir=dump_dir, dump_modules=True))
    res.configure_vector_index(VectorIndexConfig(db_dir=db_dir), EmbedderConfig(device="cpu"))
    return res
