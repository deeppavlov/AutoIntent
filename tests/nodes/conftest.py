import pytest

from autointent import Context
from autointent.configs._optimization_cli import DataConfig, EmbedderConfig, LoggingConfig, VectorIndexConfig
from autointent.nodes.optimization import NodeOptimizer
from tests.conftest import get_dataset_path, setup_environment


@pytest.fixture
def retrieval_optimizer_multiclass():
    return get_retrieval_optimizer(multilabel=False)


@pytest.fixture
def retrieval_optimizer_multilabel():
    return get_retrieval_optimizer(multilabel=True)


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
                "embedder_name": [
                    "sentence-transformers/all-MiniLM-L6-v2",
                ],
                "module_type": "vector_db",
            },
        ],
    }

    return NodeOptimizer(**retrieval_optimizer_config)


@pytest.fixture
def scoring_optimizer_multiclass(retrieval_optimizer_multiclass):
    context = get_context(multilabel=False)
    retrieval_optimizer_multiclass.fit(context)

    scoring_optimizer_config = {
        "metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {"module_type": "linear"},
        ],
    }

    return context, NodeOptimizer(**scoring_optimizer_config)


@pytest.fixture
def scoring_optimizer_multilabel(retrieval_optimizer_multilabel):
    context = get_context(multilabel=True)
    retrieval_optimizer_multilabel.fit(context)

    scoring_optimizer_config = {
        "metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {"module_type": "linear"},
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
