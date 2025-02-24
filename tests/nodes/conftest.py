import pytest

from autointent import Context, Dataset
from autointent.configs import DataConfig, LoggingConfig
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
        "target_metric": metric,
        "node_type": "embedding",
        "search_space": [
            {
                "k": [10],
                "embedder_config": [
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
        "target_metric": "scoring_roc_auc",
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
        "target_metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {"module_name": "linear"},
        ],
    }

    return context, NodeOptimizer(**scoring_optimizer_config)


def get_context(multilabel):
    project_dir = setup_environment()

    res = Context()
    dataset = Dataset.from_json(get_dataset_path())
    if multilabel:
        dataset = dataset.to_multilabel()
    res.set_dataset(dataset, DataConfig(scheme="ho", separate_nodes=True))
    res.configure_logging(LoggingConfig(project_dir=project_dir, dump_modules=True))
    return res
