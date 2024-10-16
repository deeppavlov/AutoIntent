import pytest

from autointent.nodes.optimization import NodeOptimizer


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
                "model_name": [
                    "sentence-transformers/all-MiniLM-L6-v2",
                ],
                "module_type": "vector_db",
            }
        ],
    }

    return NodeOptimizer.from_dict_config(retrieval_optimizer_config)


@pytest.fixture
def scoring_optimizer_multiclass(context_multiclass, retrieval_optimizer_multiclass):
    retrieval_optimizer_multiclass.fit(context_multiclass)

    scoring_optimizer_config = {
        "metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {"module_type": "linear"},
        ],
    }

    return NodeOptimizer.from_dict_config(scoring_optimizer_config)


@pytest.fixture
def scoring_optimizer_multilabel(context_multilabel, retrieval_optimizer_multilabel):
    retrieval_optimizer_multilabel.fit(context_multilabel)

    scoring_optimizer_config = {
        "metric": "scoring_roc_auc",
        "node_type": "scoring",
        "search_space": [
            {"module_type": "linear"},
        ],
    }

    return NodeOptimizer.from_dict_config(scoring_optimizer_config)
