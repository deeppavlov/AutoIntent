import pathlib

import numpy as np
import pytest

from autointent.context import Context
from autointent.metrics import retrieval_map
from autointent.modules.retrieval.vectordb import VectorDBModule, retrieve_candidates


@pytest.fixture
def context(load_clinc_subset):

    def _get_context(dataset_type: str) -> Context:
        dataset_path = pathlib.Path("tests/minimal_optimization/data/").joinpath(
            f"clinc_subset_{dataset_type}.json",
        )
        return Context(
            dataset=load_clinc_subset(dataset_path),
            test_dataset=None,
            device="cpu",
            multilabel_generation_config="",
            db_dir=dataset_type,
            regex_sampling=0,
            seed=0,
        )

    return _get_context


def test_fit_creates_collection(context):
    context = context("multiclass")
    module = VectorDBModule(k=5, model_name="sergeyzh/rubert-tiny-turbo")
    module.fit(context)
    assert module.collection is not None


@pytest.mark.xfail
def test_score_returns_correct_metrics(context):
    context = context("multiclass")
    module = VectorDBModule(k=5, model_name="sergeyzh/rubert-tiny-turbo")
    module.fit(context)
    score, labels_pred = module.score(context, retrieval_map)
    assert score == 1.0
    np.testing.assert_array_equal(labels_pred, np.array([[1, 0, 1]]))


def test_get_assets_returns_correct_artifact():
    module = VectorDBModule(k=5, model_name="sergeyzh/rubert-tiny-turbo")
    artifact = module.get_assets()
    assert artifact.embedder_name == "sergeyzh/rubert-tiny-turbo"


@pytest.mark.xfail
def test_retrieve_candidates_returns_correct_labels(context):
    context = context("multiclass")
    collection = context.vector_index.create_collection("sergeyzh/rubert-tiny-turbo", context.data_handler)
    labels = retrieve_candidates(collection, 5, ["test utterance"], context.vector_index.metadata_as_labels)
    np.testing.assert_array_equal(labels, np.array([[1, 0, 1]]))
