import numpy as np
import pytest

from autointent.context import Context
from autointent.metrics import retrieval_map
from autointent.modules.retrieval.vectordb import VectorDBModule


@pytest.fixture
def context(load_clinic_subset):
    return Context(multiclass_intent_records=load_clinic_subset, multilabel_utterance_records=[],
                   test_utterance_records=[], device="cpu", mode="multiclass", multilabel_generation_config="",
                   regex_sampling=0, seed=0)


def test_fit_creates_collection(context):
    module = VectorDBModule(k=5, model_name="sergeyzh/rubert-tiny-turbo")
    module.fit(context)
    assert module.collection is not None


@pytest.mark.xfail
def test_score_returns_correct_metrics(context):
    module = VectorDBModule(k=5, model_name="sergeyzh/rubert-tiny-turbo")
    module.fit(context)
    score, labels_pred = module.score(context, retrieval_map)
    assert score == 1.0
    np.testing.assert_array_equal(labels_pred, np.array([[1, 0, 1]]))


def test_get_assets_returns_correct_artifact():
    module = VectorDBModule(k=5, model_name="sergeyzh/rubert-tiny-turbo")
    artifact = module.get_assets()
    assert artifact.embedder_name == "sergeyzh/rubert-tiny-turbo"
