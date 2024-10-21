import pytest

from autointent.metrics import retrieval_map
from autointent.modules.retrieval.vectordb import VectorDBModule


@pytest.mark.xfail
def test_score_returns_correct_metrics(context):
    context = context("multiclass")
    module = VectorDBModule(k=5, model_name="sergeyzh/rubert-tiny-turbo")
    module.fit(context)
    score = module.score(context, retrieval_map)
    assert score == 1.0


def test_get_assets_returns_correct_artifact():
    module = VectorDBModule(k=5, model_name="sergeyzh/rubert-tiny-turbo")
    artifact = module.get_assets()
    assert artifact.embedder_name == "sergeyzh/rubert-tiny-turbo"
