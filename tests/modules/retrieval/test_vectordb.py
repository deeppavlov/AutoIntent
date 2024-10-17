import pytest

from autointent.metrics import retrieval_map
from autointent.modules.retrieval.vectordb import VectorDBModule


@pytest.mark.xfail
def test_score_returns_correct_metrics(context_multiclass):
    module = VectorDBModule(k=5, model_name="sergeyzh/rubert-tiny-turbo")
    module.fit(context_multiclass)
    score = module.score(context_multiclass, retrieval_map)
    assert score == 1.0


def test_get_assets_returns_correct_artifact():
    module = VectorDBModule(k=5, model_name="sergeyzh/rubert-tiny-turbo")
    artifact = module.get_assets()
    assert artifact.embedder_name == "sergeyzh/rubert-tiny-turbo"
