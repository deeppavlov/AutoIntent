from autointent.modules.embedding import RetrievalEmbedding


def test_get_assets_returns_correct_artifact():
    module = RetrievalEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo")
    artifact = module.get_assets()
    assert artifact.embedder_name == "sergeyzh/rubert-tiny-turbo"
