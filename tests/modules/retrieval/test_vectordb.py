from autointent.modules.retrieval.vectordb import VectorDBModule


def test_get_assets_returns_correct_artifact(tmp_path):
    module = VectorDBModule(k=5, embedder_name="sergeyzh/rubert-tiny-turbo", db_dir=str(tmp_path))
    artifact = module.get_assets()
    assert artifact.embedder_name == "sergeyzh/rubert-tiny-turbo"
