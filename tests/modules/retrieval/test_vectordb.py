from autointent.modules.embedding import RetrievalEmbedding
from tests.conftest import setup_environment


def test_get_assets_returns_correct_artifact():
    dump_dir, logs_dir = setup_environment()
    module = RetrievalEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo")
    artifact = module.get_assets()
    assert artifact.embedder_name == "sergeyzh/rubert-tiny-turbo"
