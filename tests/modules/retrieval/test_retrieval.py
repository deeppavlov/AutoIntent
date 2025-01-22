import shutil

from autointent.modules.embedding import RetrievalEmbedding
from tests.conftest import setup_environment


def test_get_assets_returns_correct_artifact():
    module = RetrievalEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo")
    artifact = module.get_assets()
    assert artifact.embedder_name == "sergeyzh/rubert-tiny-turbo"


def test_dump_and_load_preserves_model_state():
    project_dir = setup_environment()
    module = RetrievalEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo")

    utterances = ["hello", "goodbye", "hi", "bye", "bye", "hello", "welcome", "hi123", "hiii", "bye-bye", "bye!"]
    labels = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    module.fit(utterances, labels)

    module.dump(project_dir)

    loaded_module = RetrievalEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo")
    loaded_module.load(project_dir)

    assert loaded_module.embedder_name == module.embedder_name

    shutil.rmtree(project_dir)
