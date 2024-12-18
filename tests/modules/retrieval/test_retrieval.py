import shutil
from pathlib import Path

from autointent.modules.embedding import RetrievalEmbedding
from tests.conftest import setup_environment


def test_get_assets_returns_correct_artifact():
    db_dir, dump_dir, logs_dir = setup_environment()
    module = RetrievalEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo", db_dir=db_dir)
    artifact = module.get_assets()
    assert artifact.embedder_name == "sergeyzh/rubert-tiny-turbo"


def test_dump_and_load_preserves_model_state():
    db_dir, dump_dir, logs_dir = setup_environment()
    module = RetrievalEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo", db_dir=db_dir)

    utterances = ["hello", "goodbye", "hi", "bye", "bye", "hello", "welcome", "hi123", "hiii", "bye-bye", "bye!"]
    labels = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    module.fit(utterances, labels)

    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    module.dump(str(dump_path))

    loaded_module = RetrievalEmbedding(k=5, embedder_name="sergeyzh/rubert-tiny-turbo", db_dir=db_dir)
    loaded_module.load(str(dump_path))

    assert loaded_module.embedder_name == module.embedder_name

    shutil.rmtree(dump_path)
