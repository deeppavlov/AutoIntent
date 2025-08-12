from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sentence_transformers import SentenceTransformer

from autointent import Embedder
from autointent.configs import EmbedderConfig


def test_dump_load():
    model = SentenceTransformer("sergeyzh/rubert-tiny-turbo")

    with TemporaryDirectory() as tmp_dir:
        model.save(str(Path(tmp_dir) / "weights"))
        embedder = Embedder(EmbedderConfig(model_name=str(Path(tmp_dir) / "weights")))
        predictions = embedder.embed(["hi!"])
        embedder.dump(Path(tmp_dir) / "embedder")
        embedder_loaded = Embedder.load(Path(tmp_dir) / "embedder")
        predictions_after = embedder_loaded.embed(["hi!"])

    np.testing.assert_almost_equal(predictions_after, predictions, decimal=4)
