from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import numpy as np

from autointent import Pipeline
from autointent.context.data_handler import DataHandler
from autointent.modules.scoring import KNNScorer
from tests._helpers import is_strict_labels
from tests.conftest import get_test_embedder_config

if TYPE_CHECKING:
    from autointent import Dataset


def test_base_knn(dataset: Dataset) -> None:
    data_handler = DataHandler(dataset)

    scorer = KNNScorer(k=3, weights="distance", embedder_config=get_test_embedder_config())

    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]

    # tests use the non-OOS clinc_subset, so train_labels never returns None entries.
    labels = data_handler.train_labels(0)
    assert is_strict_labels(labels)
    scorer.fit(data_handler.train_utterances(0), labels)
    predictions = scorer.predict(test_data)
    assert (
        predictions
        == np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
    ).all()

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert metadata is not None
    assert "neighbors" in metadata[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        scorer.dump(temp_dir)
        del scorer
        new_scorer = KNNScorer.load(temp_dir)
        new_predictions = new_scorer.predict(test_data)
        assert np.allclose(predictions, new_predictions)


def test_knn_in_pipeline(dataset: Dataset) -> None:
    """Test KNNScorer as part of an AutoML pipeline."""
    search_space = [
        {
            "node_type": "scoring",
            "target_metric": "scoring_roc_auc",
            "search_space": [
                {
                    "module_name": "knn",
                    "k": [3],
                    "weights": ["distance"],
                }
            ],
        },
        {"node_type": "decision", "target_metric": "decision_accuracy", "search_space": [{"module_name": "argmax"}]},
    ]

    pipeline = Pipeline.from_search_space(search_space)
    pipeline.set_config(get_test_embedder_config())
    pipeline.fit(dataset)
    predictions = pipeline.predict(["test utterance"])
    assert len(predictions) == 1
