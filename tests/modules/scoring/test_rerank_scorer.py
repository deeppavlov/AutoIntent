from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from autointent import Pipeline
from autointent.context.data_handler import DataHandler
from autointent.modules.scoring import RerankScorer

if TYPE_CHECKING:
    from autointent import Dataset
    from autointent.custom_types import ListOfLabels

pytest.importorskip("sentence_transformers")


def test_base_rerank_scorer(dataset: Dataset) -> None:
    data_handler = DataHandler(dataset)

    scorer = RerankScorer(
        k=3,
        weights="distance",
        embedder_config="sergeyzh/rubert-tiny-turbo",
        m=2,
        cross_encoder_config="cross-encoder/ms-marco-MiniLM-L6-v2",
    )

    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]

    # cast: tests use the non-OOS clinc_subset, so train_labels never returns None entries.
    scorer.fit(data_handler.train_utterances(0), cast("ListOfLabels", data_handler.train_labels(0)))
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
        new_scorer = RerankScorer.load(temp_dir)
        new_predictions = new_scorer.predict(test_data)
        assert np.allclose(predictions, new_predictions)


def test_rerank_in_pipeline(dataset: Dataset) -> None:
    """Test RerankScorer as part of an AutoML pipeline."""
    search_space = [
        {
            "node_type": "scoring",
            "target_metric": "scoring_roc_auc",
            "search_space": [
                {
                    "module_name": "rerank",
                    "k": [3],
                    "weights": ["distance"],
                    "cross_encoder_config": [{"model_name": "cross-encoder/ms-marco-MiniLM-L6-v2"}],
                }
            ],
        },
        {"node_type": "decision", "target_metric": "decision_accuracy", "search_space": [{"module_name": "argmax"}]},
    ]

    pipeline = Pipeline.from_search_space(search_space)
    pipeline.fit(dataset)
    predictions = pipeline.predict(["test utterance"])
    assert len(predictions) == 1
