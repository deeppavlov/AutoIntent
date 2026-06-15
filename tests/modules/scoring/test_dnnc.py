from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pytest

from autointent import Pipeline
from autointent.context.data_handler import DataHandler
from autointent.modules.scoring import DNNCScorer
from tests._helpers import is_strict_labels

if TYPE_CHECKING:
    from autointent import Dataset

pytest.importorskip("sentence_transformers")


@pytest.mark.parametrize(("train_head", "pred_score"), [(True, 1)])
def test_base_dnnc(dataset: Dataset, train_head: bool, pred_score: int) -> None:
    data_handler = DataHandler(dataset)

    scorer = DNNCScorer(
        cross_encoder_config={"model_name": "cross-encoder/ms-marco-MiniLM-L6-v2", "train_head": train_head},
        embedder_config="sergeyzh/rubert-tiny-turbo",
        k=3,
    )

    # tests use the non-OOS clinc_subset, so train_labels never returns None entries.
    labels = data_handler.train_labels(0)
    assert is_strict_labels(labels)
    scorer.fit(data_handler.train_utterances(0), labels)
    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]
    predictions = scorer.predict(test_data)
    np.testing.assert_almost_equal(
        np.array([[0.0, pred_score, 0.0, 0.0]] * len(test_data)),
        predictions,
        decimal=0.5,  # type: ignore[arg-type]  # reason: numpy stubs require int but assert_almost_equal rounds float decimal; preserves pre-typing behavior
    )

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert metadata is not None
    assert "neighbors" in metadata[0]
    assert "scores" in metadata[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        scorer.dump(temp_dir)
        del scorer
        new_scorer = DNNCScorer.load(temp_dir)
        new_predictions = new_scorer.predict(test_data)
        np.testing.assert_almost_equal(predictions, new_predictions, decimal=5)


def test_dnnc_in_pipeline(dataset: Dataset) -> None:
    """Test DNNCScorer as part of an AutoML pipeline."""
    search_space = [
        {
            "node_type": "scoring",
            "target_metric": "scoring_roc_auc",
            "search_space": [
                {
                    "module_name": "dnnc",
                    "cross_encoder_config": [{"model_name": "cross-encoder/ms-marco-MiniLM-L6-v2"}],
                    "k": [3],
                }
            ],
        },
        {"node_type": "decision", "target_metric": "decision_accuracy", "search_space": [{"module_name": "argmax"}]},
    ]

    pipeline = Pipeline.from_search_space(search_space)
    pipeline.fit(dataset)
    predictions = pipeline.predict(["test utterance"])
    assert len(predictions) == 1
