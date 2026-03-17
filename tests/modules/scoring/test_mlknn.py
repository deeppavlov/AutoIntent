import tempfile

import numpy as np
import pytest

from autointent import Pipeline
from autointent.context.data_handler import DataHandler
from autointent.modules.scoring import MLKnnScorer
from tests.conftest import get_test_embedder_config


def test_base_mlknn(dataset):
    pytest.importorskip("sentence_transformers", reason="Sentence Transformers library is required for these tests")

    data_handler = DataHandler(dataset.to_multilabel())

    scorer = MLKnnScorer(embedder_config=get_test_embedder_config(), k=3)
    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]

    predictions = scorer.predict_labels(test_data)
    assert (
        predictions
        == np.array(
            [
                [
                    0,
                    1,
                    0,
                    0,
                ],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ]
        )
    ).all()

    predictions, metadata = scorer.predict_with_metadata(test_data)
    assert len(predictions) == len(test_data)
    assert "neighbors" in metadata[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        scorer.dump(temp_dir)
        del scorer
        new_scorer = MLKnnScorer.load(temp_dir)
        new_predictions = new_scorer.predict(test_data)
        assert np.allclose(predictions, new_predictions)


def test_mlknn_in_pipeline(dataset):
    """Test MLKnnScorer as part of an AutoML pipeline."""
    search_space = [
        {
            "node_type": "scoring",
            "target_metric": "scoring_hit_rate",
            "search_space": [
                {
                    "module_name": "mlknn",
                    "k": [3],
                }
            ],
        },
        {
            "node_type": "decision",
            "target_metric": "decision_accuracy",
            "search_space": [{"module_name": "threshold", "thresh": [0.5]}],
        },
    ]

    pipeline = Pipeline.from_search_space(search_space)
    pipeline.set_config(get_test_embedder_config())
    pipeline.fit(dataset.to_multilabel())
    predictions = pipeline.predict(["test utterance"])
    assert len(predictions) == 1
