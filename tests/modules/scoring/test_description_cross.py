import tempfile

import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules import CrossEncoderDescriptionScorer


@pytest.mark.parametrize(
    ("expected_prediction", "multilabel"),
    [
        ([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]], True),
        ([[0.2, 0.3, 0.2, 0.2], [0.2, 0.3, 0.2, 0.2]], False),
    ],
)
def test_description_scorer_cross_encoder(dataset, expected_prediction, multilabel):
    if multilabel:
        dataset = dataset.to_multilabel()
    data_handler = DataHandler(dataset)

    scorer = CrossEncoderDescriptionScorer(cross_encoder_config="cross-encoder/ms-marco-MiniLM-L6-v2", temperature=0.3)

    scorer.fit(
        data_handler.train_utterances(0),
        data_handler.train_labels(0),
        data_handler.intent_descriptions,
    )
    assert scorer._description_texts is not None
    assert len(scorer._description_texts) == len(data_handler.intent_descriptions)
    assert scorer._cross_encoder is not None

    test_utterances = [
        "What is the balance on my account?",
        "How do I reset my online banking password?",
    ]

    predictions = scorer.predict(test_utterances)
    if multilabel:
        assert np.sum(predictions) <= len(test_utterances) * 4
    else:
        np.testing.assert_almost_equal(np.sum(predictions), len(test_utterances))

    assert predictions.shape == (len(test_utterances), len(data_handler.intent_descriptions))
    np.testing.assert_almost_equal(predictions, np.array(expected_prediction).reshape(predictions.shape), decimal=1)

    predictions, metadata = scorer.predict_with_metadata(test_utterances)
    assert len(predictions) == len(test_utterances)
    assert metadata is None

    scorer.clear_cache()

    with tempfile.TemporaryDirectory() as temp_dir:
        scorer.dump(temp_dir)

        new_scorer = CrossEncoderDescriptionScorer.load(temp_dir)

        loaded_predictions = new_scorer.predict(test_utterances)

        np.testing.assert_almost_equal(predictions, loaded_predictions, decimal=5)

        new_scorer.clear_cache()
