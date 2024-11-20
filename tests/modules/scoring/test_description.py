import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules import DescriptionScorer
from tests.conftest import setup_environment


@pytest.mark.parametrize(
    ("expected_prediction", "multilabel"),
    [
        ([[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]], True),
        ([[0.2, 0.3, 0.2], [0.2, 0.3, 0.2]], False),
    ],
)
def test_description_scorer(dataset, expected_prediction, multilabel):
    db_dir, dump_dir, logs_dir = setup_environment()
    data_handler = DataHandler(dataset, force_multilabel=multilabel)

    scorer = DescriptionScorer(embedder_name="sergeyzh/rubert-tiny-turbo", temperature=0.3, device="cpu")

    scorer.fit(data_handler.train_utterances, data_handler.train_labels, data_handler.intent_descriptions)
    assert scorer.description_vectors.shape[0] == len(data_handler.intent_descriptions)

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
