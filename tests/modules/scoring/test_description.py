import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules import DescriptionScorer


@pytest.mark.parametrize(
    ("expected_prediction", "multilabel"),
    [
        ([[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]], True),
        ([[0.2, 0.3, 0.2], [0.2, 0.3, 0.2]], False),
    ],
)
def test_description_scorer(setup_environment, dataset, expected_prediction, multilabel):
    db_dir, dump_dir, logs_dir = setup_environment
    data_handler = DataHandler(dataset, force_multilabel=multilabel)

    scorer = DescriptionScorer(model_name="sergeyzh/rubert-tiny-turbo", db_dir=db_dir(), temperature=0.3)

    scorer.fit(data_handler.utterances_train, data_handler.labels_train, data_handler.label_description)
    assert scorer.description_vectors.shape[0] == len(data_handler.label_description)
    assert scorer.metadata["model_name"] == "sergeyzh/rubert-tiny-turbo"

    test_utterances = [
        "What is the balance on my account?",
        "How do I reset my online banking password?",
    ]

    predictions = scorer.predict(test_utterances)
    if multilabel:
        assert np.sum(predictions) <= len(test_utterances) * 4
    else:
        np.testing.assert_almost_equal(np.sum(predictions), len(test_utterances))

    assert predictions.shape == (len(test_utterances), len(data_handler.label_description))
    np.testing.assert_almost_equal(predictions, np.array(expected_prediction).reshape(predictions.shape), decimal=1)

    scorer.clear_cache()
