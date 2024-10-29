from unittest.mock import Mock

import numpy as np
import pytest

from autointent import Context
from autointent.modules import DescriptionScorer
from tests.conftest import load_clinc_subset


@pytest.mark.parametrize("expected_prediction", [
    ([[1, 0, 0, 0], [1, 0, 0, 0]])
])
@pytest.mark.parametrize("multilabel", [True, False])
def test_description_scorer(setup_environment, expected_prediction, multilabel):
    db_dir, dump_dir, logs_dir = setup_environment
    dataset = load_clinc_subset("description")

    context = Context(
        dataset=dataset,
        test_dataset=None,
        device="cpu",
        multilabel_generation_config="",
        regex_sampling=0,
        seed=0,
        dump_dir=dump_dir,
        db_dir=db_dir(),
        force_multilabel=multilabel,
    )

    mock_embedder = Mock()
    mock_embedder.embed.side_effect = [
        np.array([[100, 200], [5, -5], [-4, -4], [-100, -150]]),
        np.array([[100, 200], [100, 250]]),
    ]

    mock_vector_index = Mock()
    mock_vector_index.embedder = mock_embedder
    mock_vector_index.model_name = "mock-model"
    context.get_best_index = Mock(return_value=mock_vector_index)

    scorer = DescriptionScorer(
        model_name="sergeyzh/rubert-tiny-turbo",
        db_dir=db_dir(),
        n_classes=3,
        temperature=0.1
    )

    scorer.fit(context.data_handler.utterances_train, context.data_handler.labels_train, context.data_handler.label_description)
    assert scorer.description_vectors.shape[0] == len(context.data_handler.label_description)
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

    assert predictions.shape == (len(test_utterances), len(context.data_handler.label_description))
    np.testing.assert_almost_equal(predictions, np.array(expected_prediction).reshape(predictions.shape), decimal=1)

    scorer.clear_cache()
    mock_vector_index.delete.assert_called_once()
