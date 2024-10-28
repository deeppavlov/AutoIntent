from unittest.mock import Mock

import numpy as np
import pytest

from autointent import Context
from autointent.modules import DescriptionScorer


@pytest.mark.parametrize(("similarity_metric", "expected_prediction"), [("euclidean", [[1, 0, 0, 0], [0, 0, 0, 1]])])
@pytest.mark.parametrize("multilabel", [True, False])
def test_description_scorer(
    setup_environment, load_clinc_subset, similarity_metric, expected_prediction, dump_dir, multilabel
):
    run_name, db_dir = setup_environment
    dataset = load_clinc_subset("description")

    context = Context(
        dataset=dataset,
        test_dataset=None,
        device="cpu",
        multilabel_generation_config="",
        regex_sampling=0,
        seed=0,
        dump_dir=dump_dir,
        db_dir=db_dir,
        force_multilabel=multilabel,
    )

    mock_embedder = Mock()
    mock_embedder.embed.side_effect = [
        np.array([[1, 2], [10, 16], [20, 21], [100, 150]]),
        np.array([[1, 2], [100, 150]]),
    ]

    mock_vector_index = Mock()
    mock_vector_index.embedder = mock_embedder
    mock_vector_index.model_name = "mock-model"
    context.get_best_index = Mock(return_value=mock_vector_index)

    scorer = DescriptionScorer(similarity_metric=similarity_metric, temperature=1e-4)

    scorer.fit(context)
    assert scorer.description_vectors.shape[0] == len(context.data_handler.label_description)
    assert scorer.metadata["model_name"] == "mock-model"

    test_utterances = [
        "What is the balance on my account?",
        "How do I reset my online banking password?",
    ]

    predictions = scorer.predict(test_utterances)
    if multilabel:
        assert np.sum(predictions) <= len(test_utterances) * 4
    else:
        assert np.sum(predictions) == len(test_utterances)

    assert predictions.shape == (len(test_utterances), len(context.data_handler.label_description))
    np.testing.assert_almost_equal(predictions, np.full(predictions.shape, expected_prediction), decimal=2)

    scorer.clear_cache()
    mock_vector_index.delete.assert_called_once()
