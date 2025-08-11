import os
import tempfile

import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules import LLMDescriptionScorer


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_MODEL_NAME"),
    reason="OPENAI_API_KEY and OPENAI_MODEL_NAME environment variables are required for this test",
)
@pytest.mark.parametrize("multilabel", [True, False])
def test_description_scorer_llm(dataset, multilabel):
    if multilabel:
        dataset = dataset.to_multilabel()
    data_handler = DataHandler(dataset)

    scorer = LLMDescriptionScorer(temperature=0.3, generator_config={"temperature": 0}, multilabel=multilabel)

    scorer.fit(
        data_handler.train_utterances(0),
        data_handler.train_labels(0),
        data_handler.intent_descriptions,
    )
    assert scorer._description_texts == data_handler.intent_descriptions

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

    predictions, metadata = scorer.predict_with_metadata(test_utterances)
    assert len(predictions) == len(test_utterances)
    assert metadata is None

    with tempfile.TemporaryDirectory() as temp_dir:
        scorer.dump(temp_dir)
        del scorer
        new_scorer = LLMDescriptionScorer.load(temp_dir)
        new_predictions = new_scorer.predict(test_utterances)
        np.testing.assert_almost_equal(predictions, new_predictions, decimal=5)
