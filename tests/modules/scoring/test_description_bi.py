import tempfile

import numpy as np
import pytest

from autointent import Pipeline
from autointent.context.data_handler import DataHandler
from autointent.modules import BiEncoderDescriptionScorer

pytest.importorskip("sentence-transformers")


@pytest.mark.parametrize(
    ("expected_prediction", "multilabel"),
    [
        ([[0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9]], True),
        ([[0.2, 0.3, 0.2, 0.2], [0.2, 0.3, 0.2, 0.2]], False),
    ],
)
def test_description_scorer(dataset, expected_prediction, multilabel):
    if multilabel:
        dataset = dataset.to_multilabel()
    data_handler = DataHandler(dataset)

    scorer = BiEncoderDescriptionScorer(
        embedder_config="sergeyzh/rubert-tiny-turbo", temperature=0.3, multilabel=multilabel
    )

    scorer.fit(
        data_handler.train_utterances(0),
        data_handler.train_labels(0),
        data_handler.intent_descriptions,
    )
    assert scorer._description_vectors.shape[0] == len(data_handler.intent_descriptions)

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

    with tempfile.TemporaryDirectory() as temp_dir:
        scorer.dump(temp_dir)
        del scorer
        new_scorer = BiEncoderDescriptionScorer.load(temp_dir)
        new_predictions = new_scorer.predict(test_utterances)
        np.testing.assert_almost_equal(predictions, new_predictions, decimal=5)


def test_description_bi_in_pipeline(dataset):
    """Test BiEncoderDescriptionScorer as part of an AutoML pipeline."""
    search_space = [
        {
            "node_type": "scoring",
            "search_space": [
                {
                    "module_name": "description_bi",
                    "embedder_config": [{"model_name": "sergeyzh/rubert-tiny-turbo"}],
                    "temperature": [0.3],
                }
            ],
        },
        {"node_type": "decision", "search_space": [{"module_name": "argmax"}]},
    ]

    pipeline = Pipeline.from_search_space(search_space)
    pipeline.fit(dataset)
    predictions = pipeline.predict(["test utterance"])
    assert len(predictions) == 1
