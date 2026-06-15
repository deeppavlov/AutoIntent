from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from autointent import Pipeline
from autointent.context.data_handler import DataHandler
from autointent.modules.scoring import LLMDescriptionScorer
from tests._helpers import is_strict_labels

if TYPE_CHECKING:
    import numpy.typing as npt

    from autointent import Dataset
    from autointent.generation import Generator


@pytest.mark.parametrize("multilabel", [True, False])
def test_description_scorer_llm(dataset: Dataset, multilabel: bool, patch_llm_scorer_generator: Generator) -> None:
    if multilabel:
        dataset = dataset.to_multilabel()
    data_handler = DataHandler(dataset)

    scorer = LLMDescriptionScorer(temperature=0.3, generator_config={"temperature": 0}, multilabel=multilabel)

    # clinc_subset has descriptions defined for every intent, and uses non-OOS labels.
    labels = data_handler.train_labels(0)
    assert is_strict_labels(labels)
    descriptions = data_handler.intent_descriptions
    # Pattern C: mypy cannot narrow list[str | None] from `all(...)` alone; keep the cast.
    assert all(d is not None for d in descriptions)
    scorer.fit(
        data_handler.train_utterances(0),
        labels,
        cast("list[str]", descriptions),
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

    # cast: base predict_with_metadata signature is wider than scoring subclasses actually return.
    predictions, metadata = cast(
        "tuple[npt.NDArray[Any], list[dict[str, Any]] | None]", scorer.predict_with_metadata(test_utterances)
    )
    assert len(predictions) == len(test_utterances)
    assert metadata is None


@pytest.mark.parametrize("multilabel", [True, False])
def test_description_scorer_llm_dump_load_roundtrip(
    dataset: Dataset, multilabel: bool, patch_llm_scorer_generator: Generator
) -> None:
    if multilabel:
        dataset = dataset.to_multilabel()
    data_handler = DataHandler(dataset)

    scorer = LLMDescriptionScorer(temperature=0.3, generator_config={"temperature": 0}, multilabel=multilabel)
    # clinc_subset has descriptions defined for every intent, and uses non-OOS labels.
    labels = data_handler.train_labels(0)
    assert is_strict_labels(labels)
    descriptions = data_handler.intent_descriptions
    # Pattern C: mypy cannot narrow list[str | None] from `all(...)` alone; keep the cast.
    assert all(d is not None for d in descriptions)
    scorer.fit(
        data_handler.train_utterances(0),
        labels,
        cast("list[str]", descriptions),
    )

    test_utterances = ["What is the balance on my account?", "How do I reset my online banking password?"]
    predictions = scorer.predict(test_utterances)

    with tempfile.TemporaryDirectory() as temp_dir:
        scorer.dump(temp_dir)
        del scorer
        new_scorer = LLMDescriptionScorer.load(temp_dir)
        new_predictions = new_scorer.predict(test_utterances)
        np.testing.assert_almost_equal(predictions, new_predictions, decimal=5)


def test_llm_description_in_pipeline(dataset: Dataset, patch_llm_scorer_generator: Generator) -> None:
    """Test LLMDescriptionScorer as part of an AutoML pipeline."""
    search_space = [
        {
            "node_type": "scoring",
            "target_metric": "scoring_roc_auc",
            "search_space": [
                {
                    "module_name": "description_llm",
                    "temperature": [0.3],
                }
            ],
        },
        {"node_type": "decision", "target_metric": "decision_accuracy", "search_space": [{"module_name": "argmax"}]},
    ]

    pipeline = Pipeline.from_search_space(search_space)
    pipeline.fit(dataset)
    predictions = pipeline.predict(["test utterance"])
    assert len(predictions) == 1
