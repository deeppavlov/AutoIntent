from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from autointent import Pipeline
from autointent.context.data_handler import DataHandler
from autointent.modules.scoring import SklearnScorer
from tests._helpers import is_strict_labels
from tests.conftest import get_test_embedder_config

if TYPE_CHECKING:
    import numpy.typing as npt

    from autointent import Dataset


def test_base_sklearn(dataset: Dataset) -> None:
    data_handler = DataHandler(dataset)

    scorer = SklearnScorer(
        embedder_config=get_test_embedder_config(),
        clf_name="LogisticRegression",
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
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
        np.array(
            [
                [0.19808616, 0.33850935, 0.20807189, 0.25533256],
                [0.21305655, 0.28760493, 0.22420657, 0.275132],
                [0.21481034, 0.2826606, 0.22563915, 0.27688998],
                [0.21779545, 0.27305433, 0.22861205, 0.2805381],
                [0.18922822, 0.3680897, 0.19876744, 0.2439147],
            ]
        ),
        predictions,
        decimal=2,
    )

    # cast: base predict_with_metadata signature is wider than scoring subclasses actually return.
    predictions, metadata = cast(
        "tuple[npt.NDArray[Any], list[dict[str, Any]] | None]", scorer.predict_with_metadata(test_data)
    )
    assert len(predictions) == len(test_data)
    assert metadata is None

    with tempfile.TemporaryDirectory() as temp_dir:
        scorer.dump(temp_dir)
        del scorer
        new_scorer = SklearnScorer.load(temp_dir)
        new_predictions = new_scorer.predict(test_data)
        np.testing.assert_almost_equal(predictions, new_predictions, decimal=5)


def test_sklearn_in_pipeline(dataset: Dataset) -> None:
    """Test SklearnScorer as part of an AutoML pipeline."""
    search_space = [
        {
            "node_type": "scoring",
            "target_metric": "scoring_roc_auc",
            "search_space": [
                {
                    "module_name": "sklearn",
                    "clf_name": ["LogisticRegression"],
                }
            ],
        },
        {"node_type": "decision", "target_metric": "decision_accuracy", "search_space": [{"module_name": "argmax"}]},
    ]

    pipeline = Pipeline.from_search_space(search_space)
    pipeline.set_config(get_test_embedder_config())
    pipeline.fit(dataset)
    predictions = pipeline.predict(["test utterance"])
    assert len(predictions) == 1
