from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from autointent import Pipeline
from autointent.context.data_handler import DataHandler
from autointent.modules.scoring import LinearScorer
from tests._helpers import is_strict_labels
from tests.conftest import get_test_embedder_config

if TYPE_CHECKING:
    import numpy.typing as npt

    from autointent import Dataset


def test_base_linear(dataset: Dataset) -> None:
    data_handler = DataHandler(dataset)

    scorer = LinearScorer(embedder_config=get_test_embedder_config())

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
                [4.42261625e-03, 9.80002146e-01, 5.84225268e-03, 9.73298532e-03],
                [3.48457612e-02, 8.67882177e-01, 5.26664920e-02, 4.46055700e-02],
                [6.60129036e-02, 6.81724763e-01, 6.13724992e-02, 1.90889834e-01],
                [3.19191741e-01, 3.05030337e-01, 1.57439488e-01, 2.18338434e-01],
                [1.25137105e-04, 9.99343901e-01, 2.06237249e-04, 3.24724282e-04],
            ]
        ),
        predictions,
        decimal=2,
    )

    # cast: base predict_with_metadata signature is wider than what scoring subclasses actually return.
    predictions, metadata = cast(
        "tuple[npt.NDArray[Any], list[dict[str, Any]] | None]", scorer.predict_with_metadata(test_data)
    )
    assert len(predictions) == len(test_data)
    assert metadata is None

    with tempfile.TemporaryDirectory() as temp_dir:
        scorer.dump(temp_dir)
        del scorer
        new_scorer = LinearScorer.load(temp_dir)
        new_predictions = new_scorer.predict(test_data)
        np.testing.assert_almost_equal(predictions, new_predictions, decimal=5)


def test_linear_in_pipeline(dataset: Dataset) -> None:
    """Test LinearScorer as part of an AutoML pipeline."""
    search_space = [
        {
            "node_type": "scoring",
            "target_metric": "scoring_roc_auc",
            "search_space": [
                {
                    "module_name": "linear",
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
