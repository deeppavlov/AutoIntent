from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from autointent import Pipeline

if TYPE_CHECKING:
    from autointent import Dataset


def test_validate_search_space_multiclass(dataset: Dataset) -> None:
    search_space = [
        {
            "node_type": "decision",
            "target_metric": "decision_accuracy",
            "search_space": [{"module_name": "threshold", "thresh": [0.5]}, {"module_name": "adaptive"}],
        },
    ]

    pipeline_optimizer = Pipeline.from_search_space(search_space)
    with pytest.raises(ValueError, match=r"Module 'adaptive' does not support multiclass datasets."):
        pipeline_optimizer.validate_modules(dataset, mode="raise")


def test_validate_search_space_multilabel(dataset: Dataset) -> None:
    dataset = dataset.to_multilabel()

    search_space = [
        {
            "node_type": "decision",
            "target_metric": "decision_accuracy",
            "search_space": [{"module_name": "threshold", "thresh": [0.5]}, {"module_name": "argmax"}],
        },
    ]
    pipeline_optimizer = Pipeline.from_search_space(search_space)
    with pytest.raises(ValueError, match=r"Module 'argmax' does not support multilabel datasets."):
        pipeline_optimizer.validate_modules(dataset, mode="raise")
