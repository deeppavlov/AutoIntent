from typing import get_args

import pytest

from autointent import Pipeline
from autointent.nodes.schemes import OptimizationSearchSpaceConfig
from tests.conftest import TaskType, get_search_space


def test_validate_search_space_multiclass(dataset):
    search_space = [
        {
            "node_type": "decision",
            "target_metric": "decision_accuracy",
            "search_space": [{"module_name": "threshold", "thresh": [0.5]}, {"module_name": "adaptive"}],
        },
    ]

    pipeline_optimizer = Pipeline.from_search_space(search_space)
    with pytest.raises(ValueError, match="Module 'adaptive' does not support multiclass datasets."):
        pipeline_optimizer.validate_modules(dataset, mode="raise")


def test_validate_search_space_multilabel(dataset):
    dataset = dataset.to_multilabel()

    search_space = [
        {
            "node_type": "decision",
            "target_metric": "decision_accuracy",
            "search_space": [{"module_name": "threshold", "thresh": [0.5]}, {"module_name": "argmax"}],
        },
    ]
    pipeline_optimizer = Pipeline.from_search_space(search_space)
    with pytest.raises(ValueError, match="Module 'argmax' does not support multilabel datasets."):
        pipeline_optimizer.validate_modules(dataset, mode="raise")


# for now validation for sklearn scorer doesn't work
@pytest.mark.xfail
@pytest.mark.parametrize("search_space", get_args(TaskType))
def test_search_space(search_space):
    OptimizationSearchSpaceConfig(get_search_space(search_space))
