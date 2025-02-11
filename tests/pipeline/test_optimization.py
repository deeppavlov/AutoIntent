import os
from typing import Literal

import pytest

from autointent import Pipeline
from autointent.configs import (
    LoggingConfig,
    VectorIndexConfig,
)
from tests.conftest import get_search_space, setup_environment

TaskType = Literal["multiclass", "multilabel", "description"]


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_cv(dataset, task_type):
    project_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(VectorIndexConfig())

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset, scheme="cv", refit_after=True)
    context.dump()

    assert os.listdir(pipeline_optimizer.logging_config.dump_dir)


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_no_context_optimization(dataset, task_type):
    project_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=False, clear_ram=False))
    pipeline_optimizer.set_config(VectorIndexConfig(save_db=True))

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset)
    context.dump()


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_dump_modules(dataset, task_type):
    project_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(VectorIndexConfig())

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset)
    context.dump()

    assert os.listdir(pipeline_optimizer.logging_config.dump_dir)


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
        pipeline_optimizer.validate_modules(dataset)


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
        pipeline_optimizer.validate_modules(dataset)
