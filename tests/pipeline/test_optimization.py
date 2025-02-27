import importlib.resources as ires
import os

import pytest

from autointent import Pipeline
from autointent.configs import DataConfig, LoggingConfig
from tests.conftest import get_search_space, setup_environment


def test_no_node_separation(dataset_no_oos):
    project_dir = setup_environment()
    search_space = get_search_space("light")

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(DataConfig(scheme="ho", separate_nodes=False))

    pipeline_optimizer.fit(dataset_no_oos, refit_after=False)


def test_full_config(dataset_no_oos):
    config_path = ires.files("tests.assets.configs").joinpath("full_training.yaml")
    pipeline_optimizer = Pipeline.from_optimization_config(config_path)
    pipeline_optimizer.fit(dataset_no_oos, refit_after=False)


@pytest.mark.parametrize(
    "sampler",
    ["tpe", "random"],
)
def test_bayes(dataset, sampler):
    project_dir = setup_environment()
    search_space = get_search_space("optuna")

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(DataConfig(scheme="ho", separate_nodes=True))

    pipeline_optimizer.fit(dataset, refit_after=False, sampler=sampler)


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_cv(dataset, task_type):
    project_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(DataConfig(scheme="cv", separate_nodes=True))

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset, refit_after=True)
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
    pipeline_optimizer.set_config(DataConfig(scheme="ho", separate_nodes=True))

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

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset)
    context.dump()

    assert os.listdir(pipeline_optimizer.logging_config.dump_dir)
