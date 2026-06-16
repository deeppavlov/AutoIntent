from __future__ import annotations

import importlib.resources as ires
from typing import TYPE_CHECKING, cast

import pytest

from autointent import Pipeline
from autointent.configs import DataConfig, HPOConfig, LoggingConfig
from tests.conftest import apply_test_models, get_search_space

if TYPE_CHECKING:
    from pathlib import Path

    from autointent import Dataset
    from autointent.custom_types import SamplerType
    from autointent.generation import Generator
    from tests.conftest import TaskType


@pytest.mark.parametrize(
    ("data_config", "refit_after"),
    [
        (DataConfig(scheme="ho", separation_ratio=None), False),
        (DataConfig(scheme="ho", separation_ratio=0.5), False),
        (DataConfig(scheme="cv", separation_ratio=None), False),
        (DataConfig(scheme="cv", separation_ratio=0.5), False),
        (DataConfig(scheme="ho", separation_ratio=None), True),
        (DataConfig(scheme="ho", separation_ratio=0.5), True),
        (DataConfig(scheme="cv", separation_ratio=None), True),
        (DataConfig(scheme="cv", separation_ratio=0.5), True),
    ],
)
def test_with_regex(dataset: Dataset, data_config: DataConfig, refit_after: bool, tmp_path: Path) -> None:
    project_dir = tmp_path
    search_space = get_search_space("regex")

    pipeline_optimizer = Pipeline.from_search_space(search_space)
    apply_test_models(pipeline_optimizer)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(data_config)

    pipeline_optimizer.fit(dataset, refit_after=refit_after)


def test_no_node_separation(dataset_no_oos: Dataset, tmp_path: Path) -> None:
    project_dir = tmp_path
    search_space = get_search_space("light")

    pipeline_optimizer = Pipeline.from_search_space(search_space)
    apply_test_models(pipeline_optimizer)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(DataConfig(scheme="ho", separation_ratio=None))

    pipeline_optimizer.fit(dataset_no_oos, refit_after=False)


def test_full_config(dataset_no_oos: Dataset) -> None:
    # tests.assets.configs is a regular package, so importlib.resources.files
    # returns a concrete Path; cast asserts that to mypy without changing
    # behavior (matches the pattern used in tests/conftest.py).
    # reason: importlib.resources.files() returns Traversable typed as Any
    config_path = cast("Path", ires.files("tests.assets.configs").joinpath("full_training.yaml"))
    pipeline_optimizer = Pipeline.from_optimization_config(config_path)
    apply_test_models(pipeline_optimizer)
    pipeline_optimizer.fit(dataset_no_oos, refit_after=False)


@pytest.mark.parametrize(
    "sampler",
    ["tpe", "random"],
)
def test_bayes(dataset: Dataset, sampler: SamplerType, tmp_path: Path) -> None:
    project_dir = tmp_path
    search_space = get_search_space("optuna")

    pipeline_optimizer = Pipeline.from_search_space(search_space)
    apply_test_models(pipeline_optimizer)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(DataConfig(scheme="ho", separation_ratio=0.5))
    pipeline_optimizer.set_config(HPOConfig(sampler=sampler))

    pipeline_optimizer.fit(dataset, refit_after=False)


@pytest.mark.parametrize(
    "task_type",
    [
        "multiclass",
        "multilabel",
        "description_no_llm",
        "description_with_llm",
    ],
)
def test_cv(dataset: Dataset, task_type: TaskType, patch_llm_scorer_generator: Generator, tmp_path: Path) -> None:
    project_dir = tmp_path
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)
    apply_test_models(pipeline_optimizer)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(DataConfig(scheme="cv", separation_ratio=0.5))

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset, refit_after=True)
    context.dump()

    assert len(list(pipeline_optimizer.logging_config.dump_dir.iterdir())) > 0


@pytest.mark.parametrize(
    "task_type",
    [
        "multiclass",
        "multilabel",
        "description_no_llm",
        "description_with_llm",
    ],
)
def test_no_context_optimization(
    dataset: Dataset, task_type: TaskType, patch_llm_scorer_generator: Generator, tmp_path: Path
) -> None:
    project_dir = tmp_path
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)
    apply_test_models(pipeline_optimizer)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=False, clear_ram=False))
    pipeline_optimizer.set_config(DataConfig(scheme="ho", separation_ratio=0.5))

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset)
    context.dump()


@pytest.mark.parametrize(
    "task_type",
    [
        "multiclass",
        "multilabel",
        "description_no_llm",
        "description_with_llm",
    ],
)
def test_dump_modules(
    dataset: Dataset, task_type: TaskType, patch_llm_scorer_generator: Generator, tmp_path: Path
) -> None:
    project_dir = tmp_path
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)
    apply_test_models(pipeline_optimizer)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset)
    context.dump()

    assert len(list(pipeline_optimizer.logging_config.dump_dir.iterdir())) > 0


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel"],
)
def test_optimization_validation_metric_names(dataset: Dataset, task_type: TaskType) -> None:
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)
    apply_test_models(pipeline_optimizer)

    if task_type == "multiclass":
        dataset = dataset.to_multilabel()

    with pytest.raises(ValueError, match=r"Target metric .*"):
        pipeline_optimizer.fit(dataset, incompatible_search_space="raise")
