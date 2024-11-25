# %% [markdown]
"""
# Optimization demo
"""

# %%
import importlib.resources as ires
from pathlib import Path
from typing import Literal
from uuid import uuid4

from autointent.configs import (
    DataConfig,
    LoggingConfig,
    OptimizationConfig,
    TaskConfig,
    VectorIndexConfig,
)
from autointent.pipeline.optimization._cli_endpoint import main as optimize_pipeline
from autointent.pipeline.optimization._utils import load_config

# %%
TaskType = Literal["multiclass", "multilabel", "description"]


def setup_environment() -> tuple[str, str, str]:
    logs_dir = ires.files("tests").joinpath("logs") / str(uuid4())
    db_dir = logs_dir / "db"
    dump_dir = logs_dir / "modules_dump"
    return db_dir, dump_dir, logs_dir


def get_search_space_path(task_type: TaskType) -> None:
    return ires.files("tests.assets.configs").joinpath(f"{task_type}.yaml")


def get_search_space(task_type: TaskType) -> None:
    path = get_search_space_path(task_type)
    return load_config(str(path), multilabel=task_type == "multilabel")


# %%
def optimize(task_type: TaskType) -> None:
    db_dir, dump_dir, logs_dir = setup_environment()
    config = OptimizationConfig(
        data=DataConfig(
            train_path=ires.files("tests.assets.data").joinpath("clinc_subset.json"),
            force_multilabel=(task_type == "multilabel"),
        ),
        task=TaskConfig(
            search_space_path=get_search_space_path(task_type),
        ),
        vector_index=VectorIndexConfig(
            db_dir=db_dir,
            device="cpu",
        ),
        logs=LoggingConfig(
            dirpath=Path(logs_dir),
        ),
    )
    optimize_pipeline(config)


# %%
optimize("multiclass")
