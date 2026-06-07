from __future__ import annotations

import importlib.resources as ires
from typing import TYPE_CHECKING, Literal

import pytest

from autointent import Dataset
from autointent.utils import load_search_space

if TYPE_CHECKING:
    from pathlib import Path


def setup_environment() -> Path:
    return ires.files("tests").joinpath("logs")


def get_dataset_path():
    return ires.files("tests.assets.data").joinpath("clinc_subset.json")


@pytest.fixture
def dataset():
    return Dataset.from_json(get_dataset_path())


@pytest.fixture
def dataset_unsplitted():
    path = ires.files("tests.assets.data").joinpath("clinc_subset_unsplitted.json")
    return Dataset.from_json(path)


@pytest.fixture
def dataset_no_oos():
    path = ires.files("tests.assets.data").joinpath("clinc_no_oos.json")
    return Dataset.from_json(path)


TaskType = Literal["multiclass", "multilabel", "description_no_llm", "description_with_llm", "optuna", "light", "regex"]


def get_search_space_path(task_type: TaskType):
    return ires.files("tests.assets.configs").joinpath(f"{task_type}.yaml")


def get_search_space(task_type: TaskType):
    path = get_search_space_path(task_type)
    return load_search_space(path)


def get_test_embedder_config(**kwargs):
    """Get lightweight embedder config for tests (HashingVectorizer-based).

    This function returns a HashingVectorizer-based embedder config that is:
    - Fast (no model downloads or loading)
    - Lightweight (minimal memory usage)
    - Stateless (no training required)

    Perfect for testing non-embedder specific functionality.

    Args:
        **kwargs: Additional keyword arguments to override defaults.

    Returns:
        HashingVectorizerEmbeddingConfig: Configured embedder for testing.
    """
    from autointent.configs import HashingVectorizerEmbeddingConfig

    defaults = {
        "n_features": 512,
        "use_cache": False,
    }
    defaults.update(kwargs)
    return HashingVectorizerEmbeddingConfig(**defaults)


# ---------------------------------------------------------------------------
# Shared mocking fixtures. See docs/superpowers/specs/2026-06-07-live-api-test-mocking-strategy.md
# ---------------------------------------------------------------------------

from tests._fixtures.fake_openai_embedding import (  # noqa: E402, F401
    FakeOpenaiEmbeddingBackend,
    patch_openai_embedding_backend,
)
from tests._fixtures.mock_generator import (  # noqa: E402, F401
    mock_async_generator,
    mock_generator,
    patch_llm_scorer_generator,
)
