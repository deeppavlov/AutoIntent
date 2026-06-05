from __future__ import annotations

import importlib.resources as ires
from typing import TYPE_CHECKING, Literal

import pytest

from autointent import Dataset
from autointent.utils import load_search_space

if TYPE_CHECKING:
    from pathlib import Path


def _disable_transformers_mistral_regex_patch() -> None:
    # transformers.PreTrainedTokenizerBase._patch_mistral_regex calls
    # huggingface_hub.model_info() for every tokenizer load with vocab > 100k
    # (e.g. XLM-RoBERTa-based models like intfloat/multilingual-e5-*). On CI
    # that uncacheable API call hammers the HF rate limit (429s). Tests never
    # load mistralai tokenizers, so the correction is pure overhead — replace
    # it with a no-op for the whole test session.
    #
    # Upstream bug & fix (merged for transformers 5.0.0+, NOT backported to 4.x):
    #   https://github.com/huggingface/transformers/issues/44843
    #   https://github.com/huggingface/transformers/pull/45444
    # Drop this workaround when we upgrade to transformers>=5.0:
    #   https://github.com/deeppavlov/AutoIntent/issues/295
    try:
        from transformers import tokenization_utils_base
    except ImportError:
        return

    base = getattr(tokenization_utils_base, "PreTrainedTokenizerBase", None)
    if base is None or not hasattr(base, "_patch_mistral_regex"):
        return

    def _noop_patch_mistral_regex(cls, tokenizer, *args, **kwargs):  # noqa: ARG001
        return tokenizer

    base._patch_mistral_regex = classmethod(_noop_patch_mistral_regex)


_disable_transformers_mistral_regex_patch()


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
