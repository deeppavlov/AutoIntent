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


# ---------------------------------------------------------------------------
# Canonical test models. See docs/superpowers/specs/2026-06-06-hf-test-refactor-design.md
#
# Every test that genuinely needs an HF model uses one of these three. The
# SHAs are pinned in src/autointent/configs/_transformers.DEFAULT_REVISIONS,
# so HFModelConfig._apply_default_revision auto-fills `revision` on each
# config instantiated below and no HF API call is needed to resolve a tag.
# ---------------------------------------------------------------------------

TINY_BERT = "prajjwal1/bert-tiny"
TINY_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"
TINY_SENTENCE_TRANSFORMER = "sergeyzh/rubert-tiny-turbo"


def tiny_bert_config():
    """HFModelConfig pinned at TINY_BERT; revision auto-filled by validator."""
    from autointent.configs import HFModelConfig
    return HFModelConfig(model_name=TINY_BERT)


def tiny_cross_encoder_config():
    """CrossEncoderConfig pinned at TINY_CROSS_ENCODER."""
    from autointent.configs import CrossEncoderConfig
    return CrossEncoderConfig(model_name=TINY_CROSS_ENCODER)


def tiny_sentence_transformer_config(**overrides):
    """SentenceTransformerEmbeddingConfig pinned at TINY_SENTENCE_TRANSFORMER.

    Default kwargs match the lightweight test profile used in
    tests/embedder/conftest.py: batch_size=4, device='cpu', use_cache=False.
    """
    from autointent.configs import SentenceTransformerEmbeddingConfig
    base = {
        "model_name": TINY_SENTENCE_TRANSFORMER,
        "batch_size": 4,
        "device": "cpu",
        "use_cache": False,
    }
    base.update(overrides)
    return SentenceTransformerEmbeddingConfig(**base)


def apply_test_models(pipeline) -> None:
    """Retarget every HF model slot in a Pipeline at the canonical test set.

    Use this right after Pipeline.from_preset(...) in preset tests. After this
    call:
      - pipeline.embedder_config        -> SentenceTransformerEmbeddingConfig(TINY_SENTENCE_TRANSFORMER)
      - pipeline.cross_encoder_config   -> CrossEncoderConfig(TINY_CROSS_ENCODER)
      - pipeline.transformer_config     -> HFModelConfig(TINY_BERT)
      - any search-space module entry that hardcodes a model_name (e.g. the
        deberta entries in transformers-{light,heavy,no-hpo}) is rewritten to
        TINY_BERT.

    HashingVectorizer is intentionally NOT used here: the whole point of a
    preset test is to exercise real SentenceTransformer / cross-encoder
    machinery end-to-end. Non-preset tests where the embedder is incidental
    should call get_test_embedder_config() directly instead.
    """
    pipeline.set_config(tiny_sentence_transformer_config())
    pipeline.set_config(tiny_cross_encoder_config())
    pipeline.set_config(tiny_bert_config())
    _retarget_search_space_models(pipeline)


def _retarget_search_space_models(pipeline) -> None:
    """Walk pipeline.nodes -> NodeOptimizer.modules_search_spaces and rewrite
    any embedded model_name fields to the canonical tiny equivalents.

    The fields we touch are exactly those that presets are known to pin:
      - classification_model_config (used by module_name='bert') -> TINY_BERT
      - embedder_config (when used as a module-level override)   -> TINY_SENTENCE_TRANSFORMER
      - cross_encoder_config (module-level)                      -> TINY_CROSS_ENCODER

    Each field can be either a dict (single value) or a list of dicts
    (Optuna categorical). We rewrite the model_name in every dict found.
    """
    # pipeline.nodes is a dict[NodeType, NodeOptimizer]; iterate values.
    nodes = pipeline.nodes.values() if isinstance(pipeline.nodes, dict) else pipeline.nodes
    for node in nodes:
        for entry in node.modules_search_spaces:
            _rewrite_field(entry, "classification_model_config", TINY_BERT)
            _rewrite_field(entry, "embedder_config", TINY_SENTENCE_TRANSFORMER)
            _rewrite_field(entry, "cross_encoder_config", TINY_CROSS_ENCODER)


def _rewrite_field(entry: dict, field_name: str, new_model_name: str) -> None:
    value = entry.get(field_name)
    if value is None:
        return
    # When rewriting model_name, also drop any explicit revision: the YAML's
    # revision was pinned to the OLD model and is wrong for the new one. The
    # HFModelConfig validator will refill `revision` from DEFAULT_REVISIONS
    # when the config is finally instantiated.
    if isinstance(value, dict):
        if "model_name" in value:
            value["model_name"] = new_model_name
            value.pop("revision", None)
    elif isinstance(value, list):
        for cfg in value:
            if isinstance(cfg, dict) and "model_name" in cfg:
                cfg["model_name"] = new_model_name
                cfg.pop("revision", None)


# ---------------------------------------------------------------------------
# Unpinned-HF-call guard. See spec §6.5.
# ---------------------------------------------------------------------------

import re as _re  # noqa: E402

_HF_SHA = _re.compile(r"^[0-9a-f]{40}$")


def _make_hf_guard(orig, label: str):
    """Wrap an HF entry point so calls with revision not matching a 40-hex SHA raise."""
    def guarded(repo_id, *args, revision=None, **kwargs):
        if revision is None or not _HF_SHA.match(revision):
            msg = (
                f"Unpinned HF call: {label}({repo_id!r}, ..., revision={revision!r}). "
                "Pin the SHA via DEFAULT_REVISIONS or pass revision=<40-hex-sha> explicitly. "
                "If a test legitimately needs an unpinned call, mark it with "
                "@pytest.mark.allow_unpinned_hf."
            )
            raise AssertionError(msg)
        return orig(repo_id, *args, revision=revision, **kwargs)
    return guarded


@pytest.fixture(autouse=True, scope="session")
def _forbid_unpinned_hf_calls():
    """Session-scoped autouse guard: every call into huggingface_hub goes through
    a wrapper that fails if revision isn't a 40-hex SHA."""
    import huggingface_hub
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        mp.setattr(huggingface_hub, "hf_hub_download",
                   _make_hf_guard(huggingface_hub.hf_hub_download, "hf_hub_download"))
        mp.setattr(huggingface_hub, "snapshot_download",
                   _make_hf_guard(huggingface_hub.snapshot_download, "snapshot_download"))
        mp.setattr(huggingface_hub, "model_info",
                   _make_hf_guard(huggingface_hub.model_info, "model_info"))
        # HfApi.model_info is a bound method; wrap as a regular function on the class.
        orig_api_model_info = huggingface_hub.HfApi.model_info
        def _guarded_api_model_info(self, repo_id, *args, revision=None, **kwargs):
            if revision is None or not _HF_SHA.match(revision):
                msg = (
                    f"Unpinned HF call: HfApi.model_info({repo_id!r}, revision={revision!r}). "
                    "Pin the SHA or mark the test with @pytest.mark.allow_unpinned_hf."
                )
                raise AssertionError(msg)
            return orig_api_model_info(self, repo_id, *args, revision=revision, **kwargs)
        mp.setattr(huggingface_hub.HfApi, "model_info", _guarded_api_model_info)
        yield
    finally:
        mp.undo()
