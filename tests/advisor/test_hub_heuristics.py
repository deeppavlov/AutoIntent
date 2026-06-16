"""Tests for the offline heuristic fallback in `_hub`.

The advisor must produce a sensible estimate even when HF Hub is unreachable.
Without a per-name heuristic, every offline lookup collapses to a single
BERT-base-sized default — these tests pin that contract.
"""

from __future__ import annotations

import pytest

from autointent._advisor import _hub


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _hub.resolve_model.cache_clear()
    # Force `_hub_metadata` to behave as if the live Hub were unreachable so
    # resolve_model falls through to `_heuristic_metadata`.
    monkeypatch.setattr(_hub, "_hub_metadata", lambda _name: None)
    monkeypatch.setattr(_hub, "_is_warm_cached", lambda _name: False)


def test_offline_lookup_uses_bert_base_default() -> None:
    """Every offline lookup returns the same BERT-base-sized fallback."""
    for name in (
        "microsoft/deberta-v3-large",
        "sentence-transformers/all-MiniLM-L6-v2",
        "totally-made-up/no-such-model",
    ):
        meta = _hub.resolve_model(name)
        assert meta.confidence == "heuristic"
        assert meta.total_params == _hub._DEFAULT_HEURISTIC_PARAMS


def test_weights_gb_matches_params_times_bytes() -> None:
    meta = _hub.resolve_model("microsoft/deberta-v3-large")
    expected_gb = meta.total_params * meta.weight_bytes_per_param / (1024**3)
    assert meta.weights_gb == pytest.approx(expected_gb)


def test_local_path_returns_zero_disk() -> None:
    meta = _hub.resolve_model("/tmp/local/path/to/model")
    assert meta.total_file_bytes == 0
    assert meta.cached_locally is True


def test_disk_gb_falls_back_to_param_size_when_siblings_unknown() -> None:
    meta = _hub.resolve_model("intfloat/multilingual-e5-large-instruct")
    assert meta.disk_gb > 0
    assert meta.disk_gb == pytest.approx(meta.weights_gb, rel=0.01)


def test_resolve_is_memoized() -> None:
    a = _hub.resolve_model("microsoft/deberta-v3-large")
    b = _hub.resolve_model("microsoft/deberta-v3-large")
    assert a is b


def test_metadata_fallback_uses_heuristic_when_hub_unreachable() -> None:
    """End-to-end: resolve_model must return a usable ModelMeta even when
    the live Hub is unreachable (autouse fixture forces offline)."""
    meta = _hub.resolve_model("microsoft/deberta-v3-large")
    assert meta.confidence == "heuristic"
    assert meta.total_params > 0
    assert meta.disk_gb > 0
