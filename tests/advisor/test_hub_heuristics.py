"""Tests for the offline name-pattern heuristics in `_hub`.

The advisor must produce a sensible estimate even when HF Hub is
unreachable, so these tests pin the public `hub_reachable` to False and
exercise the heuristic path directly.
"""

from __future__ import annotations

import pytest

from autointent._advisor import _hub


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _hub.hub_reachable.cache_clear()
    _hub.resolve_model.cache_clear()
    monkeypatch.setattr(_hub, "hub_reachable", lambda *_a, **_kw: False)
    monkeypatch.setattr(_hub, "_is_warm_cached", lambda _name: False)


@pytest.mark.parametrize(
    ("name", "expected_min_m", "expected_max_m"),
    [
        ("microsoft/deberta-v3-large", 200, 500),
        ("microsoft/deberta-v3-small", 30, 200),
        ("sentence-transformers/all-MiniLM-L6-v2", 20, 80),
        ("intfloat/multilingual-e5-large-instruct", 300, 700),
        ("intfloat/e5-small", 20, 80),
        ("distilbert-base-uncased", 40, 150),
        ("bert-base-uncased", 70, 200),
    ],
)
def test_name_heuristic_picks_reasonable_bucket(name: str, expected_min_m: int, expected_max_m: int) -> None:
    meta = _hub.resolve_model(name)
    assert meta.confidence == "heuristic"
    assert expected_min_m <= meta.params_millions <= expected_max_m, (
        f"{name} got {meta.params_millions}M; expected [{expected_min_m}, {expected_max_m}]"
    )


def test_unknown_name_falls_back_to_bert_base() -> None:
    meta = _hub.resolve_model("totally-made-up/no-such-model")
    assert meta.confidence == "heuristic"
    assert meta.params_millions == pytest.approx(110.0)


def test_weights_gb_matches_params_times_bytes() -> None:
    meta = _hub.resolve_model("microsoft/deberta-v3-large")
    expected_gb = meta.params_millions * 1_000_000 * meta.weight_bytes_per_param / (1024**3)
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
    assert meta.params_millions > 0
    assert meta.disk_gb > 0
