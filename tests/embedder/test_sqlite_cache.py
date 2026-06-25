from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import numpy as np

from autointent._wrappers.embedder._sqlite_cache import (
    SCHEMA_VERSION,
    SQLiteEmbeddingCache,
    get_embedding_cache,
    utterance_key,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    import pytest


def _vec(values: list[float]) -> npt.NDArray[np.float32]:
    return np.asarray(values, dtype=np.float32)


def test_set_get_roundtrip(tmp_path: Path) -> None:
    cache = SQLiteEmbeddingCache(tmp_path / "e.db")
    cache.set_many(123, {"k1": _vec([1.0, 2.0, 3.0])})
    got = cache.get_many(123, ["k1"])
    assert set(got) == {"k1"}
    np.testing.assert_array_equal(got["k1"], _vec([1.0, 2.0, 3.0]))
    assert got["k1"].shape == (3,)


def test_get_partial_hit(tmp_path: Path) -> None:
    cache = SQLiteEmbeddingCache(tmp_path / "e.db")
    cache.set_many(1, {"a": _vec([1.0, 1.0])})
    got = cache.get_many(1, ["a", "b"])
    assert set(got) == {"a"}


def test_get_empty_keys_returns_empty(tmp_path: Path) -> None:
    cache = SQLiteEmbeddingCache(tmp_path / "e.db")
    assert cache.get_many(1, []) == {}


def test_set_empty_entries_is_noop(tmp_path: Path) -> None:
    cache = SQLiteEmbeddingCache(tmp_path / "e.db")
    cache.set_many(1, {})  # must not create/raise
    assert cache.get_many(1, ["anything"]) == {}


def test_model_hash_filter(tmp_path: Path) -> None:
    cache = SQLiteEmbeddingCache(tmp_path / "e.db")
    cache.set_many(111, {"shared": _vec([1.0, 2.0])})
    # A different model must not read model 111's row even for the same key string.
    assert cache.get_many(222, ["shared"]) == {}
    assert set(cache.get_many(111, ["shared"])) == {"shared"}


def test_insert_or_ignore_does_not_overwrite(tmp_path: Path) -> None:
    cache = SQLiteEmbeddingCache(tmp_path / "e.db")
    cache.set_many(1, {"k": _vec([1.0, 2.0])})
    cache.set_many(1, {"k": _vec([9.0, 9.0])})  # ignored
    np.testing.assert_array_equal(cache.get_many(1, ["k"])["k"], _vec([1.0, 2.0]))


def test_chunking_over_variable_limit(tmp_path: Path) -> None:
    cache = SQLiteEmbeddingCache(tmp_path / "e.db")
    entries = {f"k{i}": _vec([float(i)]) for i in range(2000)}
    cache.set_many(1, entries)
    got = cache.get_many(1, list(entries))
    assert len(got) == 2000
    np.testing.assert_array_equal(got["k1999"], _vec([1999.0]))


def test_schema_version_and_columns(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    cache = SQLiteEmbeddingCache(db)
    cache.set_many(1, {"k": _vec([1.0])})  # triggers schema init
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        cols = {row[1] for row in conn.execute("PRAGMA table_info(embeddings)")}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(embeddings)")}
    assert {"key", "model_hash", "dim", "vector", "size_bytes", "created_at", "last_accessed"} <= cols
    assert {
        "idx_embeddings_last_accessed",
        "idx_embeddings_created_at",
        "idx_embeddings_model_hash",
    } <= indexes


def test_version_mismatch_triggers_rebuild(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    SQLiteEmbeddingCache(db).set_many(1, {"old": _vec([1.0])})
    # Simulate an older/newer schema: bump user_version so the next instance rebuilds.
    with sqlite3.connect(db) as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    fresh = SQLiteEmbeddingCache(db)
    fresh.set_many(1, {"new": _vec([2.0])})  # forces _ensure_schema -> rebuild
    assert fresh.get_many(1, ["old"]) == {}  # old row dropped by rebuild


def test_corrupted_db_degrades_to_miss(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    db.write_bytes(b"this is not a sqlite database")
    cache = SQLiteEmbeddingCache(db)
    # Must not raise; reads miss and writes no-op.
    assert cache.get_many(1, ["k"]) == {}
    cache.set_many(1, {"k": _vec([1.0])})


def test_dim_mismatch_row_skipped(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    cache = SQLiteEmbeddingCache(db)
    cache.set_many(1, {"k": _vec([1.0, 2.0])})
    # Corrupt the stored dim so blob length disagrees.
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE embeddings SET dim = 99 WHERE key = 'k'")
        conn.commit()
    assert cache.get_many(1, ["k"]) == {}  # skipped, not raised


def test_get_embedding_cache_memoized_by_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTOINTENT_CACHE_DIR", str(tmp_path / "c"))
    first = get_embedding_cache()
    second = get_embedding_cache()
    assert first is second


def test_utterance_key_distinctness() -> None:
    base = utterance_key(1, "hello", None)
    assert base == utterance_key(1, "hello", None)
    assert base != utterance_key(2, "hello", None)
    assert base != utterance_key(1, "world", None)
    assert base != utterance_key(1, "hello", "Query:")
