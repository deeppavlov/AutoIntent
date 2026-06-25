# SQLite per-utterance embedding cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `.npy`-file-per-call embedding cache with a single SQLite database keyed per utterance, lifting the triplicated cache code into one template method.

**Architecture:** A new `SQLiteEmbeddingCache` stores one float32 vector per `(model, utterance, prompt)` key in `<cache_dir>/embeddings.db`. `BaseEmbeddingBackend.embed` becomes a concrete template that splits a call into cache hits/misses, computes only misses via each backend's new `_embed_uncached`, and reassembles in input order. Cache location is configurable via `AUTOINTENT_CACHE_DIR`.

**Tech Stack:** Python 3.10, stdlib `sqlite3` (no new dependency), numpy, xxhash (`Hasher`), pytest, ruff (`select=ALL`), mypy (strict).

**Design spec:** `docs/superpowers/specs/2026-06-25-sqlite-embedding-cache-design.md` — read it before starting; it carries the rationale for every decision below.

## Global Constraints

- **Verification policy (maintainer rule):** Do **NOT** run heavy/exhaustive pytest locally — it can freeze the machine. The local gate for every task is **`ruff check`** + **`mypy src/autointent tests`** only. All pytest verification happens **on CI after the draft PR is pushed**. Each task's pytest commands are listed for reference / CI; the red→green TDD signal is: write the test first, gate locally on ruff+mypy, confirm on CI.
- **Scope:** Embedding cache only. Do **NOT** touch `src/autointent/generation/_cache.py` (structured-output cache) or any non-embedding subsystem.
- **No new dependency.** Stdlib `sqlite3` only.
- **mypy:** strict, `python_version = "3.10"`, covers **both** `src/autointent` and `tests`. Every new function/test needs full annotations.
- **ruff:** `select = ["ALL"]`, `target-version = "py310"`. New non-`utils` modules need module/class/function docstrings, `%`-style logging args (no f-strings in `logger.*`), named constants instead of magic numbers, `from __future__ import annotations`, `pathlib` for paths, `zip(..., strict=True)`.
- **No behavior change to per-utterance vector values** or to public `Embedder.embed` / backend `embed` signatures/return types.
- **Fresh start:** do not migrate or delete the old `.npy` cache.
- **Commit messages** end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/autointent/_cache_dir.py` (new) | `get_cache_dir()` — resolve cache base dir from `AUTOINTENT_CACHE_DIR` or appdirs |
| `src/autointent/_wrappers/embedder/_sqlite_cache.py` (new) | `SQLiteEmbeddingCache`, `utterance_key`, `get_embedding_cache`, constants |
| `src/autointent/_wrappers/embedder/base.py` (mod) | template `embed` + `_embed_cached` + `_to_tensor` + abstract `_embed_uncached` + `config`/`supports_cache` |
| `…/sentence_transformers.py`, `openai.py`, `vllm.py`, `hashing_vectorizer.py` (mod) | each: `config` narrowing + `_embed_uncached` |
| `…/utils.py` (delete) | obsolete `get_embeddings_path` |
| `tests/_fixtures/fake_openai_embedding.py` (mod) | `config` narrowing + `_embed_uncached` |
| `tests/conftest.py` (mod) | global autouse `AUTOINTENT_CACHE_DIR` isolation fixture |
| `tests/test_cache_dir.py` (new) | `get_cache_dir()` unit tests |
| `tests/embedder/test_sqlite_cache.py` (new) | `SQLiteEmbeddingCache` / `utterance_key` unit tests |
| `tests/embedder/test_caching.py` (mod) | per-utterance reuse / dedup / order / empty-input tests |
| `CHANGELOG.md` (mod) | Unreleased entry |

---

## Task 1: Cache-dir helper + global test isolation fixture

**Files:**
- Create: `src/autointent/_cache_dir.py`
- Modify: `tests/conftest.py` (add autouse fixture)
- Test: `tests/test_cache_dir.py`

**Interfaces:**
- Produces: `get_cache_dir() -> pathlib.Path` (honors `AUTOINTENT_CACHE_DIR`, else `appdirs.user_cache_dir("autointent")`).

- [ ] **Step 1: Write the failing test** — `tests/test_cache_dir.py`

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from autointent._cache_dir import get_cache_dir

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_get_cache_dir_honors_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOINTENT_CACHE_DIR", str(tmp_path / "custom"))
    assert get_cache_dir() == tmp_path / "custom"


def test_get_cache_dir_falls_back_to_appdirs(monkeypatch: pytest.MonkeyPatch) -> None:
    # The global autouse isolation fixture sets the env var for every test, so unset it here.
    monkeypatch.delenv("AUTOINTENT_CACHE_DIR", raising=False)
    result = get_cache_dir()
    assert result.name == "autointent" or "autointent" in str(result)
```

- [ ] **Step 2: (reference) test command for CI**

Run on CI: `pytest tests/test_cache_dir.py -v` → expected FAIL initially (`No module named autointent._cache_dir`).

- [ ] **Step 3: Implement `src/autointent/_cache_dir.py`**

```python
"""Resolution of the base directory for autointent on-disk caches."""

from __future__ import annotations

import os
from pathlib import Path

from appdirs import user_cache_dir


def get_cache_dir() -> Path:
    """Return the base directory for autointent on-disk caches.

    Honors the ``AUTOINTENT_CACHE_DIR`` environment variable; otherwise falls back to
    ``appdirs.user_cache_dir("autointent")``. Resolved fresh on each call so tests and
    parallel workers can redirect it via the env var.

    Note:
        Currently consumed only by the embedding cache. The structured-output cache
        still uses ``user_cache_dir("autointent")`` directly and is unaffected by this
        variable.

    Returns:
        The cache base directory as a ``Path``.
    """
    override = os.environ.get("AUTOINTENT_CACHE_DIR")
    return Path(override) if override else Path(user_cache_dir("autointent"))
```

- [ ] **Step 4: Add the global autouse isolation fixture** to `tests/conftest.py`

Append at the end of `tests/conftest.py` (it already imports `pytest` at runtime and `Path` under `TYPE_CHECKING`; the annotation stays unquoted because `from __future__ import annotations` is at the top — a quoted `"Path"` would trip ruff `UP037`):

```python
@pytest.fixture(autouse=True)
def _isolate_embedding_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the embedding SQLite cache to a per-test directory.

    Because ``use_cache`` defaults to True, any test that builds a default-config
    embedder could otherwise write the embedding DB to the real OS cache dir. A unique
    per-test ``tmp_path`` also keeps the per-utterance reuse test in
    tests/embedder/test_caching.py hermetic (its two embeds must share one DB file).
    """
    monkeypatch.setenv("AUTOINTENT_CACHE_DIR", str(tmp_path / "ai_cache"))
```

- [ ] **Step 5: Local gate**

Run: `ruff check src/autointent/_cache_dir.py tests/test_cache_dir.py tests/conftest.py`
Run: `mypy src/autointent tests`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/autointent/_cache_dir.py tests/test_cache_dir.py tests/conftest.py
git commit -m "feat(cache): add get_cache_dir() + global embedding-cache test isolation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `SQLiteEmbeddingCache` store + unit tests

**Files:**
- Create: `src/autointent/_wrappers/embedder/_sqlite_cache.py`
- Test: `tests/embedder/test_sqlite_cache.py`

**Interfaces:**
- Consumes: `get_cache_dir()` (Task 1), `autointent._hash.Hasher`.
- Produces:
  - `SCHEMA_VERSION: int = 1`, `BUSY_TIMEOUT_MS: int = 30000`
  - `utterance_key(model_hash: int, utterance: str, prompt: str | None) -> str`
  - `SQLiteEmbeddingCache(db_path: Path)` with
    `get_many(model_hash: int, keys: list[str]) -> dict[str, npt.NDArray[np.float32]]` and
    `set_many(model_hash: int, entries: dict[str, npt.NDArray[np.float32]]) -> None`
  - `get_embedding_cache() -> SQLiteEmbeddingCache` (memoized by resolved db path)

- [ ] **Step 1: Write the failing tests** — `tests/embedder/test_sqlite_cache.py`

```python
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
```

- [ ] **Step 2: (reference) test command for CI**

Run on CI: `pytest tests/embedder/test_sqlite_cache.py -v` → expected FAIL initially (module missing).

- [ ] **Step 3: Implement `src/autointent/_wrappers/embedder/_sqlite_cache.py`**

```python
"""SQLite-backed per-utterance embedding cache.

Stores one float32 vector per ``(model, utterance, prompt)`` key in a single SQLite
database, replacing the previous one-``.npy``-file-per-call cache. See
``docs/superpowers/specs/2026-06-25-sqlite-embedding-cache-design.md``.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from typing import TYPE_CHECKING, cast

import numpy as np

from autointent._cache_dir import get_cache_dir
from autointent._hash import Hasher

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 30_000
_DB_FILENAME = "embeddings.db"
_FLOAT32_NBYTES = 4
# SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 999 on older builds; stay well under it.
_KEY_CHUNK_SIZE = 900

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS embeddings (
    key           TEXT    PRIMARY KEY,
    model_hash    TEXT    NOT NULL,
    dim           INTEGER NOT NULL,
    vector        BLOB    NOT NULL,
    size_bytes    INTEGER NOT NULL,
    created_at    REAL    NOT NULL,
    last_accessed REAL    NOT NULL
)
"""
_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_embeddings_last_accessed ON embeddings(last_accessed)",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_model_hash ON embeddings(model_hash)",
)
_INSERT = (
    "INSERT OR IGNORE INTO embeddings "
    "(key, model_hash, dim, vector, size_bytes, created_at, last_accessed) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)


def utterance_key(model_hash: int, utterance: str, prompt: str | None) -> str:
    """Compute the per-utterance cache key from model identity, utterance, and prompt.

    Args:
        model_hash: The backend's model-identity hash (``get_hash()``).
        utterance: The original (non-prompted) utterance text.
        prompt: The resolved task prompt, or ``None``.

    Returns:
        A hex digest uniquely identifying ``(model_hash, utterance, prompt)``.
    """
    hasher = Hasher()
    hasher.update(model_hash)
    hasher.update(utterance)
    if prompt:
        hasher.update(prompt)
    return hasher.hexdigest()


class SQLiteEmbeddingCache:
    """Per-utterance embedding cache backed by a single SQLite database.

    Thread-safe (a fresh short-lived connection per call) and process-safe on a local
    filesystem (WAL + ``busy_timeout``). Never raises into callers: any cache I/O failure
    degrades to a miss / no-op and is logged.
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize the cache bound to ``db_path`` (schema is created lazily)."""
        self._db_path = db_path
        self._initialized = False
        self._init_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        """Create the table/indexes once per instance; rebuild on a schema-version change.

        The version check + (re)create runs inside ``BEGIN IMMEDIATE`` with a post-lock
        re-read of ``user_version`` so two processes opening a stale DB cannot double-drop.
        """
        if self._initialized:
            return
        with self._init_lock:
            if not self._initialized:  # another thread may have initialized while we waited
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = self._connect()
                try:
                    mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()
                    if mode is not None and str(mode[0]).lower() != "wal":
                        logger.debug("SQLite embedding cache: WAL unavailable (journal_mode=%s)", mode[0])
                    conn.execute("BEGIN IMMEDIATE")
                    version = conn.execute("PRAGMA user_version").fetchone()[0]
                    if version != SCHEMA_VERSION:
                        conn.execute("DROP TABLE IF EXISTS embeddings")
                        conn.execute(_CREATE_TABLE)
                        for index_sql in _CREATE_INDEXES:
                            conn.execute(index_sql)
                        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                    conn.execute("COMMIT")
                finally:
                    conn.close()
                self._initialized = True

    def get_many(self, model_hash: int, keys: list[str]) -> dict[str, npt.NDArray[np.float32]]:
        """Return cached vectors for ``keys`` under ``model_hash`` (missing keys omitted)."""
        if not keys:
            return {}
        model_hash_str = str(model_hash)
        result: dict[str, npt.NDArray[np.float32]] = {}
        try:
            self._ensure_schema()
            conn = self._connect()
            try:
                for start in range(0, len(keys), _KEY_CHUNK_SIZE):
                    chunk = keys[start : start + _KEY_CHUNK_SIZE]
                    placeholders = ",".join("?" * len(chunk))
                    query = (
                        "SELECT key, vector, dim FROM embeddings "  # noqa: S608 - only '?' is interpolated; values are bound
                        f"WHERE model_hash = ? AND key IN ({placeholders})"
                    )
                    for row_key, blob, dim in conn.execute(query, (model_hash_str, *chunk)):
                        vector = self._deserialize(blob, dim)
                        if vector is not None:
                            result[row_key] = vector
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("SQLite embedding cache read failed (%s); recomputing.", exc)
            return {}
        return result

    def set_many(self, model_hash: int, entries: dict[str, npt.NDArray[np.float32]]) -> None:
        """Insert vectors for new keys under ``model_hash`` (existing keys are untouched)."""
        if not entries:
            return
        model_hash_str = str(model_hash)
        now = time.time()
        rows: list[tuple[str, str, int, bytes, int, float, float]] = []
        for key, vector in entries.items():
            blob = np.ascontiguousarray(vector, dtype=np.float32).tobytes()
            rows.append((key, model_hash_str, int(vector.shape[-1]), blob, len(blob), now, now))
        try:
            self._ensure_schema()
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.executemany(_INSERT, rows)
                conn.execute("COMMIT")
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("SQLite embedding cache write failed (%s); continuing uncached.", exc)

    @staticmethod
    def _deserialize(blob: bytes, dim: int) -> npt.NDArray[np.float32] | None:
        try:
            if len(blob) != dim * _FLOAT32_NBYTES:
                logger.warning("SQLite embedding cache: blob length %d != dim %d; skipping.", len(blob), dim)
                return None
            return cast("npt.NDArray[np.float32]", np.frombuffer(blob, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001 - a bad row must never break embed()
            logger.warning("SQLite embedding cache: failed to deserialize a row (%s); skipping.", exc)
            return None


_INSTANCES: dict[str, SQLiteEmbeddingCache] = {}
_INSTANCES_LOCK = threading.Lock()


def get_embedding_cache() -> SQLiteEmbeddingCache:
    """Return the process-wide cache for the current cache dir (memoized by db path)."""
    db_path = get_cache_dir() / _DB_FILENAME
    key = str(db_path)
    with _INSTANCES_LOCK:
        cache = _INSTANCES.get(key)
        if cache is None:
            cache = SQLiteEmbeddingCache(db_path)
            _INSTANCES[key] = cache
        return cache
```

- [ ] **Step 4: Local gate**

Run: `ruff check src/autointent/_wrappers/embedder/_sqlite_cache.py tests/embedder/test_sqlite_cache.py`
Run: `mypy src/autointent tests`
Expected: clean. If ruff flags `C901`/`PLR0912` on `_ensure_schema` or `get_many`, extract a small helper (e.g. `_run_schema_init(conn)`); do not add blanket noqas.

- [ ] **Step 5: Commit**

```bash
git add src/autointent/_wrappers/embedder/_sqlite_cache.py tests/embedder/test_sqlite_cache.py
git commit -m "feat(cache): add SQLiteEmbeddingCache per-utterance store

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Lift caching into the backend base + migrate all backends

This is one atomic refactor: making `_embed_uncached` abstract forces every subclass to implement it in the same commit. Touches `base.py`, four backends, the test fake, and deletes `utils.py`.

**Files:**
- Modify: `src/autointent/_wrappers/embedder/base.py`
- Modify: `…/sentence_transformers.py`, `…/openai.py`, `…/vllm.py`, `…/hashing_vectorizer.py`
- Modify: `tests/_fixtures/fake_openai_embedding.py`
- Delete: `src/autointent/_wrappers/embedder/utils.py`

**Interfaces:**
- Consumes: `get_embedding_cache`, `utterance_key` (Task 2).
- Produces (on `BaseEmbeddingBackend`): concrete `embed(...)`; `_embed_cached(utterances, prompt)`;
  `_to_tensor(embeddings) -> torch.Tensor`; abstract `_embed_uncached(utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]`; class attrs `config: EmbedderConfig`, `supports_cache: bool = True`.

- [ ] **Step 1: Rewrite `base.py`** to the following full content

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np

from ._sqlite_cache import get_embedding_cache, utterance_key

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    import torch

    from autointent.configs import EmbedderConfig, TaskTypeEnum


class BaseEmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    config: EmbedderConfig
    supports_training: bool = False
    supports_cache: bool = True

    @abstractmethod
    def __init__(self, config: EmbedderConfig) -> None:
        """Initialize the embedding backend with configuration."""
        ...

    @abstractmethod
    def clear_ram(self) -> None:
        """Clear the backend from RAM."""
        ...

    @overload
    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, *, return_tensors: Literal[True]
    ) -> torch.Tensor: ...

    @overload
    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, *, return_tensors: Literal[False] = False
    ) -> npt.NDArray[np.float32]: ...

    def embed(
        self,
        utterances: list[str],
        task_type: TaskTypeEnum | None = None,
        return_tensors: bool = False,
    ) -> npt.NDArray[np.float32] | torch.Tensor:
        """Calculate embeddings for a list of utterances, using a per-utterance cache.

        Empty input, ``use_cache=False``, or a backend that opts out of caching
        (``supports_cache=False``) bypasses the cache and calls ``_embed_uncached``
        directly, preserving each backend's existing empty-input behavior.

        Args:
            utterances: List of input texts to calculate embeddings for.
            task_type: Type of task for which embeddings are calculated.
            return_tensors: If True, return a PyTorch tensor; otherwise, a numpy array.

        Returns:
            A numpy array or PyTorch tensor of embeddings.
        """
        prompt = self.config.get_prompt(task_type)
        if not utterances or not self.config.use_cache or not self.supports_cache:
            embeddings = self._embed_uncached(utterances, prompt)
        else:
            embeddings = self._embed_cached(utterances, prompt)
        if return_tensors:
            return self._to_tensor(embeddings)
        return embeddings

    def _embed_cached(self, utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:
        """Embed via the SQLite per-utterance cache: reuse hits, compute only misses."""
        cache = get_embedding_cache()
        model_hash = self.get_hash()
        keys = [utterance_key(model_hash, utterance, prompt) for utterance in utterances]
        unique_keys = list(dict.fromkeys(keys))
        cached = cache.get_many(model_hash, unique_keys)
        missing = [key for key in unique_keys if key not in cached]
        if missing:
            key_to_utterance: dict[str, str] = {}
            for utterance, key in zip(utterances, keys, strict=True):
                if key in cached or key in key_to_utterance:
                    continue
                key_to_utterance[key] = utterance
            missing_utterances = [key_to_utterance[key] for key in missing]
            computed = self._embed_uncached(missing_utterances, prompt)
            new_entries = {key: computed[index] for index, key in enumerate(missing)}
            cache.set_many(model_hash, new_entries)
            cached.update(new_entries)
        return cast("npt.NDArray[np.float32]", np.stack([cached[key] for key in keys]))

    @abstractmethod
    def _embed_uncached(self, utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:
        """Compute embeddings WITHOUT caching, returning a ``(N, dim)`` float32 array.

        The backend applies ``prompt`` in its own way (ST passes it to ``encode``;
        OpenAI/vLLM prepend it; HashingVectorizer ignores it). Each backend keeps its
        current empty-input behavior here (ST/OpenAI/vLLM raise; HV returns ``(0, dim)``).
        """
        ...

    def _to_tensor(self, embeddings: npt.NDArray[np.float32]) -> torch.Tensor:
        """Convert a numpy embedding matrix to a torch tensor (CPU by default)."""
        import torch

        return torch.from_numpy(embeddings)

    @abstractmethod
    def similarity(
        self, embeddings1: npt.NDArray[np.float32], embeddings2: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Calculate similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings (size n).
            embeddings2: Second set of embeddings (size m).

        Returns:
            A numpy array of similarities (size n x m).
        """
        ...

    @abstractmethod
    def get_hash(self) -> int:
        """Compute a hash value for the backend configuration and model state.

        Returns:
            The hash value of the backend.
        """
        ...

    @abstractmethod
    def dump(self, path: Path) -> None:
        """Save the backend state to disk.

        Args:
            path: Path to the directory where the backend will be saved.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> BaseEmbeddingBackend:
        """Load the backend state from disk.

        Args:
            path: Path to the directory where the backend is stored.

        Returns:
            Loaded backend instance.
        """
        ...
```

- [ ] **Step 2: Migrate `sentence_transformers.py`**

  1. Remove the import `from .utils import get_embeddings_path` (line ~22).
  2. Add a narrowing class annotation just below the class docstring, beside `_model`:
     ```python
     class SentenceTransformerEmbeddingBackend(BaseEmbeddingBackend):
         """SentenceTransformer-based embedding backend implementation."""

         supports_training: bool = True
         config: SentenceTransformerEmbeddingConfig
         _model: SentenceTransformer | None
     ```
  3. Delete the entire `embed` method **and its two `@overload` stubs** (lines ~165–254) and replace with `_embed_uncached` + a `_to_tensor` override:
     ```python
     def _embed_uncached(self, utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:
         """Compute SentenceTransformer embeddings without caching."""
         if len(utterances) == 0:
             msg = "Empty input"
             logger.error(msg)
             raise ValueError(msg)

         model = self._load_model()
         logger.debug(
             "Calculating embeddings with model %s, batch_size=%d, max_seq_length=%s, embedder_device=%s, prompt=%s",
             self.config.model_name,
             self.config.batch_size,
             str(self.config.tokenizer_config.max_length),
             self.config.device,
             prompt,
         )
         if self.config.tokenizer_config.max_length is not None:
             model.max_seq_length = self.config.tokenizer_config.max_length

         embeddings = cast(
             "npt.NDArray[np.float32]",
             model.encode(
                 utterances,
                 convert_to_numpy=True,
                 batch_size=self.config.batch_size,
                 normalize_embeddings=True,
                 prompt=prompt,
             ),
         )
         return embeddings.astype(np.float32, copy=False)

     def _to_tensor(self, embeddings: npt.NDArray[np.float32]) -> torch.Tensor:
         """Convert to a tensor on the configured device (preserves prior cache-hit behavior)."""
         device = self.config.device or "cpu"
         return torch.from_numpy(embeddings).to(device)
     ```
  **Imports to remove (ruff F401):** drop `Literal, overload` from the `typing` import (keep `TYPE_CHECKING, cast`); drop `TaskTypeEnum` from the `if TYPE_CHECKING:` block (the old `embed` signature was its only user). **Keep** `cast` and `torch` (used by `_embed_uncached`/`_to_tensor`/`clear_ram`/`_set_training_seed`) and `npt`.

- [ ] **Step 3: Migrate `openai.py`**

  1. Remove `from .utils import get_embeddings_path` (line ~20).
  2. Add narrowing annotation under the class docstring:
     ```python
     class OpenaiEmbeddingBackend(BaseEmbeddingBackend):
         """OpenAI-based embedding backend implementation."""

         config: OpenaiEmbeddingConfig
         _client: openai.OpenAI | None = None
         _async_client: openai.AsyncOpenAI | None = None
     ```
  3. Replace the `embed` method **and its two `@overload` stubs** (lines ~169–241) with:
     ```python
     def _embed_uncached(self, utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:
         """Compute OpenAI embeddings without caching."""
         if len(utterances) == 0:
             msg = "Empty input"
             logger.error(msg)
             raise ValueError(msg)

         if prompt:
             utterances = [f"{prompt} {utterance}" for utterance in utterances]

         logger.debug(
             "Calculating embeddings with OpenAI model %s, batch_size=%d, max_tokens_in_batch=%s, "
             "dimensions=%s, prompt=%s, max_concurrent=%s",
             self.config.model_name,
             self.config.batch_size,
             str(self.config.max_tokens_in_batch),
             str(self.config.dimensions),
             prompt,
             self.config.max_concurrent,
         )

         if self.config.max_concurrent is not None:
             return self._process_embeddings_async(utterances)
         return self._process_embeddings_sync(utterances)
     ```
  **Imports to remove (ruff F401):** drop `Literal, overload` from the `typing` import (keep `cast`, used by `similarity`); remove `import torch` (line ~13 — only the old `embed` used it); drop `TaskTypeEnum` from the `if TYPE_CHECKING:` block. **Keep** `np`, `npt`, and `Hasher` (used by `get_hash`).

- [ ] **Step 4: Migrate `vllm.py`**

  1. Remove `from .utils import get_embeddings_path` (line ~17).
  2. Add narrowing annotation under the class docstring:
     ```python
     class VllmEmbeddingBackend(BaseEmbeddingBackend):
         """vLLM-based embedding backend implementation."""

         supports_training: bool = False
         config: VllmEmbeddingConfig
     ```
  3. Replace the `embed` method (lines ~80–139, no overloads in this file) with:
     ```python
     def _embed_uncached(self, utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:
         """Compute vLLM embeddings without caching."""
         if len(utterances) == 0:
             msg = "Empty input"
             logger.error(msg)
             raise ValueError(msg)

         if prompt:
             utterances = [f"{prompt} {utterance}" for utterance in utterances]

         model = self._load_model()
         logger.debug(
             "Calculating embeddings with vLLM model %s, batch_size=%d",
             self.config.model_name,
             self.config.batch_size,
         )
         outputs = model.encode(utterances, pooling_task="embed", **self.config.extra_encode_kwargs)
         all_embeddings = [output.outputs.embedding for output in outputs]
         return np.array(all_embeddings, dtype=np.float32)
     ```
  **Imports to remove (ruff F401):** drop `TaskTypeEnum` from the `if TYPE_CHECKING:` block (the old `embed` signature was its only user). **Keep** `cast` (used by `similarity`), `torch` (used by `clear_ram`), `np`, `npt`, and `Hasher` (used by `get_hash`). (This file has no `embed` overloads to remove.)

- [ ] **Step 5: Migrate `hashing_vectorizer.py`**

  1. Add narrowing annotation + `supports_cache = False` under the class docstring:
     ```python
     class HashingVectorizerEmbeddingBackend(BaseEmbeddingBackend):
         """HashingVectorizer-based embedding backend implementation.

         This backend uses sklearn's HashingVectorizer for fast, stateless text vectorization.
         Ideal for testing as it requires no model downloads and is very fast.
         """

         supports_training: bool = False
         supports_cache: bool = False
         config: HashingVectorizerEmbeddingConfig
     ```
  2. Replace the `embed` method **and its two `@overload` stubs** (lines ~77–109) with:
     ```python
     def _embed_uncached(self, utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:  # noqa: ARG002
         """Compute HashingVectorizer embeddings (prompt is ignored; never cached)."""
         embeddings_sparse = self._vectorizer.transform(utterances)
         embeddings: npt.NDArray[np.float32] = embeddings_sparse.toarray().astype(np.float32)
         return embeddings
     ```
  **Imports to remove (ruff F401):** drop `Literal, overload` from the `typing` import; remove `import torch` (only the old `embed` used it); remove `from autointent.configs import TaskTypeEnum` (a runtime import on line ~15, now unused). **Keep** `np`, `npt`, and `Hasher` (used by `get_hash`). `# noqa: ARG002` on `_embed_uncached` covers the unused `prompt` parameter.

- [ ] **Step 6: Migrate `tests/_fixtures/fake_openai_embedding.py`**

  1. Add narrowing annotation under the class docstring:
     ```python
     class FakeOpenaiEmbeddingBackend(BaseEmbeddingBackend):
         """In-process stand-in for OpenaiEmbeddingBackend. ... (keep existing docstring)"""

         supports_training = False
         config: OpenaiEmbeddingConfig
     ```
     (`OpenaiEmbeddingConfig` is already imported under `TYPE_CHECKING` in this file.)
  2. Replace the `embed` method **and its two `@overload` stubs** with:
     ```python
     def _embed_uncached(self, utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:
         # Touch the lazy attribute so test_client_lazy_loading observes the transition.
         self._client = self._client or object()
         dim = self.config.dimensions or 1536
         # Prompt is already resolved by the base; mirror BaseEmbedderConfig.get_prompt seeding.
         seed_extra = f"{self.config.model_name}|{prompt or ''}"
         vectors: npt.NDArray[np.float32] = np.stack(
             [_seeded_vector(text, dim, seed_extra=seed_extra) for text in utterances]
         )
         return vectors
     ```
  **Imports to remove (ruff F401):** drop `Literal, overload` from the `typing` import; remove `import torch` (only the old `embed` used it); drop `TaskTypeEnum` from the `if TYPE_CHECKING:` block. **Keep** `np`, `npt`, `pytest` (used by the `patch_openai_embedding_backend` fixture), `hashlib`, `json`, and `OpenaiEmbeddingConfig`. The fake now inherits `embed`/`_to_tensor` from the base.

- [ ] **Step 7: Delete the obsolete util**

```bash
git rm src/autointent/_wrappers/embedder/utils.py
```

(Confirm no remaining importer: `grep -rn get_embeddings_path src tests` returns nothing.)

- [ ] **Step 8: Local gate**

Run: `grep -rn "get_embeddings_path" src tests` → expect no output.
Run: `ruff check src/autointent/_wrappers/embedder tests/_fixtures/fake_openai_embedding.py`
Run: `mypy src/autointent tests`
Expected: all clean. (mypy must show no `attr-defined` on `self.config.*`; this validates the narrowing.)

- [ ] **Step 9: (reference) CI test commands**

On CI: `pytest tests/embedder/test_caching.py tests/embedder/test_hash.py tests/embedder/test_memory.py tests/embedder/test_dump_load.py tests/embedder/test_openai_backend.py tests/embedder/test_prompts.py -v` → expect PASS (consistency preserved).

- [ ] **Step 10: Commit**

```bash
git add -A src/autointent/_wrappers/embedder tests/_fixtures/fake_openai_embedding.py
git commit -m "refactor(embedder): lift embedding cache into a per-utterance template method

Move the triplicated .npy cache block out of the ST/OpenAI/vLLM backends into a
single BaseEmbeddingBackend.embed template backed by SQLiteEmbeddingCache. Backends
now implement _embed_uncached; HashingVectorizer opts out via supports_cache=False.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Per-utterance behavior tests (reuse / dedup / order / empty)

**Files:**
- Modify: `tests/embedder/test_caching.py` (append new tests)

**Interfaces:**
- Consumes: `Embedder`, `create_sentence_transformer_config`, the global isolation fixture (Task 1), the SQLite store (Task 2), the refactored backends (Task 3).

- [ ] **Step 1: Append the new tests** to `tests/embedder/test_caching.py`

Add these runtime imports at the top (next to the existing ones — `os`, `sqlite3`, and `Path` are used at runtime here via `Path(os.environ[...])`):

```python
import os
import sqlite3
from pathlib import Path

from autointent.configs import HashingVectorizerEmbeddingConfig
```

And add `import numpy.typing as npt` to the file's existing `if TYPE_CHECKING:` block (used only in the `spy` annotation below).

Append:

```python
def _embedding_row_count() -> int:
    db_path = Path(os.environ["AUTOINTENT_CACHE_DIR"]) / "embeddings.db"
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])


class TestPerUtteranceCaching:
    """Per-utterance keying: shared utterances are stored once and reused across calls."""

    def test_overlapping_calls_store_each_utterance_once(self) -> None:
        config = create_sentence_transformer_config(use_cache=True)
        embedder = Embedder(config)

        embedder.embed(["alpha", "beta"])
        embedder.embed(["beta", "gamma"])  # 'beta' overlaps

        # Whole-list keying would store 2 list blobs; per-utterance stores 3 rows.
        assert _embedding_row_count() == 3

    def test_duplicate_in_list_computed_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = create_sentence_transformer_config(use_cache=True)
        embedder = Embedder(config)
        backend = embedder._backend

        computed: list[list[str]] = []
        original = backend._embed_uncached

        def spy(utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:
            computed.append(list(utterances))
            return original(utterances, prompt)

        monkeypatch.setattr(backend, "_embed_uncached", spy)

        result = embedder.embed(["dup", "dup"])

        assert result.shape[0] == 2
        np.testing.assert_array_equal(result[0], result[1])
        assert computed == [["dup"]]  # computed only once

    def test_order_preserved_after_partial_hit(self) -> None:
        config = create_sentence_transformer_config(use_cache=True)
        embedder = Embedder(config)

        first = embedder.embed(["one", "two", "three"])
        second = embedder.embed(["three", "one", "two"])  # reordered, fully cached

        np.testing.assert_allclose(second[0], first[2], rtol=1e-5)
        np.testing.assert_allclose(second[1], first[0], rtol=1e-5)
        np.testing.assert_allclose(second[2], first[1], rtol=1e-5)

    def test_empty_input_hashing_vectorizer_returns_empty(self) -> None:
        embedder = Embedder(HashingVectorizerEmbeddingConfig(n_features=512, use_cache=True))
        result = embedder.embed([])
        assert result.shape == (0, 512)

    def test_empty_input_sentence_transformer_raises(self) -> None:
        embedder = Embedder(create_sentence_transformer_config(use_cache=True))
        with pytest.raises(ValueError, match="Empty input"):
            embedder.embed([])
```

- [ ] **Step 2: Local gate**

Run: `ruff check tests/embedder/test_caching.py`
Run: `mypy src/autointent tests`
Expected: clean. (Accessing `embedder._backend` / `backend._embed_uncached` is fine in tests; if ruff flags `SLF001` here, add `# noqa: SLF001` on those lines — the tests/ ruff profile typically already relaxes it.)

- [ ] **Step 3: (reference) CI test command**

On CI: `pytest tests/embedder/test_caching.py -v` → expect PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/embedder/test_caching.py
git commit -m "test(cache): cover per-utterance reuse, dedup, order, and empty input

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md` (repo root)

- [ ] **Step 1: Insert an Unreleased section** at the top of `CHANGELOG.md`, immediately after the intro paragraph and before `## [0.3.2] — 2026-06-22`:

```markdown
## [Unreleased]

### Features

- **Embedding cache rewritten on SQLite with per-utterance keys.** Embeddings are now cached one row per `(model, utterance, prompt)` in a single SQLite database (`<cache_dir>/embeddings.db`) instead of one `.npy` file per call. Utterances shared across calls are embedded and stored once, so overlapping calls reuse the overlap — removing the old whole-list-or-nothing cache misses and the unbounded `.npy` inode growth. Writes are atomic and safe for concurrent processes/threads on one host (WAL).
- **`AUTOINTENT_CACHE_DIR`** environment variable to relocate the on-disk cache (defaults to the OS cache dir). It currently governs the embedding cache only; the structured-output cache is unchanged.

### Notes

- The new cache uses a different key scheme, so existing `.npy` embedding caches are not reused (a one-time recompute on first run). The old `embeddings/` directory is left untouched and may be deleted manually.

---
```

- [ ] **Step 2: Local gate**

Run: `git diff --stat CHANGELOG.md` → expect only additions. (No ruff/mypy on Markdown.)

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): note SQLite embedding cache and AUTOINTENT_CACHE_DIR

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final verification (before opening the draft PR)

- [ ] **Whole-tree static gate:** `ruff check .` and `mypy src/autointent tests` → both clean.
- [ ] **Grep guards:** `grep -rn "get_embeddings_path" src tests` (empty); `grep -rn "from .utils import" src/autointent/_wrappers/embedder` (empty).
- [ ] **Push branch + open draft PR**, then inspect CI (the only place pytest runs). Iterate on CI failures by pushing fixes. Key CI signals to watch: the `tests/embedder/*` suite (consistency + new behavior), the 85% **combined** coverage floor (Task 2 tests cover the main `_sqlite_cache.py` branches; a few defensive branches — the WAL-unavailable debug log, the double-checked-lock re-entry, the `_deserialize` `except` — are hard to hit single-threaded and may stay uncovered, which is fine against the combined total per the spec), and mypy on Python 3.10.

---

## Self-Review (completed by plan author)

**Spec coverage:** §4.1 cache-dir → Task 1. §4.2 utterance_key → Task 2. §4.3 SQLite store (schema, pragmas, versioning, degradation, model_hash filter, chunking, memoized accessor) → Task 2. §4.4 template method + `supports_cache` + per-subclass `config` narrowing + `_embed_uncached` per backend + fake → Task 3. §6.1 unit tests → Task 2. §6.2 global isolation fixture → Task 1. §6.3 reuse/dedup/order/empty tests → Task 4. §7 file list (incl. `utils.py` removal) → Tasks 1–4. CHANGELOG → Task 5. All covered.

**Placeholder scan:** No TBD/TODO; all steps carry complete code or exact commands.

**Type consistency:** `get_many(model_hash, keys)` / `set_many(model_hash, entries)` / `utterance_key(model_hash, utterance, prompt)` / `_embed_uncached(utterances, prompt)` / `_to_tensor(embeddings)` / `get_embedding_cache()` are used identically across Tasks 2, 3, and 4. `supports_cache` set on base (True) and HV (False) consistently.
