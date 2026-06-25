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
