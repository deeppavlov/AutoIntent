# Design: SQLite per-utterance embedding cache

**Date:** 2026-06-25
**Status:** Approved (scope decisions confirmed by maintainer)
**Scope owner:** voorhs

## 1. Motivation

AutoIntent caches embeddings as one NumPy `.npy` file per `embed()` call, named by
`hash(model_identity + entire_utterance_list + prompt)`, under
`appdirs.user_cache_dir("autointent")/embeddings/`. This has three structural problems:

1. **Whole-list keying = zero reuse.** Two calls whose utterance lists differ by even one
   element (reorder, add, drop) produce completely different files and full cache misses.
   A shared utterance embedded in 50 different lists is recomputed and re-stored 50 times.
2. **Inode explosion.** Every distinct list is its own file. Long-running optimization with many
   search-space points and folds produces thousands of `.npy` files with no index, no bound,
   no eviction.
3. **Triplicated, non-atomic cache code.** The identical read/key/write block is copy-pasted
   across three backends (`sentence_transformers.py`, `openai.py`, `vllm.py`). Writes are a bare
   `np.save` with no atomicity and no concurrency story for parallel Optuna workers.

The fix is two coordinated changes:

- **Per-utterance keying:** key each utterance by `hash(model_identity + utterance + prompt)`,
  one row per utterance. Shared utterances are stored once; a call that overlaps a previous call
  reuses the overlap (partial hits).
- **SQLite store:** move the embedding cache behind a single SQLite database. One file instead of
  K inodes, atomic transactions, indexed point lookups, safe concurrent access (WAL), and the
  schema groundwork for future eviction/TTL.

**Honest scoping (per maintainer):** the warm read path is already sub-millisecond; SQLite will
**not** make cache *hits* faster. Its value is **correctness** (atomic writes), **operability**
(one file, future eviction, concurrency), and **enabling per-utterance keys without an inode
explosion**. We justify it by operational pain, not hit latency.

## 2. Goals and non-goals

### Goals
- Replace the `.npy`-per-list embedding cache with a single SQLite database, keyed **per utterance**.
- Deduplicate within and across calls: a given `(model, utterance, prompt)` is computed and stored once.
- Eliminate the triplicated cache code by lifting caching into `BaseEmbeddingBackend` as a template method.
- Add a configurable cache location via the `AUTOINTENT_CACHE_DIR` environment variable + helper,
  defaulting to today's `appdirs.user_cache_dir("autointent")`.
- Lay **schema groundwork** for eviction (`created_at`, `last_accessed`, `size_bytes`, `model_hash`
  columns + indexes) without changing today's unbounded behavior.
- Safe concurrent access from multiple processes (parallel Optuna trials) and threads.
- Graceful degradation: a cache I/O failure logs and falls back to recompute; it never breaks `embed()`.

### Non-goals (explicitly out of scope)
- **The structured-output / LLM cache** (`generation/_cache.py`) is **not touched.** It keeps its
  current directory-per-entry format. (It *may* adopt `get_cache_dir()` in a later PR; not here.)
- **No active eviction policy.** No size cap, no TTL enforcement, no LRU sweeping. Columns + indexes
  only. The cache stays unbounded by default, matching today.
- **No migration of existing `.npy` caches.** Fresh start. The old whole-list hashes cannot be
  decomposed into per-utterance rows (the original utterances were never stored), so migration is
  infeasible. Old files are left as orphans (the user may delete them).
- **No change to the public `Embedder.embed` / backend `embed` signatures or return types.**
- **No new third-party dependency.** Uses the Python stdlib `sqlite3`.
- **No LMDB / Parquet / memmap sidecar.** Per-utterance vectors are small (≈1.5–4 KB); a BLOB column
  is the right fit. A sidecar is noted as possible future work only.

## 3. Current state (reference)

- `BaseEmbeddingBackend` (`_wrappers/embedder/base.py`): abstract `embed`, `get_hash`, `similarity`,
  `clear_ram`, `dump`, `load`.
- Four backends implement `embed` independently: `SentenceTransformerEmbeddingBackend`,
  `OpenaiEmbeddingBackend`, `VllmEmbeddingBackend`, `HashingVectorizerEmbeddingBackend`.
  The first three contain the duplicated cache block; HashingVectorizer has no cache block and is
  always used with `use_cache=False`.
- Cache key: `Hasher()` (xxhash-64, pickle-based) over `get_hash()` (model identity) + the whole
  `utterances` list + `prompt` (if non-empty).
- `get_hash()` differs per backend (model name + HF commit SHA + max_length for ST; model name +
  dimensions + max_tokens for OpenAI; model name + max_model_len for vLLM; config params for HV).
- Prompt handling differs: ST passes `prompt=` to `model.encode`; OpenAI and vLLM **prepend**
  `f"{prompt} {utterance}"` before encoding.
- Cache path: `get_embeddings_path(hexdigest)` → `user_cache_dir("autointent")/embeddings/<hex>.npy`.
  Only the three backends import it; `utils.py` contains nothing else.
- `FakeOpenaiEmbeddingBackend` (`tests/_fixtures/fake_openai_embedding.py`) subclasses
  `BaseEmbeddingBackend` and overrides `embed()` (no caching); an autouse fixture swaps it in for the
  real OpenAI backend across `tests/embedder/`.
- **Test gap:** `tests/embedder/test_caching.py` runs with `use_cache=True` but no fixture redirects
  the cache directory, so it writes to the **real OS cache dir**. The new config seam will let us
  isolate it.

## 4. Design

### 4.1 Cache directory resolution — `autointent/_cache_dir.py`

```python
def get_cache_dir() -> Path:
    """Base directory for autointent on-disk caches.

    Honors the AUTOINTENT_CACHE_DIR environment variable; otherwise falls back to
    appdirs.user_cache_dir("autointent"). Resolved fresh on each call so tests and
    parallel workers can point it at an isolated directory via the env var.
    """
    override = os.environ.get("AUTOINTENT_CACHE_DIR")
    return Path(override) if override else Path(user_cache_dir("autointent"))
```

- `AUTOINTENT_CACHE_DIR` matches the existing `AUTOINTENT_`-prefixed env convention
  (`AUTOINTENT_PATH`, `AUTOINTENT_EXTRA_VALIDATION`, server `env_prefix="AUTOINTENT_"`).
- The embedding DB lives at `get_cache_dir() / "embeddings.db"` (replacing the `embeddings/` dir of
  `.npy` files). WAL adds `embeddings.db-wal` and `embeddings.db-shm` sidecars — still ~3 files vs.
  K inodes.
- This helper is used **only** by the embedding path in this PR. The structured-output cache keeps
  calling `user_cache_dir("autointent")` directly (untouched, per scope).

### 4.2 Per-utterance key — `autointent/_wrappers/embedder/_sqlite_cache.py`

```python
def utterance_key(model_hash: int, utterance: str, prompt: str | None) -> str:
    hasher = Hasher()
    hasher.update(model_hash)
    hasher.update(utterance)
    if prompt:
        hasher.update(prompt)
    return hasher.hexdigest()
```

Mirrors the existing scheme but on a single string instead of the whole list. `model_hash` is the
backend's existing `get_hash()` (so all model-identity stability work from #321/#334 is reused
unchanged). `prompt` is the resolved task prompt, included only when non-empty (matches current
behavior). The key is the original utterance text — **not** the prompt-prepended form — so a backend's
internal prompt application stays an implementation detail of `_embed_uncached`.

**Backward compatibility:** this is a brand-new keying scheme and a brand-new store. All existing
`.npy` caches are invalid and ignored (the approved fresh start). First run after upgrade recomputes;
subsequent runs hit the new cache.

### 4.3 SQLite store — `SQLiteEmbeddingCache`

**Schema (version 1):**

```sql
CREATE TABLE IF NOT EXISTS embeddings (
    key           TEXT    PRIMARY KEY,   -- utterance_key() hexdigest
    model_hash    TEXT    NOT NULL,      -- str(get_hash()); enables per-model purge
    dim           INTEGER NOT NULL,      -- vector length
    vector        BLOB    NOT NULL,      -- float32 bytes, C-contiguous, length dim
    size_bytes    INTEGER NOT NULL,      -- len(vector blob); eviction groundwork
    created_at    REAL    NOT NULL,      -- time.time() at insert
    last_accessed REAL    NOT NULL       -- = created_at at insert (see note)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_last_accessed ON embeddings(last_accessed);
CREATE INDEX IF NOT EXISTS idx_embeddings_created_at    ON embeddings(created_at);
CREATE INDEX IF NOT EXISTS idx_embeddings_model_hash    ON embeddings(model_hash);
```

`model_hash` is stored as **TEXT** because `Hasher.intdigest()` is an unsigned 64-bit value that can
exceed SQLite's signed-64-bit `INTEGER` range. Vectors are stored as raw **float32** bytes
(`np.ascontiguousarray(vec, dtype=np.float32).tobytes()`), the dtype used everywhere in the codebase;
reconstructed with `np.frombuffer(blob, dtype=np.float32)` and validated against `dim`.

**Connection pragmas:**
- `PRAGMA journal_mode=WAL` — set once at schema init, persists in the DB file. Enables concurrent
  readers with a single writer (the parallel-worker use case).
- `PRAGMA busy_timeout=<N ms>` — per connection; writers wait instead of raising
  "database is locked". Default 5000 ms.
- `PRAGMA synchronous=NORMAL` — safe with WAL, faster than FULL; on power loss you may lose the last
  transaction but the DB does not corrupt — acceptable for a cache.

**Schema versioning:** `PRAGMA user_version` holds `SCHEMA_VERSION` (1). On open, if the stored
version differs from the code's version, the table is dropped and recreated (cache rebuild). This is
the forward-migration story: a schema bump = automatic fresh start, no manual cleanup.

**Connection model:** every public method opens a **short-lived connection** via a private
`_connect()` context manager that applies `busy_timeout`/`synchronous` and closes on exit. No
connection is shared across threads, so the cache is inherently thread-safe; WAL handles
inter-process safety. `embed()` is coarse-grained (one `get_many` + one `set_many` per call), so
per-call connection overhead is negligible next to model inference.

**Instance lifecycle:** a module-level `get_embedding_cache() -> SQLiteEmbeddingCache` resolves the DB
path from `get_cache_dir()` and returns a cache instance **memoized by resolved path** (dict + lock).
Schema init (CREATE TABLE / version check / WAL) runs once per path per process. Tests that set
`AUTOINTENT_CACHE_DIR` to a fresh `tmp_path` naturally get a distinct, isolated instance — no global
reset needed.

**Public API:**

```python
class SQLiteEmbeddingCache:
    def __init__(self, db_path: Path) -> None: ...
        # stores path; lazily ensures parent dir + schema on first connect

    def get_many(self, keys: list[str]) -> dict[str, npt.NDArray[np.float32]]:
        # SELECT key, vector, dim WHERE key IN (...), chunked to stay under
        # SQLITE_MAX_VARIABLE_NUMBER (chunk size 900). Returns only found keys,
        # each reconstructed to a (dim,) float32 array. Read-only: does NOT
        # update last_accessed (avoids read amplification; see note).

    def set_many(self, model_hash: int, entries: dict[str, npt.NDArray[np.float32]]) -> None:
        # INSERT OR IGNORE within a single transaction (executemany).
        # OR IGNORE => two workers computing the same key never conflict, and
        # an existing entry is never overwritten (entries are deterministic).
        # created_at = last_accessed = time.time(); size_bytes = len(blob).

    # graceful degradation: get_many returns {} and set_many is a no-op (both log
    # a warning) if a sqlite3.Error or corruption is encountered. The cache never
    # raises into embed().
```

**`last_accessed` note:** populated at insert but **not** updated on read. Updating it per read would
turn every cache hit into a write, defeating the WAL concurrency benefit. The column exists so a
future eviction PR can choose its own access-tracking policy; for now it equals `created_at`. This is
deliberate groundwork, documented as such.

### 4.4 Backend refactor — template method in `BaseEmbeddingBackend`

Lift the whole cache+dedup+reassemble flow into the base class once; backends implement only the pure
model call.

```python
class BaseEmbeddingBackend(ABC):
    def embed(self, utterances, task_type=None, return_tensors=False):
        if not utterances:
            raise ValueError("Empty input")
        prompt = self.config.get_prompt(task_type)
        if not self.config.use_cache:
            arr = self._embed_uncached(utterances, prompt)
        else:
            arr = self._embed_cached(utterances, prompt)
        return self._to_tensor(arr) if return_tensors else arr

    def _embed_cached(self, utterances, prompt) -> npt.NDArray[np.float32]:
        cache = get_embedding_cache()
        model_hash = self.get_hash()
        keys = [utterance_key(model_hash, u, prompt) for u in utterances]
        unique_keys = list(dict.fromkeys(keys))          # de-dup, preserve order
        cached = cache.get_many(unique_keys)
        missing = [k for k in unique_keys if k not in cached]
        if missing:
            key_to_utt = {}
            for u, k in zip(utterances, keys):
                if k in cached or k in key_to_utt:
                    continue
                key_to_utt[k] = u
            missing_utts = [key_to_utt[k] for k in missing]
            computed = self._embed_uncached(missing_utts, prompt)   # (M, dim) float32
            new_entries = {k: computed[i] for i, k in enumerate(missing)}
            cache.set_many(model_hash, new_entries)
            cached.update(new_entries)
        return np.stack([cached[k] for k in keys])        # (N, dim), original order

    @abstractmethod
    def _embed_uncached(self, utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:
        """Compute embeddings WITHOUT caching. Always returns a (N, dim) float32 array.
        The backend applies `prompt` in its own way (ST: pass to encode; OpenAI/vLLM: prepend)."""

    def _to_tensor(self, arr: npt.NDArray[np.float32]) -> "torch.Tensor":
        import torch
        return torch.from_numpy(arr)        # ST overrides to move to its device
```

- `embed()` becomes **concrete** (one implementation, the overloaded signatures preserved). `get_hash`,
  `similarity`, `clear_ram`, `dump`, `load` stay abstract. `_embed_uncached` is the new abstract method.
- Each backend's `embed` body collapses to a `_embed_uncached` that **always returns float32 numpy**:
  - **ST:** set `max_seq_length`, `model.encode(..., convert_to_numpy=True, normalize_embeddings=True,
    prompt=prompt)`, cast float32. Override `_to_tensor` to `torch.from_numpy(arr).to(device or "cpu")`
    (preserves the current cache-hit device behavior).
  - **OpenAI:** prepend prompt if present, run sync/async path, return the float32 array.
  - **vLLM:** prepend prompt if present, `model.encode`, stack float32.
  - **HashingVectorizer:** ignore prompt (as today), transform → dense float32. (Gains `use_cache`
    support for free; remains off by default.)
- **`FakeOpenaiEmbeddingBackend`** is migrated to implement `_embed_uncached` (its current `embed`
  body, minus tensor conversion, returning numpy; the lazy `_client` touch moves into it) and **inherits**
  the template `embed`. This gives the fake genuine cache coverage in tests (made hermetic by the
  cache-dir isolation fixture) and keeps a single embed code path.

**Tensor/device semantics:** the cache always stores/reconstructs CPU float32. When `return_tensors=True`,
the base converts via `_to_tensor`. This is equivalent to the current cache-hit behavior
(`torch.from_numpy(...)[.to(device)]`). The only nuance: previously, an ST **cache-miss** with
`return_tensors=True` returned the raw on-device encode tensor; now it round-trips through CPU numpy and
back to the device. Values are identical (float32); this is an intentional, documented unification.

**Empty-input unification:** the base raises `ValueError` on empty input for **all** backends. Three of
four already did; HashingVectorizer previously returned an empty array. This is a documented, minor
behavior unification (no known caller embeds an empty list).

### 4.5 Data flow (one `embed(["a","b","a","c"])` call, partial hit)

1. Resolve `prompt` from `task_type`.
2. `use_cache=False` → call `_embed_uncached(["a","b","a","c"], prompt)`, convert if tensor, return.
3. `use_cache=True`:
   - `model_hash = get_hash()`; `keys = [k_a, k_b, k_a, k_c]`; `unique = [k_a, k_b, k_c]`.
   - `get_many([k_a,k_b,k_c])` → say `{k_a: v_a}` (a was cached before). `missing = [k_b, k_c]`.
   - `_embed_uncached(["b","c"], prompt)` → `[v_b, v_c]`. `set_many(model_hash, {k_b:v_b, k_c:v_c})`.
   - `cached = {k_a:v_a, k_b:v_b, k_c:v_c}`. Reassemble `np.stack([v_a, v_b, v_a, v_c])` → (4, dim).
   - Convert to tensor if requested; return.

## 5. Error handling and robustness

- **Cache read failure** (locked beyond busy_timeout, corruption, malformed blob): `get_many` logs a
  warning and returns `{}` → everything recomputed. `embed()` still succeeds.
- **Cache write failure**: `set_many` logs a warning and returns → embeddings returned uncached.
- **Corrupted DB file / schema version mismatch**: detected at connect/schema-ensure; the table is
  dropped and recreated (rebuild). If the file itself is unreadable, the cache degrades to no-op for
  the process (logged) rather than crashing.
- **Dimension mismatch on read** (`len(blob)/4 != dim`): treat the row as a miss (log, skip), recompute.
- **Concurrency**: WAL + `busy_timeout` + `INSERT OR IGNORE` make concurrent multi-process trials and
  multi-thread access safe without external locking.

## 6. Testing strategy

All tests are verified **via CI on the draft PR** (maintainer rule: no heavy/exhaustive pytest locally;
ruff + mypy run locally). New tests are designed to be fast and to **not** download models.

### 6.1 New unit tests — `tests/embedder/test_sqlite_cache.py` (pure Python, no ML)
- `set_many` then `get_many` round-trips exact float32 bytes; reconstructed shape `(dim,)`.
- Miss returns absent keys; partial hit returns only present keys.
- `INSERT OR IGNORE`: re-inserting an existing key does not overwrite or error.
- Chunking: `get_many` with > 900 keys returns all matches (exercises the IN-chunk loop).
- Schema: WAL enabled; `user_version == SCHEMA_VERSION`; columns/indexes present; mismatched
  `user_version` triggers rebuild.
- Graceful degradation: a corrupted/garbage DB file → `get_many` returns `{}`, `set_many` no-ops,
  no exception.
- `get_cache_dir()`: honors `AUTOINTENT_CACHE_DIR`; falls back to appdirs when unset.
- `utterance_key()`: stable; differs by utterance, by prompt, by model_hash; equal for equal inputs.

### 6.2 Updated integration tests — `tests/embedder/test_caching.py`
- New **autouse** fixture in `tests/embedder/conftest.py` sets `AUTOINTENT_CACHE_DIR` to a per-test
  `tmp_path` (isolates the cache; fixes today's real-OS-cache pollution).
- Keep existing parametrized consistency tests (cache on/off identical results).
- Add **per-utterance reuse** test: embed `["x","y"]` then `["y","z"]`; assert the `y` row is reused
  (e.g. by spying on `_embed_uncached` / `set_many` so only `z` is computed on the second call), and
  assert byte-for-byte equality of the shared `y` vector.
- Add **dedup-within-list** test: embed `["x","x"]`; `_embed_uncached` receives a single `x`; output
  rows 0 and 1 are identical and ordered.
- Add **order-preservation** test: a multi-element list returns rows in input order after a partial hit.
- Keep `test_cache_with_different_prompts` (different prompt ⇒ different key ⇒ different vector).
- ST backend (`sergeyzh/rubert-tiny-turbo`, the pinned tiny model) exercises the real cached path;
  the fake OpenAI backend exercises it too via the inherited template.

### 6.3 Regression / unchanged
- `tests/embedder/test_hash.py` (incl. offline #321 cases) is unaffected — `get_hash()` is unchanged.
- `mypy src/autointent tests` stays green (strict, py3.10): annotate the new module fully; `sqlite3`,
  `numpy.frombuffer/tobytes` are typed.
- Coverage: the new module's branches are covered by 6.1, keeping the 85% combined floor.

## 7. File-by-file change list

**New**
- `src/autointent/_cache_dir.py` — `get_cache_dir()`.
- `src/autointent/_wrappers/embedder/_sqlite_cache.py` — `SQLiteEmbeddingCache`, `utterance_key`,
  `get_embedding_cache`, `SCHEMA_VERSION`.
- `tests/embedder/test_sqlite_cache.py` — unit tests (6.1).

**Modified**
- `src/autointent/_wrappers/embedder/base.py` — concrete `embed`, `_embed_cached`, `_to_tensor`,
  abstract `_embed_uncached`.
- `src/autointent/_wrappers/embedder/sentence_transformers.py` — `embed` → `_embed_uncached`,
  override `_to_tensor`; drop the inline cache block and `get_embeddings_path` import.
- `src/autointent/_wrappers/embedder/openai.py` — `embed` → `_embed_uncached`; drop cache block/import.
- `src/autointent/_wrappers/embedder/vllm.py` — `embed` → `_embed_uncached`; drop cache block/import.
- `src/autointent/_wrappers/embedder/hashing_vectorizer.py` — `embed` → `_embed_uncached`.
- `tests/_fixtures/fake_openai_embedding.py` — `embed` → `_embed_uncached`, inherit template embed.
- `tests/embedder/conftest.py` — autouse `AUTOINTENT_CACHE_DIR` → tmp_path fixture.
- `tests/embedder/test_caching.py` — new reuse/dedup/order tests.

**Removed**
- `src/autointent/_wrappers/embedder/utils.py` `get_embeddings_path` (and the file if it becomes empty).

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Refactor changes per-backend embed behavior subtly | Backends keep their exact encode calls inside `_embed_uncached`; only caching/reassembly moves. Parametrized consistency tests guard cache-on == cache-off. |
| Fake backend inheriting template embed breaks other embedder tests | Cache-dir isolation fixture makes it hermetic; default `use_cache=False` in conftest keeps most tests on the no-cache path; per-change review + CI catch regressions. |
| vLLM path can't run in CI (no GPU) | `_embed_uncached` for vLLM is a thin wrapper; shared cache logic is tested via ST + fake + pure unit tests. vLLM test stays `skipif` as today. |
| SQLite "database is locked" under parallel trials | WAL + `busy_timeout` + `INSERT OR IGNORE`; writes batched in one transaction. |
| `model_hash` int overflowing INTEGER | Stored as TEXT. |
| Unsigned 64-bit key/blob edge cases | Key is a hex string PK; blob is raw float32 bytes with `dim` validation. |
| Empty-input behavior change for HashingVectorizer | Documented unification; no known empty-list caller. |

## 9. Future work (not in this PR)
- Active eviction: size-cap LRU and/or TTL using the groundwork columns (and a read-time
  `last_accessed` update policy).
- Route the structured-output cache through `get_cache_dir()` and/or migrate it onto the same
  SQLite layer.
- Optional LMDB/Parquet/memmap sidecar for very large vector volumes if BLOB storage ever becomes
  a bottleneck.
