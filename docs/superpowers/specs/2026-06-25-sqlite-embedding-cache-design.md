# Design: SQLite per-utterance embedding cache

**Date:** 2026-06-25
**Status:** Approved (scope decisions confirmed by maintainer); revised after adversarial review round 1.
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
  K inodes, atomic transactions, indexed point lookups, safe single-host concurrent access (WAL),
  and the schema groundwork for future eviction/TTL.

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
- Safe concurrent access from multiple processes (parallel Optuna trials) and threads **on one host**.
- Graceful degradation: a cache I/O failure logs and falls back to recompute; it never breaks `embed()`.

### Non-goals (explicitly out of scope)
- **The structured-output / LLM cache** (`generation/_cache.py`) is **not touched.** It keeps its
  current directory-per-entry format and its direct `user_cache_dir("autointent")` calls. It does
  **not** honor `AUTOINTENT_CACHE_DIR` in this PR (documented limitation; a later PR may adopt the helper).
- **No active eviction policy.** No size cap, no TTL enforcement, no LRU sweeping. Columns + indexes
  only. The cache stays unbounded by default, matching today.
- **No migration of existing `.npy` caches.** Fresh start. The old whole-list hashes cannot be
  decomposed into per-utterance rows (the original utterances were never stored), so migration is
  infeasible. Old files are left as orphans (the user may delete them).
- **No change to the public `Embedder.embed` / backend `embed` signatures or return types**, and no
  change to any backend's per-utterance vector values.
- **No new third-party dependency.** Uses the Python stdlib `sqlite3`.
- **No LMDB / Parquet / memmap sidecar.** Per-utterance vectors from the cached backends (ST, OpenAI,
  vLLM) are small (≈1.5–4 KB); a BLOB column is the right fit. The one high-dimensional backend,
  HashingVectorizer (default `n_features = 2**18` ≈ 1 MB/vector), is excluded from caching entirely via
  `supports_cache = False` (§4.4), so it never reaches the BLOB store. A sidecar is future work only.
- **No widening of the hash to 128-bit.** The existing 64-bit `Hasher` (xxh64) keying strength is
  retained (see §4.2 collision discussion); cross-model collisions are additionally defended.

## 3. Current state (reference)

- `BaseEmbeddingBackend` (`_wrappers/embedder/base.py`): abstract `embed`, `get_hash`, `similarity`,
  `clear_ram`, `dump`, `load`. `__init__` is abstract with an empty body; **the ABC does not declare
  a `config` attribute** (each concrete backend assigns `self.config`).
- Four backends implement `embed` independently: `SentenceTransformerEmbeddingBackend`,
  `OpenaiEmbeddingBackend`, `VllmEmbeddingBackend`, `HashingVectorizerEmbeddingBackend`.
  The first three contain the duplicated cache block; HashingVectorizer has no cache block.
  A fifth subclass, `FakeOpenaiEmbeddingBackend` (`tests/_fixtures/fake_openai_embedding.py`), is the
  test stand-in swapped in for the real OpenAI backend across `tests/embedder/` via an autouse fixture.
  **These five are the complete set of `BaseEmbeddingBackend` subclasses** (verified by grep; no
  cross-encoder/ranker/server subclass exists).
- Cache key: `Hasher()` (xxhash-64, pickle-based) over `get_hash()` (model identity) + the whole
  `utterances` list + `prompt` (if non-empty).
- `get_hash()` differs per backend (model name + HF commit SHA + max_length for ST; model name +
  dimensions + max_tokens for OpenAI; model name + max_model_len for vLLM; config params for HV).
- Prompt handling differs: ST passes `prompt=` to `model.encode`; OpenAI and vLLM **prepend**
  `f"{prompt} {utterance}"` before encoding.
- Empty-input behavior differs: ST/OpenAI/vLLM raise `ValueError("Empty input")`; HashingVectorizer
  returns a `(0, n_features)` array.
- Cache path: `get_embeddings_path(hexdigest)` → `user_cache_dir("autointent")/embeddings/<hex>.npy`.
  Only the three backends import it; `utils.py` contains nothing else.
- **`use_cache` defaults to `True`** (`configs/_embedder.py:33`). It is **not** generally off:
  `tests/callback/test_callback.py` and `tests/assets/configs/full_training.yaml` use HV/embedders
  with caching on, and `tests/embedder/test_caching.py` flips it on.
- **Test gap:** several suites run with `use_cache=True` but **no fixture redirects the cache
  directory**, so they write to the **real OS cache dir**. The new config seam + a global isolation
  fixture will fix this for the whole test tree (§6.2).

## 4. Design

### 4.1 Cache directory resolution — `autointent/_cache_dir.py`

```python
def get_cache_dir() -> Path:
    """Base directory for autointent on-disk caches.

    Honors the AUTOINTENT_CACHE_DIR environment variable; otherwise falls back to
    appdirs.user_cache_dir("autointent"). Resolved fresh on each call so tests and
    parallel workers can point it at an isolated directory via the env var.

    NOTE: currently consumed only by the embedding cache. The structured-output
    cache still uses user_cache_dir("autointent") directly and is unaffected by
    this variable (documented limitation; see CHANGELOG).
    """
    override = os.environ.get("AUTOINTENT_CACHE_DIR")
    return Path(override) if override else Path(user_cache_dir("autointent"))
```

- `AUTOINTENT_CACHE_DIR` matches the existing `AUTOINTENT_`-prefixed env convention
  (`AUTOINTENT_PATH`, `AUTOINTENT_EXTRA_VALIDATION`, server `env_prefix="AUTOINTENT_"`).
- The embedding DB lives at `get_cache_dir() / "embeddings.db"` (replacing the `embeddings/` dir of
  `.npy` files). WAL adds `embeddings.db-wal` and `embeddings.db-shm` sidecars — still ~3 files vs.
  K inodes.

### 4.2 Per-utterance key — in `autointent/_wrappers/embedder/_sqlite_cache.py`

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
behavior). The key is the **original** utterance text — not the prompt-prepended form — so a backend's
internal prompt application stays an implementation detail of `_embed_uncached`.

**Collision discussion.** The key is a 64-bit xxhash hexdigest, the same strength as today's
whole-list key, so this is not a regression in hash kind. Per-utterance keying produces more distinct
keys than per-list keying, so the absolute collision probability rises, but remains negligible
(~5e-10 at 1e5 utterances). Two cases:
- **Cross-model collision** (two different `model_hash` values producing the same key string): defended
  by storing `model_hash` and filtering reads with `AND model_hash = ?` (§4.3). A cross-model collision
  becomes a cache **miss**, never a wrong vector. Because the primary key is `key` alone and writes use
  `INSERT OR IGNORE`, the second model can never store its colliding key (the first model's row wins), so
  for that one key the second model takes a **permanent** miss + recompute. This is documented and
  accepted (probability ~5e-10); a composite `(key, model_hash)` PK is a possible future refinement.
- **Same-model collision** (same model+prompt, different utterance, same 64-bit digest): would return a
  wrong vector, exactly as today's scheme could; accepted as astronomically rare. Not mitigated further
  in this PR (widening to 128-bit is a non-goal).

**Backward compatibility:** this is a brand-new keying scheme and a brand-new store. All existing
`.npy` caches are invalid and ignored (the approved fresh start). First run after upgrade recomputes;
subsequent runs hit the new cache.

### 4.3 SQLite store — `SQLiteEmbeddingCache`

**Schema (version 1):**

```sql
CREATE TABLE IF NOT EXISTS embeddings (
    key           TEXT    PRIMARY KEY,   -- utterance_key() hexdigest
    model_hash    TEXT    NOT NULL,      -- str(get_hash()); cross-model filter + per-model purge
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

- `model_hash` is stored as **TEXT** because `Hasher.intdigest()` is an unsigned 64-bit value that can
  exceed SQLite's signed-64-bit `INTEGER` range. The public methods take `model_hash: int` but **bind
  `str(model_hash)` on every path** (the INSERT values *and* the `WHERE model_hash = ?` filter); binding a
  raw int > 2**63-1 would raise `OverflowError`, so the `str()` is mandatory, not cosmetic.
- **HARD INVARIANT — vector serialization.** Always store
  `np.ascontiguousarray(vec, dtype=np.float32).tobytes()`. The `dtype=np.float32` coercion is
  load-bearing: if a float64 vector were stored, the blob would be `8*dim` bytes and every read would
  fail the `len(blob)//4 == dim` check forever (permanent miss + repeated wasted writes). Reconstruct
  with `cast("npt.NDArray[np.float32]", np.frombuffer(blob, dtype=np.float32))` and validate length
  against `dim`. `np.frombuffer` returns a **read-only** array; this is safe only because
  `_embed_cached` always `np.stack`s (copies) before `_to_tensor` — do **not** add a single-utterance
  fast path that hands a frombuffer view to torch.

**Connection & pragmas.** `_connect()` opens a connection with `isolation_level=None` (autocommit; we
issue explicit `BEGIN IMMEDIATE` / `COMMIT` for the write paths) and applies, **per connection in
autocommit mode** (never inside an open transaction):
- `PRAGMA busy_timeout=30000` (30 s) — writers wait instead of raising "database is locked". Generous to
  absorb many parallel trials flushing a fold's rows at once. Module constant `BUSY_TIMEOUT_MS`
  (could become configurable later; not in this PR).
- `PRAGMA synchronous=NORMAL` — safe with WAL, faster than FULL; on power loss you may lose the last
  transaction but the DB does not corrupt — acceptable for a cache.

`PRAGMA journal_mode=WAL` is set **once at schema init**, in autocommit before the `BEGIN IMMEDIATE`
(setting WAL inside a transaction fails). It persists in the DB file, so later connections inherit it.
If the underlying filesystem does not support WAL the PRAGMA returns the actual mode without raising; we
log a debug line and the cache still works, only with weaker concurrency.

**Single-host assumption.** WAL's shared-memory index (`-shm`) means multi-process safety holds **only
for processes on the same host.** Pointing `AUTOINTENT_CACHE_DIR` at a network filesystem (NFS/SMB)
shared across nodes is unsupported and may corrupt. This is documented (helper docstring + CHANGELOG);
no NFS detection/fallback is implemented (out of scope).

**Schema init + versioning (cross-process safe).** `PRAGMA user_version` holds `SCHEMA_VERSION` (1).
Schema-ensure runs **once per cache instance** (guarded by an instance flag + lock); after the WAL
pragma (autocommit), the version-check-and-create steps run inside a single `BEGIN IMMEDIATE` write
transaction to be atomic against other processes:
1. ensure WAL (autocommit, see above);
2. open `BEGIN IMMEDIATE` (acquire the write lock);
3. re-read `user_version` **after** acquiring the lock;
4. if `user_version == SCHEMA_VERSION`, do nothing (another process already initialized at this version);
   otherwise `DROP TABLE IF EXISTS embeddings`, `CREATE TABLE` + indexes, and `PRAGMA user_version =
   SCHEMA_VERSION`. (A fresh DB starts at `user_version == 0`, so it takes this branch and is created;
   there is no separate "table absent" case to special-case.)
5. `COMMIT`.
Re-reading under the write lock closes the two-process race where both see a stale version and double-drop
(the second would otherwise destroy the first's fresh rows). A version bump = automatic, safe fresh start.
A rolling upgrade where two processes run different `SCHEMA_VERSION` values causes repeated rebuilds /
cache misses (never corruption); acceptable and noted.

**Connection model.** Every public method opens a **short-lived connection** via `_connect()` and closes
it on exit. No connection is shared across threads, so the cache is inherently thread-safe (the stdlib
`sqlite3` `check_same_thread` guard is never tripped); WAL handles inter-process safety. `embed()` is
coarse-grained (one `get_many` + one `set_many` per call), so per-call connection overhead is negligible
next to model inference.

**Instance lifecycle.** A module-level `get_embedding_cache() -> SQLiteEmbeddingCache` resolves the DB
path from `get_cache_dir()` and returns an instance **memoized by resolved path** (module dict + lock).
Schema init runs once per path per process. Tests that set `AUTOINTENT_CACHE_DIR` to a fresh `tmp_path`
naturally get a distinct, isolated instance — no global reset needed.

**Public API:**

```python
class SQLiteEmbeddingCache:
    def __init__(self, db_path: Path) -> None: ...
        # stores path; ensures parent dir + schema lazily on first connect (idempotent, locked)

    def get_many(self, model_hash: int, keys: list[str]) -> dict[str, npt.NDArray[np.float32]]:
        # SELECT key, vector, dim WHERE model_hash = ? AND key IN (...), chunked to stay under
        # SQLITE_MAX_VARIABLE_NUMBER (chunk size 900; the placeholder string is built with `?`
        # only, annotated `# noqa: S608`). Returns only found+valid keys, each reconstructed to a
        # (dim,) float32 array. A row whose blob length disagrees with `dim` is skipped (logged),
        # treated as a miss. Read-only: does NOT update last_accessed (avoids read amplification;
        # see note). On sqlite3.Error / unreadable DB: log warning, return {} (recompute).

    def set_many(self, model_hash: int, entries: dict[str, npt.NDArray[np.float32]]) -> None:
        # INSERT OR IGNORE within a single transaction (executemany).
        # OR IGNORE => two workers computing the same key never conflict, and an existing entry is
        # never overwritten (entries are deterministic). created_at = last_accessed = time.time();
        # size_bytes = len(blob). On sqlite3.Error: log warning, return (uncached, never raises).
```

**Graceful degradation (control flow).** The try/except in `get_many`/`set_many` wraps
**connection-open + parent-dir creation + lazy schema-ensure + statement execution end-to-end**, not just
the SQL, so a corrupt header / permission error / "path is a directory" degrades to no-op rather than
raising into `embed()`. Caught exceptions at this outer level: **`(sqlite3.Error, OSError)`** —
`sqlite3.Error` for locking/corruption, and `OSError` (incl. `PermissionError`, `NotADirectoryError`) for
the `mkdir`/file-open path. Per-row blob reconstruction is additionally guarded with
`except Exception:  # noqa: BLE001` (a malformed blob / `dim` mismatch raises `ValueError`, not
`sqlite3.Error`), skipping just that row. With `str(model_hash)` binding (above), no `OverflowError` path
exists. `embed()` never observes a cache failure as anything but a miss.

**`last_accessed` note:** populated at insert but **not** updated on read. Updating it per read would
turn every cache hit into a write, defeating the WAL concurrency benefit. The column exists so a future
eviction PR can choose its own access-tracking policy; for now it equals `created_at`. Deliberate
groundwork, documented as such.

### 4.4 Backend refactor — template method in `BaseEmbeddingBackend`

Lift the whole cache+dedup+reassemble flow into the base class once; backends implement only the pure
model call.

**ABC change (mypy-blocking, must do BOTH halves):**

1. Declare `config` on the ABC so the base's concrete methods can type-check `self.config.use_cache` /
   `self.config.get_prompt(...)` (the only fields the base touches, both on `BaseEmbedderConfig`):

   ```python
   class BaseEmbeddingBackend(ABC):
       config: EmbedderConfig          # union; narrowed in each subclass (see #2)
       supports_training: bool = False
       supports_cache: bool = True     # HV overrides to False (see below)
   ```

2. **Re-declare `config` with the specific type in EVERY concrete subclass.** A base annotation of the
   union type *overrides* mypy's previously-narrow per-`__init__` inference, which would break
   `self.config.tokenizer_config`/`device` (ST), `model_name`/`dimensions` (OpenAI), `max_model_len`
   (vLLM), `n_features`/`ngram_range`/… (HV), and `model_name` (fake) — empirically reproduced under the
   repo's mypy config. Each subclass body must add a covariant narrowing re-declaration:

   ```python
   class SentenceTransformerEmbeddingBackend(BaseEmbeddingBackend):
       config: SentenceTransformerEmbeddingConfig   # narrows the base union
   ```

   …and likewise `OpenaiEmbeddingConfig`, `VllmEmbeddingConfig`, `HashingVectorizerEmbeddingConfig`, and
   (in the fake) `OpenaiEmbeddingConfig`. mypy permits a subclass to narrow an attribute to a subtype, so
   this restores today's narrow access while satisfying the base's `self.config` reference. **All five
   files must do this** (it is in the §7 list).

**`supports_cache` flag.** HashingVectorizer's default `n_features = 2**18` makes each per-utterance
vector ≈ 1 MB as a float32 BLOB — far outside the "small vector" premise that justifies BLOB storage, and
HV is a fast stateless backend where recompute is cheap and caching provides ~no value. HV therefore sets
`supports_cache = False`, so the template routes it straight to `_embed_uncached` regardless of
`use_cache`. **This exactly preserves today's behavior** (HV has no cache block today and is never
cached, even when `use_cache=True` as in `tests/callback` / `full_training.yaml`). Real embedding models
(ST ≈ 0.3–1 k dims, OpenAI/fake ≈ 1.5 k dims) keep `supports_cache = True`.

**Template `embed` (concrete; overloads preserved, `@abstractmethod` removed):**

```python
def embed(self, utterances, task_type=None, return_tensors=False):
    prompt = self.config.get_prompt(task_type)
    # Empty input, cache disabled, or a backend that opts out of caching (HV) bypasses the cache and
    # goes straight to the backend, preserving each backend's existing empty-input behavior
    # (ST/OpenAI/vLLM raise; HV returns a (0, dim) array). np.stack is therefore only ever called on a
    # non-empty key list.
    if not utterances or not self.config.use_cache or not self.supports_cache:
        arr = self._embed_uncached(utterances, prompt)
    else:
        arr = self._embed_cached(utterances, prompt)
    return self._to_tensor(arr) if return_tensors else arr

def _embed_cached(self, utterances, prompt) -> npt.NDArray[np.float32]:
    cache = get_embedding_cache()
    model_hash = self.get_hash()
    keys = [utterance_key(model_hash, u, prompt) for u in utterances]
    unique_keys = list(dict.fromkeys(keys))                 # de-dup, preserve order
    cached = cache.get_many(model_hash, unique_keys)
    missing = [k for k in unique_keys if k not in cached]
    if missing:
        key_to_utt: dict[str, str] = {}
        for u, k in zip(utterances, keys):
            if k in cached or k in key_to_utt:
                continue
            key_to_utt[k] = u
        missing_utts = [key_to_utt[k] for k in missing]
        computed = self._embed_uncached(missing_utts, prompt)   # (M, dim) float32
        new_entries = {k: computed[i] for i, k in enumerate(missing)}
        cache.set_many(model_hash, new_entries)
        cached.update(new_entries)                          # update regardless of write success
    return np.stack([cached[k] for k in keys])              # (N, dim), original order

@abstractmethod
def _embed_uncached(self, utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:
    """Compute embeddings WITHOUT caching. Returns a (N, dim) float32 array, except for empty
    input where each backend keeps its current behavior. The backend applies `prompt` in its own
    way (ST: pass to encode; OpenAI/vLLM: prepend; HV: ignore)."""

def _to_tensor(self, arr: npt.NDArray[np.float32]) -> "torch.Tensor":
    import torch
    return torch.from_numpy(arr)        # ST overrides to move to its device
```

- `embed()` becomes **concrete** with one implementation; the two `@overload` stubs are kept on the ABC
  (with `@abstractmethod` removed from both stubs and impl) so direct backend-level callers keep the
  `Literal[True] -> torch.Tensor` narrowing (e.g. `tests/embedder/test_openai_real_backend.py`).
  `get_hash`, `similarity`, `clear_ram`, `dump`, `load` stay abstract; `_embed_uncached` is new abstract.
  **Every backend (ST, OpenAI, vLLM, HV) and the fake drops its own `embed` method AND its `@overload`
  stubs** (a bare overload with no implementation is a mypy error) to inherit the ABC's concrete `embed`
  + overloads.
- The dedup/order algorithm is collision-free for the reassembly: every `missing` key is in
  `unique_keys ⊆ keys`, so it is reached in the zip and mapped exactly once (first occurrence wins);
  `missing`, `missing_utts`, `computed[i]` share one index space; empty `missing` is guarded.
- Backends collapse their `embed` to `_embed_uncached` returning **float32 numpy**:
  - **ST:** keep the `if self.config.tokenizer_config.max_length is not None:` guard before setting
    `model.max_seq_length`; `model.encode(..., convert_to_numpy=True, normalize_embeddings=True,
    prompt=prompt)`; cast float32. Override `_to_tensor` to `torch.from_numpy(arr).to(self.config.device
    or "cpu")` (preserves the current cache-hit device behavior). Keep its `ValueError` on empty input.
  - **OpenAI:** keep `ValueError` on empty; prepend prompt if present; run sync/async path; return float32.
  - **vLLM:** keep `ValueError` on empty; prepend prompt if present; `model.encode`; stack float32.
  - **HashingVectorizer:** ignore prompt (as today, so `_embed_uncached`'s `prompt` param is unused →
    `# noqa: ARG002`); transform → dense float32. **Empty input returns a `(0, n_features)` array** via an
    explicit guard — note sklearn's `HashingVectorizer.transform([])` actually raises `StopIteration`
    (sklearn ≥1.5), which the old `embed` propagated; the guard makes empty input graceful (this is the
    one small, deliberate behavior improvement, pinned by a regression test in §6.3). Sets
    `supports_cache = False` so it is never cached (avoiding ~1 MB BLOBs).
- **`FakeOpenaiEmbeddingBackend`** is migrated to implement `_embed_uncached(utterances, prompt)` and
  **inherit** the template `embed`. It also re-declares `config: OpenaiEmbeddingConfig` (the narrowing from
  the ABC change). Its body uses the **passed `prompt`** directly (it must NOT call `get_prompt(task_type)`
  again, and must NOT prepend the prompt — it keeps prompt-as-seed:
  `seed_extra = f"{model_name}|{prompt or ''}"`), and moves the lazy `self._client` touch into
  `_embed_uncached`. This keeps `test_client_lazy_loading`, `test_prompts_application`, and
  `test_return_tensors_functionality` green (all use `use_cache=False`) while giving the fake genuine cache
  coverage when caching is on (hermetic via §6.2).

**`base.py` imports:** the concrete methods need runtime `numpy` (`np.stack`) and
`from ._sqlite_cache import get_embedding_cache, utterance_key`; `_to_tensor` imports `torch` lazily.
`numpy` therefore moves out of the `TYPE_CHECKING` block (ruff `TC` will require this). No import cycle:
`base → _sqlite_cache → {_hash, _cache_dir}` does not point back at `base`.

**Tensor/device semantics:** the cache always stores/reconstructs CPU float32. When `return_tensors=True`,
the base converts via `_to_tensor`. This is equivalent to the current cache-hit behavior. The only nuance
is an ST **cache-miss** with `return_tensors=True`: previously the raw on-device encode tensor was
returned; now it round-trips through CPU numpy and back to the device. sentence-transformers returns
float32 for both `convert_to_*` modes and normalizes identically, and all CI configs use `device="cpu"`,
so values are byte-identical and `.to("cpu")` is a no-op — a documented, test-invisible unification.

### 4.5 Data flow (one `embed(["a","b","a","c"])` call, partial hit, cache on)

1. Resolve `prompt` from `task_type`.
2. Non-empty + `use_cache=True` → `_embed_cached`.
3. `model_hash = get_hash()`; `keys = [k_a, k_b, k_a, k_c]`; `unique = [k_a, k_b, k_c]`.
4. `get_many(model_hash, [k_a,k_b,k_c])` → say `{k_a: v_a}` (a was cached). `missing = [k_b, k_c]`.
5. `key_to_utt = {k_b:"b", k_c:"c"}`; `_embed_uncached(["b","c"], prompt)` → `[v_b, v_c]`.
   `set_many(model_hash, {k_b:v_b, k_c:v_c})`; `cached = {k_a:v_a, k_b:v_b, k_c:v_c}`.
6. `np.stack([v_a, v_b, v_a, v_c])` → (4, dim) in input order. Convert to tensor if requested; return.

## 5. Error handling and robustness

- **Cache read failure** (locked beyond busy_timeout, corruption, unreadable file): `get_many` logs a
  warning and returns `{}` → everything recomputed. `embed()` still succeeds.
- **Cache write failure**: `set_many` logs a warning and returns; `cached.update(new_entries)` already
  ran, so the returned matrix is correct (just uncached).
- **Connect-time failure** (corrupt header, permission denied, path is a directory): caught because the
  guard wraps `_connect()` + schema-ensure end-to-end; degrades to no-op for the process.
- **Schema version mismatch**: atomic drop+recreate under `BEGIN IMMEDIATE` with a post-lock re-read
  (cross-process safe); a cache rebuild, not a crash.
- **Dimension/blob mismatch on read**: that row is skipped (logged) and treated as a miss; recomputed.
- **`_embed_uncached` raising** (real model/API error) propagates normally — that is not a cache failure.
- **Concurrency**: WAL + `busy_timeout` + `INSERT OR IGNORE` + single-transaction writes make concurrent
  multi-process trials and multi-thread access safe **on one host** without external locking.
- **`get_hash()` cost**: called once per `embed` (cached or not) — the same frequency as today's inline
  cache block, so no regression. (For local-path ST models `get_hash` hashes all parameters on every
  call; memoizing it is a possible future optimization, §9, not in scope.)

## 6. Testing strategy

All tests are verified **via CI on the draft PR** (maintainer rule: no heavy/exhaustive pytest locally;
ruff + mypy run locally). New tests are fast and do **not** download models (pure-Python unit tests plus
the pinned tiny ST model already used in CI).

### 6.1 New unit tests — `tests/embedder/test_sqlite_cache.py` (pure Python, no ML)
- `set_many` then `get_many` round-trips exact float32 bytes; reconstructed shape `(dim,)`.
- Miss returns absent keys; partial hit returns only present keys.
- `get_many` filters by `model_hash`: a key stored under model A is **not** returned for model B.
- `INSERT OR IGNORE`: re-inserting an existing key does not overwrite or error.
- Chunking: `get_many` with > 900 keys returns all matches (exercises the IN-chunk loop).
- Schema: WAL enabled (where supported); `user_version == SCHEMA_VERSION`; columns/indexes present;
  a DB pre-set to a different `user_version` triggers a rebuild (table dropped+recreated).
- Graceful degradation: a corrupted/garbage DB file → `get_many` returns `{}`, `set_many` no-ops,
  no exception; a `dim`/blob-length mismatch row is skipped, not raised.
- `get_cache_dir()`: honors `AUTOINTENT_CACHE_DIR`; falls back to appdirs when unset (the "unset" case
  must `monkeypatch.delenv("AUTOINTENT_CACHE_DIR", raising=False)` because the global isolation fixture
  in §6.2 sets it for every test).
- `utterance_key()`: stable; differs by utterance, by prompt, by model_hash; equal for equal inputs.

### 6.2 Test isolation — **global** fixture in `tests/conftest.py`
Because `use_cache` defaults to **True**, any test that builds a default-config embedder (not just
`tests/embedder/`) can write the embedding DB to the real OS cache dir. Add a **function-scoped autouse**
fixture in the top-level `tests/conftest.py`:

```python
@pytest.fixture(autouse=True)
def _isolate_embedding_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOINTENT_CACHE_DIR", str(tmp_path / "ai_cache"))
```

- Each test gets its own cache dir (unique `tmp_path`), so there is no cross-test bleed and no
  real-OS-cache pollution anywhere in the suite (fixes today's gap, incl. `tests/callback` and
  `full_training.yaml` runs).
- It only sets an env var; the structured-output cache tests (which monkeypatch
  `autointent.generation._cache.user_cache_dir` directly and never read `AUTOINTENT_CACHE_DIR`) are
  unaffected. Add a comment in the fixture noting the per-test isolation is load-bearing for the reuse test.

### 6.3 Updated integration tests — `tests/embedder/test_caching.py`
- Keep existing parametrized consistency tests (cache on/off identical results).
- **Per-utterance reuse (the headline win, with a real signal):** embed `["x","y"]`, then `["y","z"]`,
  on a cache-on ST (or fake) backend; assert the **SQLite DB contains exactly 3 rows** afterward (not 4).
  This is a black-box assertion that is true **only** with per-utterance keying (the old whole-list
  scheme would store 2 list-blobs, not 3 utterance rows), so it is a meaningful red→green signal that
  does not depend on the new private method name. Optionally also wrap the backend's underlying
  encode and assert the second call computes only `["z"]`.
- **Dedup-within-list:** embed `["x","x"]`; underlying encode receives a single `x`; output rows 0 and 1
  are byte-identical and in order.
- **Order-preservation:** a multi-element list returns rows in input order after a partial hit.
- Keep `test_cache_with_different_prompts` (different prompt ⇒ different key ⇒ different vector).
- **Empty-input behavior preserved (regression guard):** HV `embed([])` still returns a `(0, dim)`
  array; an ST/fake `embed([])` still raises `ValueError`. (Confirms the refactor did not change it.)

### 6.4 Regression / unchanged
- `tests/embedder/test_hash.py` (incl. offline #321 cases) is unaffected — `get_hash()` is unchanged.
- `tests/embedder/test_openai_backend.py` fake-contract tests (`test_client_lazy_loading`,
  `test_prompts_application`, `test_return_tensors_functionality`) stay green per §4.4.
- `mypy src/autointent tests` stays green (strict, py3.10): annotate the new module fully; add the
  `config: EmbedderConfig` ABC declaration; `cast` the `np.frombuffer` result; `sqlite3` is typed.
- ruff (`select = ["ALL"]`, strict): the new module satisfies the full ruleset like any other non-`utils`
  module — module/class/function docstrings (D1xx), `%`-style logging args (no f-strings in `logger.*`,
  G004), named constants instead of magic numbers (e.g. `_FLOAT32_NBYTES = 4` rather than `len(blob)//4`),
  `from __future__ import annotations`, `pathlib` for paths. The **non-obvious** noqas that are expected and
  deliberate: `# noqa: S608` (the chunked `IN (...)` placeholder string, built from `?` only),
  `# noqa: BLE001` (the per-row blob `except Exception` for graceful degradation), and `# noqa: ARG002`
  (HV's unused `prompt` parameter). Keep `get_many`/schema-init small enough to avoid `C901`/`PLR0912`
  (extract helpers if needed). `np.frombuffer` is wrapped in `cast("npt.NDArray[np.float32]", ...)`.
- Coverage: §6.1 covers the new module's branches **including every `except`/skip branch**, keeping the
  85% combined floor.

## 7. File-by-file change list

**New**
- `src/autointent/_cache_dir.py` — `get_cache_dir()`.
- `src/autointent/_wrappers/embedder/_sqlite_cache.py` — `SQLiteEmbeddingCache`, `utterance_key`,
  `get_embedding_cache`, `SCHEMA_VERSION`, `BUSY_TIMEOUT_MS`.
- `tests/embedder/test_sqlite_cache.py` — unit tests (6.1).

**Modified**
- `src/autointent/_wrappers/embedder/base.py` — add `config: EmbedderConfig` annotation + `supports_cache`
  class flag; concrete `embed` (+ kept overloads) / `_embed_cached` / `_to_tensor`; abstract
  `_embed_uncached`; move `numpy` to runtime import (with `from ._sqlite_cache import …`; `torch` stays
  lazy inside `_to_tensor`).
- `src/autointent/_wrappers/embedder/sentence_transformers.py` — re-declare
  `config: SentenceTransformerEmbeddingConfig`; `embed` → `_embed_uncached` (preserve max_length guard);
  override `_to_tensor`; drop the inline cache block and `get_embeddings_path` import.
- `src/autointent/_wrappers/embedder/openai.py` — re-declare `config: OpenaiEmbeddingConfig`;
  `embed` → `_embed_uncached`; drop cache block/import.
- `src/autointent/_wrappers/embedder/vllm.py` — re-declare `config: VllmEmbeddingConfig`;
  `embed` → `_embed_uncached`; drop cache block/import.
- `src/autointent/_wrappers/embedder/hashing_vectorizer.py` — re-declare
  `config: HashingVectorizerEmbeddingConfig`; set `supports_cache = False`; `embed` → `_embed_uncached`
  (keep empty → `(0, dim)`, `# noqa: ARG002` on unused `prompt`); remove its now-redundant `embed` overloads.
- `tests/_fixtures/fake_openai_embedding.py` — re-declare `config: OpenaiEmbeddingConfig`;
  `embed` → `_embed_uncached` (use passed prompt, no prepend, keep prompt-as-seed, move `_client` touch);
  inherit the template embed.
- `tests/conftest.py` — global autouse `AUTOINTENT_CACHE_DIR` → tmp_path isolation fixture (§6.2).
- `tests/embedder/test_caching.py` — reuse (row-count) / dedup / order / empty-input-preserved tests.
- `CHANGELOG.md` (repo root; latest section `[0.3.2]`) — add an Unreleased/next-version entry: new SQLite
  per-utterance embedding cache, `AUTOINTENT_CACHE_DIR` (embedding cache only), fresh-start invalidation
  of old `.npy` caches.

**Removed**
- `src/autointent/_wrappers/embedder/utils.py` `get_embeddings_path` (and the file, since it becomes empty;
  no other importer exists).

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Refactor changes per-backend embed values subtly | Backends keep their exact encode calls inside `_embed_uncached`; only caching/reassembly moves. Parametrized consistency tests guard cache-on == cache-off; values unchanged. |
| `use_cache` defaults to True → broader real caching than before (incl. HV) writing to real OS cache | Global autouse isolation fixture (§6.2) redirects the cache dir for every test; production behavior is intended (caching on by default, as today). |
| `self.config` undeclared on ABC → mypy strict failure | Declare `config: EmbedderConfig` on the ABC **and** re-declare `config: <SpecificConfig>` in all five subclasses (narrowing). |
| HV default caching → ~1 MB BLOBs / DB bloat | `supports_cache = False` on HV; it is never cached (preserves today's behavior). |
| Connect/`mkdir` errors or int model_hash bind escaping `embed()` | Outer catch is `(sqlite3.Error, OSError)`; `model_hash` bound as `str()`. |
| Fake backend inheriting template embed breaks fake-contract tests | `_embed_uncached` keeps prompt-as-seed and the lazy `_client` touch; §6.4 lists the exact tests guarded. |
| Cross-process schema-rebuild race | `BEGIN IMMEDIATE` + post-lock re-read of `user_version`; rebuild only on version bump. |
| WAL on a network filesystem corrupts | Documented single-host assumption; WAL pragma degrades silently on unsupported FS. |
| "database is locked" under heavy parallel writes | WAL + 30 s `busy_timeout` + `INSERT OR IGNORE` + one transaction per `set_many`; a timed-out write degrades to uncached, never wrong. |
| float64 leaking into the blob → permanent miss | HARD INVARIANT: `np.ascontiguousarray(vec, dtype=np.float32).tobytes()`; `dim` validated on read. |
| Cross-model 64-bit key collision → wrong vector | `get_many` filters `AND model_hash = ?`; collision becomes a recompute. Residual same-model collision accepted as today. |
| vLLM path can't run in CI (no GPU) | `_embed_uncached` for vLLM is a thin wrapper; shared logic tested via ST + fake + pure unit tests; vLLM test stays `skipif`. |
| Empty-input behavior change | Avoided: empty input bypasses the cache to `_embed_uncached`, preserving each backend's current behavior; regression-guarded in §6.3. |

## 9. Future work (not in this PR)
- Active eviction: size-cap LRU and/or TTL using the groundwork columns (and a read-time
  `last_accessed` update policy).
- Route the structured-output cache through `get_cache_dir()` and/or migrate it onto the same
  SQLite layer (then `AUTOINTENT_CACHE_DIR` would govern both).
- Memoize `get_hash()` on the backend instance (invalidated in `train()`) to cheapen cache hits for
  local-path ST models.
- Optional LMDB/Parquet/memmap sidecar for very large vector volumes if BLOB storage ever becomes
  a bottleneck.
- Optional 128-bit keys if same-model collision ever becomes a practical concern.
