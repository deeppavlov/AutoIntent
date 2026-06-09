# Compute Feasibility Advisor for AutoIntent

- **Date:** 2026-05-23
- **Status:** Proposal (pre-implementation)
- **Audience:** AutoIntent maintainers / contributor picking up the task
- **Scope of this document:** technical specification — *what* the advisor estimates and the formulas it uses. Architectural and system-design choices (where the advisor lives in the codebase, how it integrates with the optimizer, the public API surface, file/module layout) are deliberately left to the implementer.

## Problem

AutoIntent's main strength is letting a user kick off a full search-space optimization with one call:

```python
pipeline = Pipeline.from_preset("transformers-heavy")
pipeline.fit(dataset)
```

The cost of that convenience is that users — especially those running on a laptop, a single consumer GPU, or a free cloud instance — cannot tell ahead of time whether their hardware can carry the configuration they have just selected.

Concrete failure cases we see today:

- `transformers-heavy` fine-tunes `microsoft/deberta-v3-large` for up to 30 epochs across 40 HPO trials. That needs ~12–18 GB VRAM (full fine-tune, fp32) and many hours of wall time on a single GPU. A user with an 8 GB card finds out by OOM, often several minutes into a run.
- Swapping `intfloat/multilingual-e5-large-instruct` (2 GB) for `sentence-transformers/all-MiniLM-L6-v2` (90 MB) changes the resource bill by an order of magnitude — but nothing surfaces this difference up front.
- Disk is a silent failure mode: a search space referencing several large checkpoints can pull >10 GB into the HF cache before any training starts.

The target audience for this feature is users with limited resources who pick a preset, hit `fit()`, and want to know within a second whether they should change something.

## Proposed solution: pre-flight resource advisor

Add a **pre-flight advisor** that, given a parsed search space and a dataset, estimates worst-case disk, RAM, VRAM, and wall-time requirements from public Hugging Face Hub metadata and a small set of formulas, then prints a clear summary with red/yellow/green warnings. By default it is **report-only and never blocks the run**; an opt-in **reduce-to-fit** mode additionally prunes the search space to fit detected hardware.

### Scope

The advisor analyses only the **local, model-bearing** modules whose footprint can be derived from HF Hub metadata. Everything else is either trivial or out of band.


| Module category                                                                  | In scope? | Reason                                             |
| -------------------------------------------------------------------------------- | --------- | -------------------------------------------------- |
| `SentenceTransformerEmbeddingConfig`                                             | yes       | local transformer, dominant cost on small machines |
| `VllmEmbeddingConfig`                                                            | yes       | local transformer with extra engine overhead       |
| `HFModelConfig`-based scorers (`bert`, `lora`, `ptuning`, `dnnc`, cross-encoder) | yes       | the actual heavyweights                            |
| GCN scorer when configured with a transformer backbone                           | yes       | inherits the backbone cost                         |
| `LinearScorer` (sklearn `LogisticRegression` / `LogisticRegressionCV`)           | yes       | dominant cost on presets with no transformer fine-tune; the CV path multiplies a single fit by ~30 |
| `CatBoostScorer`                                                                 | yes       | dominant cost on presets with no transformer fine-tune; high default `iterations` |
| `OpenaiEmbeddingConfig`                                                          | no        | no local resources to estimate                     |
| `HashingVectorizerEmbeddingConfig`                                               | no        | trivial cost                                       |
| `knn`, `mlknn`, generic `sklearn` classifiers via `SklearnScorer`, `description` | no        | bounded so far below any in-scope module that they cannot plausibly be the bottleneck |
| `decision` and `regex` nodes                                                     | no        | negligible                                         |


Rationale: the user's real risk is whichever module is the actual bottleneck. On heavy presets that is a transformer fine-tune; on light presets it shifts to `linear` (CV-multiplied) or `catboost` (1000 default iterations × dataset shape). Modules left out of scope are ones whose cost is bounded so far below any in-scope module that they cannot plausibly be the reason a run fails.

### Phases

The advisor is one entry point, but internally splits work into three phases that share a single `PreflightReport` object. The split is internal organization — all three run at the same hook point (after `validate_modules`, before `_fit(context)`) and the user sees one summary. Separating them keeps each phase's inputs, formulas, and failure modes scoped:

- **Resource phase.** Disk / RAM / VRAM / wall-time estimates and comparisons against detected hardware. Most of the formulas in this document live here. This is the only phase consumed by the reduce-to-fit pruner.
- **Data quality phase.** Findings derived from the dataset jointly with the active search space — token-length truncation, split readiness (auto-invokes the existing `check_split_readiness` utility rather than re-implementing it), partial intent descriptions paired with the `description` scorer, embedder/scorer dimension consistency. Reports red/yellow lines but never prunes the search space; the user fixes the dataset or the config.
- **Configuration sanity phase.** Joint checks across dataset + search-space + hardware that don't slot cleanly into the other two — e.g., `hpo_config.n_jobs > 1` × per-trial VRAM contention, CatBoost `task_type="GPU"` with no CUDA. Pydantic schema validation already runs upstream on `OptimizationConfig`; this phase only adds checks that need joint inspection.

The advisor consumes `validate_modules`'s *post-filter* view of `self.nodes` — it does not duplicate that mutating filter.

### Inputs

- The parsed `OptimizationConfig` (search space, HPO config, embedder/transformer configs).
- The training `Dataset` (for `dataset_size` and an approximate token-length distribution).
- Detected local hardware:
  - Total / available RAM via `psutil`.
  - Free disk on the AutoIntent / HF cache directory via `shutil.disk_usage`.
  - Accelerator detection, in priority order:
    - **CUDA:** per-GPU VRAM and device name via `torch.cuda`.
    - **MPS (Apple Silicon):** detected via `torch.backends.mps.is_available()`. Apple chips use unified memory, so there is no separate VRAM pool — the "VRAM budget" is a fraction of total system RAM. Default budget = 70 % of total RAM (matching the macOS `PYTORCH_MPS_HIGH_WATERMARK_RATIO` default) with the remainder reserved for the OS and other apps. The fraction is exposed as a knob.
    - **CPU only:** when neither is available.

### Output

A structured estimate plus a human-readable summary printed to the logger. Example:

```
Compute feasibility check
─────────────────────────
Resource:
  Available : 8 GB VRAM (NVIDIA RTX 3060), 32 GB RAM, 120 GB free disk
  Disk    : 5.2 GB to download, 1.1 GB already cached  (3 unique checkpoints)
  RAM     : ~4 GB
  VRAM    : ~14 GB × 2 parallel trials (n_jobs=2) ⚠  exceeds available
  Time    : ~6 h  (+~12 min for refit_after)            (single-GPU, fp32, rough)

Data:
  Train tokens p95 : 612 (exceeds bert.max_length=512) ⚠  ~7% truncated
  Split readiness  : 2 classes have <3 samples — LogisticRegressionCV cv=3 will fail ✗

Config:
  CatBoost task_type=GPU but no CUDA detected — will fall back to CPU ⚠

Drivers of cost:
  scoring.bert   microsoft/deberta-v3-large   full fine-tune × 40 trials × 30 epochs  →  ~14 GB VRAM, ~5 h
  embedder       intfloat/multilingual-e5-large-instruct                              →  ~2.2 GB VRAM

Suggestions:
  • Enable mixed precision (fp16/bf16) on the bert scorer
  • Reduce batch_size from 64 to 16 or 32
  • Set hpo_config.n_jobs=1 — parallel trials are doubling VRAM demand
  • Try preset `transformers-light` or `classic-medium`

These numbers are heuristic upper bounds, not measurements.
```

Numbers are reported with honest precision (one significant figure for time, two for memory) and an explicit "estimate, not measurement" disclaimer.

### Algorithm (proposal, allowed to adjust)

1. **Collect candidates.** Walk the search space; collect every unique in-scope module. For transformer-bearing modules the identity is `(module_type, model_name, mode)` with `mode ∈ {inference, lora, full-finetune}`. For `linear` and `catboost` the identity is `(module_type, embedder_name, task_kind)` with `task_kind ∈ {multiclass, multilabel}` — the routing through `LogisticRegressionCV` vs `MultiOutputClassifier`, and CatBoost's per-class trees, both depend on it. Also collect the HPO knobs that drive cost: `n_trials` plus per-module knobs — transformer (`epochs`, `batch_size`, `max_length`, `dtype` ∈ {fp16, bf16, fp32}), `linear` (`cv`, `max_iter`), `catboost` (`iterations`, `depth`, `task_type`, `features_type`).
2. **Resolve checkpoints.** For each unique `model_name`, query HF Hub for safetensors metadata to read parameter count and weight dtype. Fall back to file-size aggregation if safetensors metadata is missing. Fall back to a "unknown — heuristic only" tag with low-confidence labelling if HF Hub is offline or the repo is private. `LinearScorer` and `CatBoostScorer` have no checkpoint of their own; they reuse the embedder resolved by this step in their formulas (their cost is parameterised by `embedder_dim`, not parameter count).
3. **Apply formulas.** All values are honest upper bounds; convergence and early stopping often terminate well below them.
  - **Disk** = sum over unique downloadable checkpoints of total file size, plus a small fixed overhead per checkpoint for tokenizers and config. `LinearScorer` and `CatBoostScorer` contribute zero (they consume embedder output that is already accounted for upstream).
  - **RAM per module:**
    - Transformer modules (any mode): `params × dtype_bytes + dataset_tokens × 4 bytes`, treated as a loose upper bound for tokenized buffers.
    - `LinearScorer`: `8 × n_samples × embedder_dim` (float64 data matrix — the dominant term) `+ 8 × n_classes × embedder_dim` (coefficients) `+ ~10 × 8 × embedder_dim` (L-BFGS history).
    - `CatBoostScorer`: `4 × n_samples × n_features` (data, float32 internally) `+ 4 × n_features × n_bins` (histograms; default `n_bins = 254`) `+ iterations × 2^depth × ~32 bytes` (tree storage). For `features_type ∈ {embedding, both}`, `n_features = embedder_dim`. For `features_type = text`, `n_features` is the BoW vocab discovered at fit; bound with a coarse default (e.g. 50 000) and tag the estimate low-confidence.
    - For `linear` and `catboost`, `embedder_dim` is taken from the largest embedder in the same node group — same worst-case stance as the rest of the estimate.
  - **VRAM per module:**
    - Inference embedder: `params × dtype_bytes × ~1.3` (small constant for activations).
    - Full fine-tune (`bert`, GCN backbone, soft-prompt `ptuning`): `params × dtype_bytes × (1 + 1 + 2)` for weights + grads + Adam state, halved when fp16/bf16 mixed precision is configured.
    - LoRA: inference VRAM + a small adapter constant.
    - Reranker (cross-encoder, `dnnc`): inference VRAM × small factor for the reranking pass.
    - `LinearScorer`: N/A (sklearn is CPU-only).
    - `CatBoostScorer`: 0 by default; if `task_type="GPU"` is configured, the RAM formula above lives on device instead.
  - **Time per module:**
    - Transformer modules: `n_trials × epochs × (dataset_size / batch_size) × per_step_seconds(params, max_length, device_class)`, where `per_step_seconds` is a small static lookup keyed on coarse device class (`cpu`, `low-gpu`, `mid-gpu`, `high-gpu`, `apple-silicon`) auto-detected from `torch.cuda.get_device_name` or `platform`/`torch.backends.mps`.
    - `LinearScorer`: `n_trials × C_cpu × n_samples × embedder_dim × max_iter × cv_multiplier × class_multiplier`, where:
      - `C_cpu ≈ 1e-8 s` per `(sample × feature × iteration)` on a single modern CPU core.
      - `cv_multiplier = Cs × cv + 1 ≈ 31` for the multiclass path (`LogisticRegressionCV` with default `Cs = 10`, repo default `cv = 3`, plus one final refit). `cv_multiplier = 1` for the multilabel path (no inner CV).
      - `class_multiplier = n_classes` for the multilabel path (`MultiOutputClassifier` fits one binary LogReg per class); `class_multiplier = 1` otherwise.
    - `CatBoostScorer`: `n_trials × iterations × C_device × n_samples × n_features × depth × class_multiplier`, where:
      - `C_device ≈ 1e-9 s` on CPU, ~5–20× faster on GPU. Resolve `C_device` via the same `device_class` lookup as the transformer time formula.
      - `class_multiplier = n_classes` for both the multiclass `MultiClass` loss (per-class trees per iteration) and the multilabel routing (one CatBoost per class).
      - Early stopping is not modelled; `iterations` is treated as the upper bound.
  - Total time = sum across modules. MPS time numbers are coarser than CUDA's (one tier for now); we accept that.
4. **Compare to detected hardware.** Per-dimension status is green / yellow / red against a configurable headroom (defaults: **red** if estimate > 100 % of available, **yellow** if > 70 %). On MPS, "VRAM" and "RAM" estimates draw from the same physical pool; we compare *the larger of the two* against the unified-memory budget rather than each independently.
5. **Render summary.** Log at INFO. If any dimension is red, emit at WARNING so it shows in non-logging contexts.

#### Resource-phase refinements

These adjust the formulas above for situations that look fine in single-trial isolation but blow up in practice:

- **Cold-vs-warm HF cache (Tier 1).** Before reporting disk, probe each unique `model_name` against the local HF cache via `huggingface_hub.try_to_load_from_cache` / `scan_cache_dir`, keyed off `HF_HOME`. Split the disk line into `to_download` vs `already_cached`. Treat a repo as cached only if the weight shard (`model.safetensors` or equivalent) is present — not just config/tokenizer files. Without this, a repeated run on the same machine alarms the user about gigabytes they already have.
- **Concurrent-trial × per-trial VRAM (Tier 1).** Multiply the per-trial VRAM estimate by `hpo_config.n_jobs` when `n_jobs > 1` and the active accelerator is GPU. Same for the `dump_modules=True` path on disk: each trial writes module weights to the dump dir, so multiply per-module dump-disk by `n_trials`. vLLM is process-isolated and its contention model differs; note this in the disclaimer.
- **`refit_after=True` time delta (Tier 2).** When `Pipeline.fit(refit_after=True)`, add one full-data training pass per node to the time estimate. Small term but easy to forget; users running close to their time budget care about it.
- **HF Hub reachability probe (Tier 2).** One up-front `HfApi().whoami()` (or unauthenticated `HEAD` to `huggingface.co`) at the start of the phase. On failure, consistently downgrade *all* model entries to the "unknown — heuristic only" path instead of timing out per-model 10× on a 10-model search space.
- **CatBoost `task_type="GPU"` sanity (Tier 2).** When CatBoost is in the search space with `task_type="GPU"` but `torch.cuda.is_available()` is false, tag yellow — CatBoost silently falls back to CPU and the user otherwise sees CPU speeds with no warning.

### Data quality phase

The resource phase predicts whether the run *fits*. The data quality phase predicts whether the run *produces a meaningful result*. Both are caught at the same hook point because both have the same failure mode from the user's perspective: hours of compute followed by a cryptic error or a silently degraded model.

- **Token-length truncation (Tier 1).** Sample ~1000 utterances from the train split, tokenize against each unique transformer's tokenizer, compute `p95_tokens` and `% truncated` against the module's `max_length`. Yellow when >1% truncated; red when >10%. Reuse the tokenizer the resource phase already loaded for parameter-count resolution — don't double-fetch. The existing pipeline silently truncates (sentence-transformers and the HF Trainer both default to `truncation=True`); there is no warning anywhere today.
- **Auto-invoke `check_split_readiness` (Tier 1).** Call the existing utility at `context/data_handler/_readiness_util.py:44–109` with the active `data_config` and surface its `SplitReadinessResult` — it already returns `underpopulated_classes`, `ready`, and a `reason` string, but is not called anywhere from `Pipeline.fit()` today. When `LinearScorer` with CV is in the search space and any class has `n < cv`, name the module by name in the red line ("`LogisticRegressionCV` cv=3 will fail: classes [X, Y] have <3 samples") rather than emitting a generic split-readiness message.
- **Partial intent descriptions × `description` scorer (Tier 1).** The dataset constructor already warns once at import when *some* but not all intents have descriptions (`_dataset/_dataset.py:199–207`). The advisor escalates this to red when the `description` scorer is also present in the active search space — otherwise the run will produce NaN embeddings for the missing intents. Action message: "fill in N missing descriptions", not "drop the scorer".
- **Embedder ↔ scorer dimension consistency (Tier 2).** For `LinearScorer` / `CatBoostScorer` with `features_type="both"`, verify the embedder reachable from the same node group exposes a stable, expected dimension. Cross-node walk; surface as yellow when the resolved dimension cannot be confirmed pre-flight.

### Configuration sanity phase

Pydantic schema validation on `OptimizationConfig` runs upstream at config-load time; this phase only adds checks that require *joint* inspection of dataset + search-space + hardware. With Tier 1 + Tier 2 in scope today, this phase holds two items:

- The `n_jobs × VRAM` callout, surfaced jointly with the resource phase (single line in the rendered output).
- The CatBoost `task_type="GPU"` without CUDA check, same.

Both could live entirely in the resource phase; they get their own phase because future additions — joint scorer↔decision shape checks, OOS-support mismatches detected up front rather than at module instantiation, embedder-dimension mismatches — slot here naturally. Keep the phase scaffold even if it is currently thin.

### Failure modes

- **HF Hub offline or private repo:** fall back to "unknown model — name-pattern heuristic only", explicit low-confidence label, never raise.
- **No accelerator (no CUDA and no MPS):** report VRAM as N/A and mark GPU-only modules as "requires GPU" without estimating a (misleading) CPU wall time.
- **MPS configured but a module is incompatible:** vLLM in particular does not run on MPS. Flag the module as "unsupported on MPS" rather than estimating; do not raise.
- **MPS with CPU fallback ops:** some PyTorch ops fall back to CPU on MPS, inflating system-RAM usage and wall time beyond the heuristic. Note this in the disclaimer; we don't try to model it.
- **vLLM configured but not installed:** still estimate (the VRAM accounting is similar), note that the engine itself has additional overhead not captured.
- **Estimate wildly wrong vs. reality:** always-on disclaimer in the printed summary that these are heuristic upper bounds.

### Reduce-to-fit mode

The feasibility check has two modes sharing the same estimation pipeline:

- **Report mode (default).** Print the summary, return the structured estimate, let the run proceed regardless of severity.
- **Reduce-to-fit mode (opt-in).** Additionally prune the search space to fit detected hardware before the run starts. Same estimates, same comparisons — just one extra step that produces a reduced search space.

Reduce-to-fit consumes only the **resource phase** output. Data-quality and config-sanity findings are reported but never trigger pruning — they require user action (fix the dataset, change a config flag), not search-space narrowing.

Using the same per-module estimates, the pruner applies three least-destructive steps in order:

1. **Filter discrete-choice hyperparameters.** For lists of cost-driving values (model name, batch size, training epochs, CatBoost `iterations` / `depth`, sklearn `cv`), keep only entries whose worst-case estimate fits.
2. **Cap continuous ranges.** For `{low, high}` ranges of cost-driving parameters, lower the upper bound to the largest fitting value. Ranges of non-cost parameters (learning rate, decision thresholds) are not touched.
3. **Drop module variants.** If a module entry has any required hyperparameter with no satisfiable value left, drop that module entry from its node's search space.

Guard rails:

- If pruning would leave any node's search space empty, the pruner **raises**. We don't silently produce a non-runnable pipeline, and we don't quietly fall back to report-only — failing loudly is the right contract for a mode whose whole purpose is to make the run feasible. The error message points the user toward a lighter preset.
- Time is not used as a filter — only memory and disk are. Time is still reported.
- Headroom thresholds are intentionally generous to avoid over-pruning and are configurable.

Alongside the standard estimate, the caller receives a structured description of what was filtered, capped, and dropped, plus the resulting search space and its recomputed (now green) estimate.

**Drawbacks worth surfacing.**

- **Silent narrowing of intent.** A search space deliberately written to include heavy/light variants for comparison gets halved. The mode is opt-in for this reason.
- **Over-pruning when our formulas overestimate.** A 30 %-high estimate on a borderline configuration throws away a run that would have succeeded. Generous headroom defaults mitigate; the knob is exposed.
- **Hard failure when nothing fits.** Raising is intentional — silent degradation to report-only would defeat the mode's purpose — but it is a sharper edge than report mode has.
- **Pre-trial only.** The rewrite happens before any HPO trial starts. This is fine because the search space is treated as immutable across a study, but worth calling out so nobody tries to make this dynamic later.

### CLI surface

The advisor is also exposed as a console script (`autointent-advisor`) so users can answer "what will this cost?" and "what should I run?" without writing Python. Two subcommands:

- **`autointent-advisor inspect <preset-name | path/to/config.yaml>`.** Resolves the preset (or a user-supplied `OptimizationConfig`), detects local hardware, runs the same three-phase advisor that `Pipeline.fit()` runs, and prints the same report. Accepts `--dataset` for a real dataset, or `--n-samples / --n-classes / --avg-tokens` placeholders when the dataset is not yet built — so the script is useful before any training data exists. `--json` emits the structured `PreflightReport` for scripting.
- **`autointent-advisor recommend [--n-samples ... | --dataset ...] [--budget-time 12h] [--budget-vram-gb 8]`.** Detects local hardware (with manual overrides applied), iterates over the bundled presets in `_presets/`, and tags each as `feasible` / `feasible-with-reduce` / `infeasible`. Ranks feasible presets by quality tier (`heavy > medium > light`) then estimated wall-time; picks the top one as the recommendation. For the heaviest infeasible preset, surfaces the single most-impactful knob change that would make it fit (e.g., "`transformers-heavy` would fit if `batch_size` ≤ 16 and `dtype=fp16`"), reusing the reduce-to-fit pruner's per-knob delta info.

**Constraints (both subcommands).** No model downloads — only HF Hub metadata endpoints (`HfApi().model_info`); never `from_pretrained`. Offline-safe — on Hub unreachability, fall back to the same "heuristic only" path and mark the report low-confidence; do not raise. Hardware-detection failures (broken CUDA install where `torch.cuda.mem_get_info()` raises) fall back to CPU detection and tag the report rather than crashing.

## Alternatives considered and rejected

### B. Smoke-test calibration

Run each unique module for one mini-batch / one step before the real fit, measure peak RAM and VRAM with `psutil`, `tracemalloc`, and `torch.cuda.max_memory_allocated`, time the step, and extrapolate to the full search space.

Rejected because:

- It **downloads weights just to estimate** — the disk-headroom check we wanted to provide is defeated by the act of performing it.
- It can **OOM while predicting OOM**, exactly on the constrained hardware that is the target audience.
- It adds **seconds to minutes** of wall time before `fit()` does anything, surprising users.
- It needs per-module "tiny run" hooks; not every scorer has a clean "stop after one step" path.
- For OpenAI- or vLLM-served embedders, a smoke test costs real money or starts the engine.
- Still not accurate due to CUDA and CPU cache, memory heating and so on.

### C. Curated benchmark table

Ship a JSON in the package with measured VRAM and per-step time for the bundled-preset checkpoints, broken out by hardware class (cpu / mid-gpu / high-gpu) and mode (inference / lora / full-finetune). Fall back to heuristics for unknown checkpoints.

Rejected because:

- **Maintenance burden:** every new model added to a preset would need entries across the hardware × precision × mode matrix.
- Numbers **go stale** when `transformers` updates change defaults (attention impl, dtype, gradient checkpointing).
- It still needs the chosen-solution heuristics as a long-tail fallback — so it adds work on top of Option A without replacing it.
- **Confident-but-wrong is worse than honest-but-fuzzy.** A table that says "4 GB on 4090" when the user OOMs at 4.5 GB damages trust more than a clearly-labelled range would.

### D. Layered (A by default, opt-in B, embedded table from C, local actuals cache)

Combine all three: ship A as the fast path, allow `calibrate=True` to trigger B for heavy modules only, embed a small table from C for the bundled-preset checkpoints, and write actuals from every real run to a local cache that feeds back into future estimates.

Rejected because:

- **Implementation surface multiplies:** two estimation code paths to keep consistent, a cache schema with versioning and eviction, two failure modes to document.
- **Discoverability:** users may not learn about `calibrate=True` and the realized value compresses back to roughly Option A anyway.
- The team's bandwidth doesn't justify the marginal accuracy gain over A for the target audience.

## Comparison


| Dimension                        | A (chosen)                     | B (smoke-test)         | C (benchmark table)                | D (layered)                           |
| -------------------------------- | ------------------------------ | ---------------------- | ---------------------------------- | ------------------------------------- |
| Wall time at pre-flight          | < 1 s                          | seconds–minutes        | < 1 s                              | < 1 s default, s–min when calibrating |
| Accuracy on common checkpoints   | medium                         | high                   | high                               | high                                  |
| Accuracy on custom checkpoints   | medium                         | high                   | medium (fallback)                  | medium–high                           |
| Time-estimate quality            | low–medium                     | high                   | high                               | high                                  |
| Disk pre-download required       | no                             | yes                    | no                                 | only when calibrating                 |
| Risk of OOM during the check     | none                           | real                   | none                               | only when calibrating                 |
| Network usage                    | 1 cached call per unique model | none beyond normal fit | none                               | combination                           |
| Implementation effort            | small                          | large                  | medium + ongoing benchmark refresh | large + cache infra                   |
| Ongoing maintenance              | low (formulas only)            | low                    | high                               | high                                  |
| Friendly to offline / air-gapped | with fallback                  | yes                    | yes                                | partial                               |


The chosen solution accepts a real accuracy gap on time and a moderate accuracy gap on VRAM in exchange for the only profile that fits the target audience's constraints: zero added wall time, zero added downloads, zero added failure modes, and a small one-time implementation cost.

## Out of scope (possible follow-ups)

- Live resource observability during `fit()` (peak RAM / VRAM per trial, abort on overrun).
- A learned calibration cache from real runs to refine estimates over time.
- **Determinism / `cudnn.deterministic` check.** Belongs in seed-setting code (`set_seed` utility, `Pipeline.__init__`), not in a feasibility advisor — reproducibility is not a hardware-budget question.
- **OpenAI / Generator token-cost ($) estimation.** Real value, but pricing tables age badly, the `StructuredOutputCache` hit rate is unknowable upfront, and the API-paying audience overlaps poorly with this advisor's stated audience (resource-constrained local users). Push to a separate `cost_estimator` tool.
- **Predictive CO₂ / emissions.** `_callbacks/emissions_tracker.py` already does this retrospectively, accurately. A predictive version multiplies our (loose) time estimate by a regional kWh/CO₂ factor — two sources of imprecision compounded. The retrospective number is the trustworthy one.
- **vLLM startup compile time.** Minutes of overhead before any work, but vLLM is unsupported on MPS, isn't the dominant cost on CUDA once running, and modelling it needs a startup-time lookup table. Note once in the disclaimer; do not model.

