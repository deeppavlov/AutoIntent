# Compute Feasibility Advisor for AutoIntent

- **Date:** 2026-05-23
- **Status:** Proposal (pre-implementation)
- **Audience:** AutoIntent maintainers / contributor picking up the task

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
| `OpenaiEmbeddingConfig`                                                          | no        | no local resources to estimate                     |
| `HashingVectorizerEmbeddingConfig`                                               | no        | trivial cost                                       |
| `knn`, `mlknn`, `linear`, `sklearn`, `catboost`, `description`                   | no        | negligible next to a fine-tune                     |
| `decision` and `regex` nodes                                                     | no        | negligible                                         |


Rationale: the user's real risk is the heavy transformer-backed modules. A cheap module cannot be the reason a run fails for resource reasons; we don't owe an estimate for it.

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
Available : 8 GB VRAM (NVIDIA RTX 3060), 32 GB RAM, 120 GB free disk
Estimated worst-case requirements for this search space:
  Disk    : 5.2 GB     (3 unique checkpoints)
  RAM     : ~4 GB
  VRAM    : ~14 GB     ⚠  exceeds available
  Time    : ~6 h       (single-GPU, fp32, rough)

Drivers of cost:
  scoring.bert   microsoft/deberta-v3-large   full fine-tune × 40 trials × 30 epochs  →  ~14 GB VRAM, ~5 h
  embedder       intfloat/multilingual-e5-large-instruct                              →  ~2.2 GB VRAM

Suggestions:
  • Enable mixed precision (fp16/bf16) on the bert scorer
  • Reduce batch_size from 64 to 16 or 32
  • Try preset `transformers-light` or `classic-medium`

These numbers are heuristic upper bounds, not measurements.
```

Numbers are reported with honest precision (one significant figure for time, two for memory) and an explicit "estimate, not measurement" disclaimer.

### Algorithm (proposal, allowed to adjust)

1. **Collect candidates.** Walk the search space; collect every unique `(module_type, model_name, mode)` triple, where `mode ∈ {inference, lora, full-finetune}`. Also collect HPO knobs that drive cost: `n_trials`, `epochs`, `batch_size`, `max_length`, `dtype` (fp16/bf16/fp32).
2. **Resolve checkpoints.** For each unique `model_name`, query HF Hub for safetensors metadata to read parameter count and weight dtype. Fall back to file-size aggregation if safetensors metadata is missing. Fall back to a "unknown — heuristic only" tag with low-confidence labelling if HF Hub is offline or the repo is private.
3. **Apply formulas.**
  - **Disk** = sum over unique checkpoints of total file size, plus a small fixed overhead per checkpoint for tokenizers and config.
  - **RAM** = max over modules of `params × dtype_bytes + dataset_tokens × 4 bytes`, treated as a loose upper bound for tokenized buffers.
  - **VRAM per module:**
    - Inference embedder: `params × dtype_bytes × ~1.3` (small constant for activations).
    - Full fine-tune (`bert`, GCN backbone, soft-prompt `ptuning`): `params × dtype_bytes × (1 + 1 + 2)` for weights + grads + Adam state, halved when fp16/bf16 mixed precision is configured.
    - LoRA: inference VRAM + a small adapter constant.
    - Reranker (cross-encoder, `dnnc`): inference VRAM × small factor for the reranking pass.
  - **Time per module** = `n_trials × epochs × (dataset_size / batch_size) × per_step_seconds(params, max_length, device_class)`, where `per_step_seconds` is a small static lookup table keyed on coarse device class (`cpu`, `low-gpu`, `mid-gpu`, `high-gpu`, `apple-silicon`) auto-detected from `torch.cuda.get_device_name` or `platform`/`torch.backends.mps`. Total time = sum across modules. MPS time numbers are coarser than CUDA's (one tier for now); we accept that.
4. **Compare to detected hardware.** Per-dimension status is green / yellow / red against a configurable headroom (defaults: **red** if estimate > 100 % of available, **yellow** if > 70 %). On MPS, "VRAM" and "RAM" estimates draw from the same physical pool; we compare *the larger of the two* against the unified-memory budget rather than each independently.
5. **Render summary.** Log at INFO. If any dimension is red, emit at WARNING so it shows in non-logging contexts.

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

Using the same per-module estimates, the pruner applies three least-destructive steps in order:

1. **Filter discrete-choice hyperparameters.** For lists of cost-driving values (model name, batch size, training epochs), keep only entries whose worst-case estimate fits.
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

