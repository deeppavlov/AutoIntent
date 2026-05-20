# %% [markdown]
"""
# Embedder Configuration

This tutorial covers comprehensive embedder configuration for text classification modules in AutoIntent. Most scoring modules use embedders to convert text into vector representations, which are crucial for model performance.

## Overview

AutoIntent supports several **embedding backends**, selected by the config type you pass (or by heuristics when you pass a plain `dict`—see `initialize_embedder_config` in the API reference):

- **Sentence Transformers** (default): Hugging Face models via `sentence-transformers`, with automatic device selection (CUDA, MPS, CPU).
- **OpenAI**: hosted embedding models via the OpenAI API.
- **vLLM**: local GPU inference for compatible Hugging Face embedding models.
- **HashingVectorizer**: fast, dependency-light vectors from scikit-learn (useful for tests and baselines).

Optional dependencies are grouped as pip extras (see `pyproject.toml`). For the default Sentence Transformers path, install:

```bash
pip install "autointent[sentence-transformers]"
```

Other backends need their own extras, for example `autointent[openai]` or `autointent[vllm]`, as shown in the sections below. When a backend package is missing, code paths that need it typically call `autointent._utils.require`, which raises an `ImportError` that includes the matching `pip install autointent[<extra>]` hint.

## Configuration Approaches

### Simple Configuration

The simplest way is to pass a model name as a string:
"""

# %%
from autointent.modules.scoring import KNNScorer, LinearScorer

# Using just the model name - sentence-transformers handles device detection
scorer = LinearScorer(embedder_config="sentence-transformers/all-MiniLM-L6-v2")

# %% [markdown]
"""
### Advanced Configuration

For more control, pass a dictionary with configuration parameters:
"""

# %%
from autointent.configs import get_default_embedder_config

# Using a dictionary for detailed configuration
advanced_embedder_config = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 64,  # Increase batch size for faster processing
    "device": "cuda:0",  # Override automatic detection if needed
    "tokenizer_config": {
        "max_length": 256,  # Set custom max sequence length
        "padding": True,
        "truncation": True,
    },
    "similarity_fn_name": "cosine",  # Choose similarity function
    "use_cache": True,  # Enable embedding caching
}

scorer = LinearScorer(embedder_config=advanced_embedder_config)

# %% [markdown]
"""
### Using EmbedderConfig Class

You can also use the `EmbedderConfig` class directly for type safety and IDE support:
"""

# %%
import torch

from autointent.configs import TokenizerConfig

embedder_config = get_default_embedder_config(
    model_name="sentence-transformers/all-mpnet-base-v2",
    batch_size=32,
    # Device is auto-detected, but you can override if needed
    device="cuda" if torch.cuda.is_available() else "cpu",
    tokenizer_config=TokenizerConfig(max_length=512, padding=True, truncation=True),
    classification_prompt="Classify the following text: ",  # Task-specific prompt
    similarity_fn_name="cosine",
    use_cache=True,
)

scorer = KNNScorer(embedder_config=embedder_config, k=10)

# %% [markdown]
"""
## Key Configuration Options

### Model Selection
- **`model_name`**: Any Sentence Transformers or Hugging Face model name
  - Popular choices: `"sentence-transformers/all-MiniLM-L6-v2"`, `"sentence-transformers/all-mpnet-base-v2"`
  - Language-specific: `"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`
  - Specialized models: `"sentence-transformers/all-distilroberta-v1"`, `"sentence-transformers/gtr-t5-base"`

### Infrastructure Settings
- **`device`**: Hardware device (`"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, etc.)
  - Usually auto-detected by sentence-transformers
  - Override only if you need specific device control
- **`batch_size`**: Number of texts to process simultaneously (higher = faster but more memory)
- **`bf16`/`fp16`**: Enable mixed precision for memory efficiency (requires compatible hardware)
- **`trust_remote_code`**: Whether to trust remote code when loading models (default: False)

### Tokenizer Settings
- **`tokenizer_config.max_length`**: Maximum sequence length (longer texts are truncated)
- **`tokenizer_config.padding`**: How to pad shorter sequences (`True`, `"longest"`, `"max_length"`, `"do_not_pad"`)
- **`tokenizer_config.truncation`**: Whether to truncate longer sequences (default: True)

### Task-Specific Prompts
Prompts can significantly improve embedding quality for specific tasks:

- **`classification_prompt`**: Prompt for classification tasks
- **`default_prompt`**: General-purpose prompt used when no task-specific prompt is available
- **`query_prompt`/`passage_prompt`**: For retrieval and search tasks
- **`cluster_prompt`**: For clustering tasks
- **`sts_prompt`**: For semantic textual similarity tasks

### Performance Settings
- **`use_cache`**: Cache embeddings to disk for repeated use (highly recommended)
- **`similarity_fn_name`**: Similarity function (default: `"cosine"`; other options like `"dot"`, `"euclidean"`, `"manhattan"` are available, but we recommend keeping the default unless you have a specific reason)

## Practical Examples

### Performance-Optimized Configuration
"""

# %%
# Example: Performance-optimized configuration
perf_config = get_default_embedder_config(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast, lightweight model
    batch_size=128,  # Large batch for speed
    # Device auto-detected by sentence-transformers
    tokenizer_config=TokenizerConfig(max_length=128),  # Shorter sequences for speed
    use_cache=True,  # Cache for repeated experiments
    fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
)

scorer = KNNScorer(embedder_config=perf_config, k=5)

# %% [markdown]
"""
### Quality-Optimized Configuration
"""

# %%
# Example: Quality-optimized configuration
quality_config = get_default_embedder_config(
    model_name="sentence-transformers/all-mpnet-base-v2",  # High-quality model
    batch_size=16,  # Smaller batch to handle longer sequences
    tokenizer_config=TokenizerConfig(max_length=512),  # Longer sequences for context
    classification_prompt="Classify the intent of this message: ",
    use_cache=True,
    similarity_fn_name="cosine",
)

scorer = LinearScorer(embedder_config=quality_config)

# %% [markdown]
"""
### Multilingual Configuration
"""

# %%
# Example: Multilingual setup
multilingual_config = get_default_embedder_config(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size=32,
    tokenizer_config=TokenizerConfig(max_length=256),
    use_cache=True,
)

scorer = KNNScorer(embedder_config=multilingual_config, k=7)

# %% [markdown]
"""
## OpenAI embeddings

Use `OpenaiEmbeddingConfig` when you want OpenAI-hosted models (install `pip install "autointent[openai]"`, which pulls in `openai` and `tiktoken`). Set `OPENAI_API_KEY` in your environment before calling `embed()`.

Important knobs:

- **`model_name`**: e.g. `"text-embedding-3-small"`.
- **`max_tokens_in_batch`**: caps each request by total tiktoken length of the batch (default `200_000`) so long texts do not hit OpenAI token limits; requests are also limited to at most **`batch_size`** strings.
- **`batch_size`**, **`max_concurrent`**, **`max_per_second`**: throughput and concurrency tuning.
"""

# %%
from autointent.configs import OpenaiEmbeddingConfig

openai_embedder_config = OpenaiEmbeddingConfig(
    model_name="text-embedding-3-small",
    batch_size=50,
    max_tokens_in_batch=200_000,
    use_cache=True,
)

# Pass the config object anywhere an embedder config is accepted, e.g.:
# LinearScorer(embedder_config=openai_embedder_config)

# %% [markdown]
"""
## vLLM embeddings

`VllmEmbeddingConfig` runs a compatible Hugging Face embedding model through **vLLM** on a GPU. Install with `pip install "autointent[vllm]"`. Typical options include `model_name`, `batch_size`, `gpu_memory_utilization`, `max_model_len`, and `dtype` (`"auto"`, `"float16"`, `"bfloat16"`, `"float32"`). See `VllmEmbeddingConfig` in `autointent.configs._embedder` for the full field list and defaults.
"""

# %%
from autointent.configs import VllmEmbeddingConfig

vllm_embedder_config = VllmEmbeddingConfig(
    model_name="BAAI/bge-base-en-v1.5",
    batch_size=32,
    dtype="auto",
    gpu_memory_utilization=0.9,
)

# %% [markdown]
"""
## HashingVectorizer (lightweight)

`HashingVectorizerEmbeddingConfig` maps text to a fixed-size sparse-ish hashed space via scikit-learn. It is **stateless**, has **no deep learning dependencies**, and is ideal for **fast tests** or CPU-only baselines. Use a smaller `n_features` (for example `512`) for quicker runs; the default is much larger for quality experiments.
"""

# %%
from autointent.configs import HashingVectorizerEmbeddingConfig

hashing_embedder_config = HashingVectorizerEmbeddingConfig(
    n_features=512,
    ngram_range=(1, 2),
)

# %% [markdown]
"""
## Fine-tuning embeddings

**Training is only implemented for the Sentence Transformers backend.** `Embedder.train(utterances, labels, config)` delegates to that backend and raises `NotImplementedError` for OpenAI, vLLM, and HashingVectorizer configs.

`EmbedderFineTuningConfig` (in `autointent.configs`) controls the training loop, including:

- **`epoch_num`**, **`batch_size`**, **`learning_rate`**, **`warmup_ratio`**
- **`margin`** (contrastive / retrieval-style objective hyperparameter used by the trainer)
- **`val_fraction`**, **`early_stopping_patience`**, **`early_stopping_threshold`**
- **`fp16`** and **`bf16`** for mixed-precision training (set at most one appropriately for your device)

The **`RetrievalAimedEmbedding`** module accepts an optional **`ft_config`**: when present, `fit()` calls `Embedder.train(...)` before building the vector index—convenient when retrieval quality is your optimization target.
"""

# %%
from autointent import Embedder
from autointent.configs import EmbedderFineTuningConfig, SentenceTransformerEmbeddingConfig

ft_cfg = EmbedderFineTuningConfig(
    epoch_num=2,
    batch_size=8,
    learning_rate=2e-5,
    val_fraction=0.2,
    fp16=False,
    bf16=False,
)

# Example (does not run training here): construct an embedder and call train when you have data.
_embedder_for_ft = Embedder(
    SentenceTransformerEmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
# _embedder_for_ft.train(utterances=[...], labels=[...], config=ft_cfg)

# %%
from autointent.modules.embedding import RetrievalAimedEmbedding

_retrieval_with_ft = RetrievalAimedEmbedding(
    k=5,
    embedder_config="sentence-transformers/all-MiniLM-L6-v2",
    ft_config=ft_cfg,
)
# _retrieval_with_ft.fit(utterances=[...], labels=[...])  # runs fine-tuning when ft_config is set

# %% [markdown]
"""
## Performance Tips

### 1. Leverage Automatic Device Detection
- Sentence-transformers automatically detects and uses the best available hardware
- Only override `device` if you need specific control (e.g., multi-GPU setups)
- The library handles CUDA, MPS (Apple Silicon), and CPU optimization automatically

### 2. Use Caching Effectively
- Enable `use_cache=True` for repeated experiments
- Cached embeddings are stored on disk and reused across runs
- Particularly useful during hyperparameter tuning

### 3. Optimize Batch Size
- Increase `batch_size` for faster processing
- Monitor memory usage - larger batches use more GPU/CPU memory

### 4. Choose Appropriate Sequence Length
- Longer sequences (`max_length`) provide more context but are slower
- For short texts (tweets, queries): 128-256 tokens
- For documents: 512+ tokens
- Balance accuracy vs. speed based on your use case

### 5. Select the Right Model
- **Tip**: For best results, choose a model from the [Massive Text Embedding Benchmark (MTEB) leaderboard](https://huggingface.co/spaces/mteb/leaderboard), which ranks models by quality and speed across many tasks.

### 6. Use Mixed Precision
- Enable `fp16=True` on compatible GPUs for faster inference
- Reduces memory usage without significant quality loss
- Automatically handled by sentence-transformers on supported hardware

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `batch_size`
   - Decrease `max_length`
   - For Sentence Transformers inference, enable mixed precision with **`fp16`** / **`bf16`** on `SentenceTransformerEmbeddingConfig` when your device supports it
   - For embedding fine-tuning, tune **`fp16`** / **`bf16`** on `EmbedderFineTuningConfig` instead

2. **Slow Inference**
   - Increase `batch_size` (if memory allows)
   - Use a lighter model (e.g., MiniLM instead of MPNet)
   - Reduce `max_length`
   - Ensure GPU/MPS utilization

3. **Inconsistent Results**
   - Use `use_cache=True` to avoid recomputation
   - Check if seed is set for your program
"""
