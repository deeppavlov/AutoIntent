# %% [markdown]
"""
# Embedder Configuration

This tutorial covers comprehensive embedder configuration for text classification modules in AutoIntent. Most scoring modules use embedders to convert text into vector representations, which are crucial for model performance.

## Overview

AutoIntent uses the **sentence-transformers** library under the hood to access embedding models from the Hugging Face Hub. The library automatically detects available devices (CUDA, MPS, CPU, etc.) and optimizes performance accordingly. This means you don't need to manually specify device preferences in most cases - the system will automatically use the best available hardware.

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
from autointent.configs import EmbedderConfig

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

embedder_config = EmbedderConfig(
    model_name="sentence-transformers/all-mpnet-base-v2",
    batch_size=32,
    # Device is auto-detected, but you can override if needed
    device="cuda" if torch.cuda.is_available() else "cpu",
    tokenizer_config=TokenizerConfig(max_length=512, padding=True, truncation=True),
    classification_prompt="Classify the following text: ",  # Task-specific prompt
    similarity_fn_name="cosine",
    use_cache=True,
    freeze=True,  # Freeze model parameters for consistent embeddings
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
- **`freeze`**: Freeze model parameters for consistent embeddings across runs
- **`similarity_fn_name`**: Similarity function (default: `"cosine"`; other options like `"dot"`, `"euclidean"`, `"manhattan"` are available, but we recommend keeping the default unless you have a specific reason)

## Practical Examples

### Performance-Optimized Configuration
"""

# %%
# Example: Performance-optimized configuration
perf_config = EmbedderConfig(
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
quality_config = EmbedderConfig(
    model_name="sentence-transformers/all-mpnet-base-v2",  # High-quality model
    batch_size=16,  # Smaller batch to handle longer sequences
    tokenizer_config=TokenizerConfig(max_length=512),  # Longer sequences for context
    classification_prompt="Classify the intent of this message: ",
    use_cache=True,
    freeze=True,
    similarity_fn_name="cosine",
)

scorer = LinearScorer(embedder_config=quality_config)

# %% [markdown]
"""
### Multilingual Configuration
"""

# %%
# Example: Multilingual setup
multilingual_config = EmbedderConfig(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size=32,
    tokenizer_config=TokenizerConfig(max_length=256),
    use_cache=True,
    freeze=True,
)

scorer = KNNScorer(embedder_config=multilingual_config, k=7)

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
   - Enable mixed precision (`fp16=True`) [planned to implement]

2. **Slow Inference**
   - Increase `batch_size` (if memory allows)
   - Use a lighter model (e.g., MiniLM instead of MPNet)
   - Reduce `max_length`
   - Ensure GPU/MPS utilization

3. **Inconsistent Results**
   - Set `freeze=True` for reproducible embeddings
   - Use `use_cache=True` to avoid recomputation
   - Check if seed is set for your program
"""
