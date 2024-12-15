# %% [markdown]
"""
# Run reporting

This script demonstrates how to report the optimization process using the AutoIntent library.
"""

# %%
search_space = [
    {
        "node_type": "embedding",
        "metric": "retrieval_hit_rate",
        "search_space": [
            {
                "module_name": "retrieval",
                "k": [10],
                "embedder_name": ["avsolatorio/GIST-small-Embedding-v0", "infgrad/stella-base-en-v2"],
            }
        ],
    },
    {
        "node_type": "scoring",
        "metric": "scoring_roc_auc",
        "search_space": [
            {"module_name": "knn", "k": [1, 3, 5, 10], "weights": ["uniform", "distance", "closest"]},
            {"module_name": "linear"},
            {
                "module_name": "dnnc",
                "cross_encoder_name": ["BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L-6-v2"],
                "k": [1, 3, 5, 10],
            },
        ],
    },
    {
        "node_type": "decision",
        "metric": "decision_accuracy",
        "search_space": [{"module_name": "threshold", "thresh": [0.5]}, {"module_name": "argmax"}],
    },
]

# %% [markdown]
"""
### Load Data

Let us use small subset of popular `clinc150` dataset:
"""

# %%

from autointent import Dataset

dataset = Dataset.from_hub("AutoIntent/clinc150_subset")

# %% [markdown]
"""
### Start Auto Configuration
"""

# %%
from autointent import Pipeline

pipeline_optimizer = Pipeline.from_search_space(search_space)

# %% [markdown]
"""
## Reporting

Currently supported reporting options are:
- tensorboard
- wandb
"""
# %%
from autointent.configs import LoggingConfig
from pathlib import Path

log_config = LoggingConfig(
    run_name="test_tensorboard", report_to=["tensorboard"], dirpath=Path("test_tensorboard"), dump_modules=False
)

pipeline_optimizer.set_config(log_config)

# %%
pipeline_optimizer.fit(dataset)

# %% [markdown]
"""
Now results of the optimization process can be viewed in the tensorboard.

```bash
tensorboard --logdir test_tensorboard
```
"""
