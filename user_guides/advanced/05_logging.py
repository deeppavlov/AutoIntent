# %% [markdown]
"""
# Logging to stdout and file

This guide will teach you how to configure logging in AutoIntent. By default, it is fully disabled.

It will be demonstrated on toy search_space example:
"""

# %%
from pathlib import Path

from autointent import Dataset, Pipeline
from autointent.configs import LoggingConfig

search_space = [
    {
        "node_type": "scoring",
        "target_metric": "scoring_roc_auc",
        "search_space": [
            {
                "module_name": "knn",
                "k": [1],
                "weights": ["uniform"],
                "embedder_config": ["avsolatorio/GIST-small-Embedding-v0"],
            },
        ],
    },
    {
        "node_type": "decision",
        "target_metric": "decision_accuracy",
        "search_space": [
            {"module_name": "threshold", "thresh": [0.5]},
            {"module_name": "argmax"},
        ],
    },
]

log_config = LoggingConfig(dirpath=Path("logging_tutorial"))
pipeline_optimizer = Pipeline.from_search_space(search_space)
pipeline_optimizer.set_config(log_config)

dataset = Dataset.from_hub("AutoIntent/clinc150_subset")

# %% [markdown]
"""
## Fully Custom Logging

One can fully customize logging via python's standard module [`logging`](https://docs.python.org/3/library/logging.html). Everything you need to do is configure it before AutoIntent execution:
"""
# %%
import logging

logging.basicConfig(level="INFO")
pipeline_optimizer.fit(dataset)

# %% [markdown]
"""
See external tutorials and guides about `logging` module.
"""

# %% [markdown]
"""
## Export from AutoIntent

If you don't have to customize logging, you can export our configuration. Everything you need to do is setup it before AutoIntent execution:
"""

# %%
from autointent import setup_logging

setup_logging("INFO", log_filename="tests/logs/my_exp")
# %%
"""
The first parameter affects the logs to the standard output stream. The second parameter is optional. If it is specified, then the "DEBUG" messages are logged to the file, regardless of what is specified by the first parameter.
"""
