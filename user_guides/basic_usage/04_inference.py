# %% [markdown]
"""
# Inference Pipeline

After you configured optimal pipeline with AutoIntent, you probably want to test its power on some new data! There are several options:

- use it right after optimization
- save to file system and then load

## Right After

Here's the basic example:
"""

# %%
from autointent import Dataset, Pipeline

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

dataset = Dataset.from_hub("AutoIntent/clinc150_subset")
pipeline = Pipeline.from_search_space(search_space)
context = pipeline.fit(dataset)
pipeline.predict(["hello, world!"])

# %% [markdown]
"""
There are several caveats.

**RAM usage.**

You can optimize RAM usage by saving all modules to file system. Just set the following options:
"""

# %%
from autointent.configs import LoggingConfig

logging_config = LoggingConfig(dump_modules=True, clear_ram=True)

# %% [markdown]
"""
## Load from File System

Firstly, your auto-configuration run should dump modules into file system:
"""

# %%
from autointent import Dataset, Pipeline
from autointent.configs import LoggingConfig

dataset = Dataset.from_hub("AutoIntent/clinc150_subset")
pipeline = Pipeline.from_search_space(search_space)
pipeline.set_config(LoggingConfig(dump_modules=True, clear_ram=True))

# %% [markdown]
"""
Secondly, after optimization finished, you need to save the auto-configuration results to file system:
"""

# %%
context = pipeline.fit(dataset)
context.dump()

# %% [markdown]
"""
This command saves all results to the run's directory:
"""

# %%
run_directory = context.logging_config.dirpath
run_directory

# %% [markdown]
"""
After that, you can load pipeline for inference:
"""

# %%
loaded_pipeline = Pipeline.load(run_directory)
loaded_pipeline.predict(["hello, world!"])

# %% [markdown]
"""
## That's all!
"""
