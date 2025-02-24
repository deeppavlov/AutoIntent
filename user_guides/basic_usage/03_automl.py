# %% [markdown]
"""
# Pipeline Auto Configuration (AutoML)
"""

# %%
from autointent import Pipeline

# %% [markdown]
"""
In this tutorial we will walk through pipeline auto configuration process.

Let us use small subset of popular `clinc150` dataset for the demonstation.
"""

# %%
from autointent import Dataset

dataset = Dataset.from_hub("AutoIntent/clinc150_subset")
dataset

# %%
dataset["train_0"][0]


# %% [markdown]
"""
## Search Space

AutoIntent provides default search spaces for multi-label and single-label classification problems. One can utilize them by constructing %mddoclink(class,,Pipeline) with factory %mddoclink(method,Pipeline,default_optimizer):
"""

# %%
multiclass_pipeline = Pipeline.default_optimizer(multilabel=False)
multilabel_pipeline = Pipeline.default_optimizer(multilabel=True)

# %% [markdown]
"""
One can explore its contents:
"""

# %%
from pprint import pprint

from autointent.utils import load_default_search_space

search_space = load_default_search_space(multilabel=True)
pprint(search_space)

# %% [markdown]
"""
Search space is allowed to customize:
"""

# %%
search_space[1]["search_space"][0]["k"] = [1, 3]
custom_pipeline = Pipeline.from_search_space(search_space)

# %% [markdown]
"""
See tutorial %mddoclink(notebook,advanced.02_search_space_configuration) on how the search space is structured.
"""

# %% [markdown]
"""
## Logging Settings

The important thing is what assets you want to save during the pipeline auto-configuration process. You can control it with %mddoclink(class,configs,LoggingConfig):
"""

# %%
from pathlib import Path
from autointent.configs import LoggingConfig

logging_config = LoggingConfig(project_dir=Path.cwd() / "runs", dump_modules=False, clear_ram=False)
custom_pipeline.set_config(logging_config)

# %% [markdown]
"""
## Complete Example
"""

# %%
from autointent import Dataset, Pipeline
from autointent.configs import LoggingConfig
from autointent.utils import load_default_search_space

# load data
dataset = Dataset.from_hub("AutoIntent/clinc150_subset")

# customize search space
search_space = load_default_search_space(multilabel=False)

# make pipeline
custom_pipeline = Pipeline.from_search_space(search_space)

# custom settings
logging_config = LoggingConfig()

custom_pipeline.set_config(logging_config)

# start auto-configuration
custom_pipeline.fit(dataset)

# inference
custom_pipeline.predict(["hello world!"])
